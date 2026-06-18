"""Application service: ties the recognition engine to the user database.

Responsibilities:
  - password hashing (stdlib PBKDF2 — no extra deps)
  - quality-gated enrollment from uploaded images
  - duplicate-face detection (is this face already registered to someone else?)
  - rebuild the FAISS search index from the DB (DB = source of truth)
  - run video identification and produce the per-user report

The FAISS gallery is labelled by *username* so annotated videos show names directly.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from occlubio import load_config
from occlubio.analytics import IdentityLog
from occlubio.db.models import Enrollment, User
from occlubio.gallery import FaissGallery
from occlubio.pipeline import RecognitionEngine
from occlubio.pipeline.aligner import norm_crop
from occlubio.utils import draw_results, ensure_dir, get_logger, l2_normalize

log = get_logger("face_service")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"{salt.hex()}:{dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split(":")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 200_000)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:  # noqa: BLE001
        return False


class FaceService:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.dim = self.cfg.recognition.embedding_dim
        self.engine = RecognitionEngine(self.cfg, gallery=FaissGallery(dim=self.dim))
        self.match_threshold = float(self.cfg.gallery.match_threshold)
        self.dup_threshold = float(os.environ.get("OCCLUBIO_DUP_THRESHOLD", "0.5"))
        # The insightface/onnxruntime session is NOT thread-safe. Serialize all engine use so
        # concurrent enroll/identify requests can't run inferences in parallel (memory blowup
        # + state corruption). For real parallelism, run multiple worker processes (see PLATFORM.md).
        self._lock = threading.RLock()
        an = getattr(self.cfg, "analytics", None)
        self.min_track_frames = int(getattr(an, "min_track_frames", 3)) if an else 3
        self.merge_gap_s = float(getattr(an, "merge_gap_s", 1.5)) if an else 1.5
        self.unknown_min_seconds = float(getattr(an, "unknown_min_seconds", 0.6)) if an else 0.6
        self.unknown_merge_sim = float(getattr(an, "unknown_merge_sim", 0.45)) if an else 0.45

    # ---- index sync (DB -> FAISS) -----------------------------------------
    def rebuild_index(self, session) -> int:
        with self._lock:
            g = FaissGallery(dim=self.dim)
            for e in session.query(Enrollment).all():
                emb = np.frombuffer(e.embedding, dtype=np.float32)
                g.add(emb, e.user.username, meta={"user_id": e.user_id})
            self.engine.gallery = g
            self.engine._buffers.clear()
            log.info("FAISS index rebuilt: %d enrolled users", len(g))
            return len(g)

    def startup(self, session) -> None:
        self.rebuild_index(session)

    # ---- embedding from uploaded images -----------------------------------
    def embed_images(self, images_bgr: List[np.ndarray]):
        """Return (template, n_accepted, n_total, mean_quality). Raises ValueError if unusable."""
        embs, quals = [], []
        with self._lock:
            for img in images_bgr:
                if img is None:
                    continue
                faces = self.engine.analyzer.analyze(img)
                if not faces:
                    continue
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                aligned = norm_crop(img, face.kps)
                ok, q = self.engine.quality.passes(aligned, face.det_score)
                quals.append(q)
                if ok:
                    embs.append(self.engine.embed_face(img, face))
        if not embs:
            raise ValueError(
                f"no usable face: {len(images_bgr)} image(s), none passed detection+quality "
                f"(best quality={max(quals) if quals else 0:.2f}, need >= {self.cfg.quality.min_score})"
            )
        template = l2_normalize(np.mean(np.stack(embs, 0), 0))
        return template, len(embs), len(images_bgr), float(np.mean([q for q in quals]))

    def find_duplicate(self, template: np.ndarray, exclude_username: Optional[str] = None):
        """Return {'username','user_id','score'} if this face matches a DIFFERENT user."""
        with self._lock:
            hits = self.engine.gallery.search(template, top_k=1)
        if hits:
            username, score, meta = hits[0]
            if score >= self.dup_threshold and username != exclude_username:
                return {"username": username, "user_id": meta.get("user_id"), "score": round(score, 3)}
        return None

    # ---- video identification ---------------------------------------------
    def identify_video(self, video_path: str, out_path: str, report_path: str,
                       session, stride: int = 2, min_track_frames: Optional[int] = None,
                       max_side: int = 1280) -> dict:
        ensure_dir(Path(out_path).parent)
        ensure_dir(Path(report_path).parent)
        name_to_id = {u.username: u.id for u in session.query(User).all()}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_seconds = total_frames / fps if total_frames else 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Cap resolution to bound peak memory on large/4K videos.
        scale = min(1.0, max_side / max(w, h)) if max(w, h) > max_side else 1.0
        ow, oh = int(w * scale), int(h * scale)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / max(1, stride), (ow, oh))
        if not writer.isOpened():
            cap.release()
            raise ValueError("could not open video writer — mp4v codec unavailable in this OpenCV build")

        mtf = self.min_track_frames if min_track_frames is None else min_track_frames
        logbook = IdentityLog(min_track_frames=mtf, merge_gap_s=self.merge_gap_s,
                              unknown_min_seconds=self.unknown_min_seconds,
                              unknown_merge_sim=self.unknown_merge_sim)
        processed = 0
        t_start = time.time()
        with self._lock:                       # one identify job uses the engine at a time
            self.engine._buffers.clear()
            frame_no = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_no += 1
                if frame_no % max(1, stride) != 0:
                    continue
                if scale != 1.0:
                    frame = cv2.resize(frame, (ow, oh))
                t = frame_no / fps
                results = self.engine.process_frame(frame)
                logbook.update(t, results)
                processed += 1
                vis = draw_results(frame, results)
                cv2.putText(vis, f"t={t:6.2f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                writer.write(vis)
        proc_seconds = time.time() - t_start

        cap.release()
        writer.release()
        if not Path(out_path).exists() or Path(out_path).stat().st_size == 0:
            raise ValueError("annotated video is empty (codec failure or no frames decoded)")
        report = logbook.report()
        for p in report["people"]:                      # attach DB ids to usernames
            p["username"] = p["id"]
            p["user_id"] = name_to_id.get(p["id"])
        report["processing"] = {
            "processing_seconds": round(proc_seconds, 2),
            "frames_processed": processed,
            "processing_fps": round(processed / proc_seconds, 2) if proc_seconds > 0 else 0.0,
            "video_seconds": round(video_seconds, 2),
            "stride": stride,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log.info("identify done in %.1fs: %d known, %d unknown alerts (%d frames, %.2f FPS)",
                 proc_seconds, report["summary"]["known_people"],
                 report["summary"]["unknown_alerts"], processed, report["processing"]["processing_fps"])
        return report


def decode_image(raw: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

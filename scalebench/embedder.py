"""Detection + alignment + batched embedding with a persistent disk cache.

Pipeline per image: SCRFD detect -> pick largest face -> Umeyama norm_crop to
112x112 -> batched ArcFace forward -> L2-normalize. Images where no face is
detected are logged and skipped (reported as skip count).
"""

from __future__ import annotations

import logging
import os
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("scalebench.embedder")


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


class FaceEmbedder:
    def __init__(self, model: str = "buffalo_l", det_size: int = 320,
                 batch_size: int = 32, providers: list[str] | None = None):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(
            name=model,
            allowed_modules=["detection", "recognition"],
            providers=providers or ["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))
        self.det = self.app.det_model
        self.rec = self.app.models["recognition"]
        self.batch_size = batch_size
        log.info("FaceEmbedder ready: model=%s det_size=%d batch=%d providers=%s",
                 model, det_size, batch_size, providers or ["CPUExecutionProvider"])

    def detect_align(self, img: np.ndarray) -> np.ndarray | None:
        """Return the aligned 112x112 crop of the largest face, or None."""
        from insightface.utils.face_align import norm_crop
        bboxes, kpss = self.det.detect(img, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best = int(np.argmax(areas))
        return norm_crop(img, kpss[best], image_size=112)

    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """Batched ArcFace forward over aligned 112x112 crops -> [B,512] unit."""
        feats = self.rec.get_feat(crops)
        return l2_normalize(np.asarray(feats, dtype=np.float32))

    def embed_paths(self, paths: list[str],
                    progress_every: int = 500) -> tuple[dict[str, np.ndarray], list[str]]:
        """Embed images from disk. Returns ({path: emb[512]}, skipped_paths)."""
        out: dict[str, np.ndarray] = {}
        skipped: list[str] = []
        pending_paths: list[str] = []
        pending_crops: list[np.ndarray] = []
        t0 = time.perf_counter()

        def flush():
            if not pending_crops:
                return
            embs = self.embed_crops(pending_crops)
            for p, e in zip(pending_paths, embs):
                out[p] = e
            pending_paths.clear()
            pending_crops.clear()

        for i, path in enumerate(paths):
            img = cv2.imread(path)
            if img is None:
                skipped.append(path)
                continue
            crop = self.detect_align(img)
            if crop is None:
                skipped.append(path)
                continue
            pending_paths.append(path)
            pending_crops.append(crop)
            if len(pending_crops) >= self.batch_size:
                flush()
            if progress_every and (i + 1) % progress_every == 0:
                rate = (i + 1) / (time.perf_counter() - t0)
                log.info("embedded %d/%d images (%.1f img/s, %d skipped)",
                         i + 1, len(paths), rate, len(skipped))
        flush()
        return out, skipped

    def embed_single_timed(self, path: str) -> tuple[np.ndarray | None, dict]:
        """One image through the full path with per-stage wall times (seconds)."""
        t = {}
        t0 = time.perf_counter()
        img = cv2.imread(path)
        t["read"] = time.perf_counter() - t0
        if img is None:
            return None, t
        t0 = time.perf_counter()
        crop = self.detect_align(img)
        t["detect_align"] = time.perf_counter() - t0
        if crop is None:
            return None, t
        t0 = time.perf_counter()
        emb = self.embed_crops([crop])[0]
        t["embed"] = time.perf_counter() - t0
        return emb, t


class EmbeddingCache:
    """npz-backed {path: emb} cache so reruns skip already-embedded images."""

    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.embs: dict[str, np.ndarray] = {}
        self.skipped: set[str] = set()
        if os.path.exists(cache_file):
            z = np.load(cache_file, allow_pickle=False)
            files = [str(f) for f in z["files"]]
            self.embs = dict(zip(files, z["embs"]))
            self.skipped = {str(s) for s in z["skipped"]}
            log.info("embedding cache: %d embeddings, %d known-skipped",
                     len(self.embs), len(self.skipped))

    def missing(self, paths: list[str]) -> list[str]:
        return [p for p in paths if p not in self.embs and p not in self.skipped]

    def add(self, embs: dict[str, np.ndarray], skipped: list[str]) -> None:
        self.embs.update(embs)
        self.skipped.update(skipped)

    def save(self) -> None:
        files = list(self.embs.keys())
        mat = (np.stack([self.embs[f] for f in files]).astype(np.float32)
               if files else np.zeros((0, 512), np.float32))
        np.savez_compressed(
            self.cache_file, files=np.array(files), embs=mat,
            skipped=np.array(sorted(self.skipped)),
        )

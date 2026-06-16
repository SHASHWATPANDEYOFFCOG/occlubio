"""RecognitionEngine — orchestrates the full per-frame flow.

   detect -> track -> (per face) align -> quality gate -> occlusion -> anti-spoof
          -> embed -> per-track embedding fusion -> 1:N gallery identify

Design principle (architecture §1): heavy recognition runs once per *track* on the best,
quality-gated, live frame; embeddings are averaged over a track for set-based robustness.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import List, Optional

import numpy as np

from occlubio.gallery import FaissGallery
from occlubio.pipeline.aligner import norm_crop
from occlubio.pipeline.antispoof import AntiSpoof
from occlubio.pipeline.face_analyzer import FaceAnalyzer, FaceResult
from occlubio.pipeline.occlusion import OcclusionEstimator
from occlubio.pipeline.quality import FaceQualityGate
from occlubio.tracking import IoUTracker
from occlubio.utils import get_logger, l2_normalize

log = get_logger(__name__)


class RecognitionEngine:
    def __init__(self, cfg, gallery: Optional[FaissGallery] = None):
        self.cfg = cfg
        custom = getattr(cfg.recognition, "custom_onnx", None)

        self.analyzer = FaceAnalyzer(
            model_name=cfg.detection.model_name,
            providers=list(cfg.device.providers),
            ctx_id=cfg.device.ctx_id,
            det_size=tuple(cfg.detection.det_size),
            score_thresh=cfg.detection.score_thresh,
            min_face=cfg.detection.min_face,
            detection_only=bool(custom),
        )

        self.recognizer = None
        if custom:
            from occlubio.pipeline.recognizer import CustomRecognizer

            self.recognizer = CustomRecognizer(custom, providers=list(cfg.device.providers))

        self.quality = FaceQualityGate(cfg)
        self.occlusion = OcclusionEstimator(cfg.occlusion.enabled)
        self.antispoof = AntiSpoof(cfg)
        self.tracker = IoUTracker(
            iou_thresh=cfg.tracking.iou_thresh,
            max_age=cfg.tracking.max_age,
            min_hits=cfg.tracking.min_hits,
        ) if cfg.tracking.enabled else None

        self.gallery = gallery or FaissGallery.load_or_new(
            cfg.gallery.path, dim=cfg.recognition.embedding_dim
        )
        self.match_threshold = float(cfg.gallery.match_threshold)
        self._buffers = defaultdict(lambda: deque(maxlen=int(cfg.gallery.per_track_buffer)))

    # ----------------------------------------------------------------------
    def _embed(self, img: np.ndarray, face: FaceResult, aligned: np.ndarray) -> np.ndarray:
        if self.recognizer is not None:
            return self.recognizer.embed(aligned)
        if face.embedding is not None:
            return face.embedding
        raise RuntimeError("No embedding available — enable recognition model or set custom_onnx.")

    def _fuse_track(self, track_id: int, emb: np.ndarray) -> np.ndarray:
        buf = self._buffers[track_id]
        buf.append(emb)
        return l2_normalize(np.mean(np.stack(buf, axis=0), axis=0))

    def process_frame(self, img: np.ndarray) -> List[FaceResult]:
        faces = self.analyzer.analyze(img)
        if self.tracker is not None:
            faces = self.tracker.update(faces)

        for f in faces:
            aligned = norm_crop(img, f.kps)

            passed, f.quality = self.quality.passes(aligned, f.det_score)
            f.occlusion = self.occlusion.estimate(aligned)

            spoof = self.antispoof.check(aligned)
            f.live, f.spoof_score = spoof["live"], spoof["score"]

            if not (passed and f.live):
                continue  # leave as "unknown"; not worth a gallery query

            emb = self._embed(img, f, aligned)
            if f.track_id is not None:
                emb = self._fuse_track(f.track_id, emb)
            f.embedding = emb  # expose the fused/representative embedding for analytics clustering
            f.identity, f.score = self.gallery.identify(emb, self.match_threshold)

        return faces

    # convenience for single-image use (no tracking/fusion)
    def recognize_image(self, img: np.ndarray) -> List[FaceResult]:
        saved = self.tracker
        self.tracker = None
        try:
            return self.process_frame(img)
        finally:
            self.tracker = saved

    def embed_face(self, img: np.ndarray, face: FaceResult) -> np.ndarray:
        """Public helper used by enrollment."""
        aligned = norm_crop(img, face.kps)
        return self._embed(img, face, aligned)

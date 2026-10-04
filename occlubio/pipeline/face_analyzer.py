from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from occlubio.utils import get_logger, l2_normalize

log = get_logger(__name__)


@dataclass
class FaceResult:
    bbox: np.ndarray
    kps: np.ndarray
    det_score: float
    embedding: Optional[np.ndarray] = None
    track_id: Optional[int] = None
    quality: Optional[float] = None
    occlusion: Optional[dict] = None
    live: bool = True
    spoof_score: float = 1.0
    identity: str = "unknown"
    score: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def wh(self):
        return self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]


class FaceAnalyzer:

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: Optional[List[str]] = None,
        ctx_id: int = 0,
        det_size=(640, 640),
        score_thresh: float = 0.5,
        min_face: int = 24,
        detection_only: bool = False,
    ):
        from insightface.app import FaceAnalysis

        from occlubio.platform_support import onnx_providers

        providers = onnx_providers(providers)
        if "CUDAExecutionProvider" not in providers:
            ctx_id = -1

        allowed = ["detection"] if detection_only else ["detection", "recognition"]
        self.app = FaceAnalysis(name=model_name, allowed_modules=allowed, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=tuple(det_size), det_thresh=score_thresh)
        self.detection_only = detection_only
        self.min_face = min_face
        log.info(
            "FaceAnalyzer ready: model=%s detection_only=%s providers=%s",
            model_name, detection_only, providers,
        )

    def analyze(self, img_bgr: np.ndarray) -> List[FaceResult]:
        faces = self.app.get(img_bgr)
        results: List[FaceResult] = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            if max(x2 - x1, y2 - y1) < self.min_face:
                continue
            emb = None
            if not self.detection_only and getattr(f, "embedding", None) is not None:
                emb = l2_normalize(f.embedding)
            results.append(
                FaceResult(
                    bbox=np.asarray(f.bbox, dtype=np.float32),
                    kps=np.asarray(f.kps, dtype=np.float32),
                    det_score=float(f.det_score),
                    embedding=emb,
                )
            )
        return results

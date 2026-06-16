"""Face image quality gate.

This is a *proxy* FIQA combining detector confidence, sharpness (variance of Laplacian),
and exposure. It is intentionally dependency-light so the baseline runs anywhere.

HOOK: for SOTA, replace `score()` with CR-FIQA (CVPR'23, fdbtrs/CR-FIQA) or CLIB-FIQA
(CVPR'24). Both return a learned usability scalar; keep the same [0,1] contract so the
rest of the pipeline is unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

from occlubio.utils import variance_of_laplacian


class FaceQualityGate:
    def __init__(self, cfg):
        q = cfg.quality
        self.enabled = q.enabled
        self.min_score = float(q.min_score)
        self.blur_min = float(q.blur_min)
        self.bright_min = float(q.brightness_min)
        self.bright_max = float(q.brightness_max)

    def score(self, aligned_crop: np.ndarray, det_score: float) -> float:
        """Return a usability score in [0,1] for an aligned face crop."""
        gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)

        sharp = variance_of_laplacian(gray)
        sharp_score = float(np.clip(sharp / (self.blur_min * 3.0), 0.0, 1.0))

        brightness = float(gray.mean())
        if self.bright_min <= brightness <= self.bright_max:
            bright_score = 1.0
        else:
            # linear falloff outside the comfortable exposure band
            dist = min(abs(brightness - self.bright_min), abs(brightness - self.bright_max))
            bright_score = float(np.clip(1.0 - dist / 60.0, 0.0, 1.0))

        det = float(np.clip(det_score, 0.0, 1.0))
        return float(0.5 * det + 0.3 * sharp_score + 0.2 * bright_score)

    def passes(self, aligned_crop: np.ndarray, det_score: float) -> tuple[bool, float]:
        s = self.score(aligned_crop, det_score)
        return (not self.enabled or s >= self.min_score), s

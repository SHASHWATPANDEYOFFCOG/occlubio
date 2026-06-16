"""Occlusion-type estimation on an aligned 112x112 face.

Heuristic baseline: measure skin coverage in the eye band vs the lower-face band using a
YCrCb skin model. It routes recognition (mask -> rely on periocular; sunglasses -> rely on
lower face) and flags heavy occlusion.

HOOK: replace `estimate()` with a trained MobileNetV3-small classifier
(classes: clear/mask/sunglasses/cap/scarf/profile) for reliable routing, and feed a
per-pixel parsing map (BiSeNet / FaRL) into the FR feature-masking head (see architecture §1.5/1.8).
"""
from __future__ import annotations

import cv2
import numpy as np

# Bands in the canonical 112x112 aligned frame.
_EYE_BAND = (slice(40, 64), slice(16, 96))      # rows, cols
_LOWER_BAND = (slice(70, 112), slice(16, 96))
_FOREHEAD_BAND = (slice(0, 38), slice(20, 92))


def _skin_ratio(crop_bgr: np.ndarray, band) -> float:
    region = crop_bgr[band]
    ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    mask = (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)
    return float(mask.mean())


class OcclusionEstimator:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def estimate(self, aligned_crop: np.ndarray) -> dict:
        if not self.enabled:
            return {"type": "clear", "occluded_ratio": 0.0}

        eye_skin = _skin_ratio(aligned_crop, _EYE_BAND)
        lower_skin = _skin_ratio(aligned_crop, _LOWER_BAND)
        fore_skin = _skin_ratio(aligned_crop, _FOREHEAD_BAND)

        # eye band darkness (sunglasses tend to be dark, low-skin)
        eye_gray = cv2.cvtColor(aligned_crop[_EYE_BAND], cv2.COLOR_BGR2GRAY).mean()

        occ_type = "clear"
        ratio = 0.0
        if lower_skin < 0.25:                       # mouth/chin covered
            occ_type, ratio = "lower_face", 1.0 - lower_skin   # mask / scarf
        if eye_skin < 0.20 and eye_gray < 70:        # eyes dark & covered
            occ_type = "upper_face" if occ_type == "clear" else "heavy"
            ratio = max(ratio, 1.0 - eye_skin)        # sunglasses
        if fore_skin < 0.15 and occ_type == "clear":
            occ_type, ratio = "cap", 1.0 - fore_skin

        return {
            "type": occ_type,
            "occluded_ratio": round(float(ratio), 3),
            "use_periocular": occ_type in ("lower_face", "heavy"),
        }

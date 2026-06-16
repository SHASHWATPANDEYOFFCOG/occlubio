"""Shared helpers: logging, numpy ops, visualization."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def get_logger(name: str = "occlubio", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Sharpness proxy: higher = sharper. Low values indicate motion/defocus blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---- visualization ---------------------------------------------------------

_COLOR_KNOWN = (60, 200, 60)
_COLOR_UNKNOWN = (60, 160, 230)
_COLOR_SPOOF = (40, 40, 230)


def draw_results(img: np.ndarray, results: Iterable, draw_kps: bool = False) -> np.ndarray:
    """Draw bbox + label for each FaceResult on a copy of img (BGR)."""
    out = img.copy()
    for f in results:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        if getattr(f, "live", True) is False:
            color, tag = _COLOR_SPOOF, "SPOOF"
        elif getattr(f, "identity", "unknown") not in (None, "unknown"):
            color, tag = _COLOR_KNOWN, f.identity
        else:
            color, tag = _COLOR_UNKNOWN, "unknown"

        score = getattr(f, "score", 0.0) or 0.0
        tid = getattr(f, "track_id", None)
        occ = getattr(f, "occlusion", None)
        occ_txt = f" [{occ['type']}]" if isinstance(occ, dict) and occ.get("type") not in (None, "clear") else ""
        label = f"{tag} {score:.2f}{occ_txt}"
        if tid is not None:
            label = f"#{tid} {label}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if draw_kps and getattr(f, "kps", None) is not None:
            for (kx, ky) in f.kps:
                cv2.circle(out, (int(kx), int(ky)), 1, (0, 0, 255), 2)
    return out

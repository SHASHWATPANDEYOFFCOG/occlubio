from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np

_MASK_POLY = np.array([[18, 66], [94, 66], [102, 90], [80, 112], [32, 112], [10, 90]], np.int32)
_SCARF_POLY = np.array([[8, 78], [104, 78], [112, 112], [0, 112]], np.int32)
_GLASSES_RECT = (16, 42, 96, 64)
_CAP_RECT = (0, 0, 112, 36)

_MASK_COLORS = [(230, 230, 230), (210, 180, 150), (90, 90, 90), (160, 200, 230)]


def _overlay_poly(img, poly, color, alpha=1.0, noise=8):
    out = img.copy()
    layer = img.copy()
    cv2.fillPoly(layer, [poly], color)
    if noise:
        n = np.random.randint(-noise, noise + 1, img.shape, dtype=np.int16)
        layer = np.clip(layer.astype(np.int16) + n, 0, 255).astype(np.uint8)
    cv2.fillPoly(out, [poly], (1, 1, 1))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    m3 = (mask[..., None] > 0)
    return np.where(m3, cv2.addWeighted(layer, alpha, img, 1 - alpha, 0), img)


def add_mask(img):
    return _overlay_poly(img, _MASK_POLY, random.choice(_MASK_COLORS))


def add_scarf(img):
    return _overlay_poly(img, _SCARF_POLY, random.choice([(60, 40, 120), (40, 40, 40), (30, 80, 120)]))


def add_sunglasses(img):
    out = img.copy()
    x1, y1, x2, y2 = _GLASSES_RECT
    cv2.rectangle(out, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.line(out, (x1, (y1 + y2) // 2), (0, (y1 + y2) // 2), (20, 20, 20), 3)
    cv2.line(out, (x2, (y1 + y2) // 2), (112, (y1 + y2) // 2), (20, 20, 20), 3)
    return out


def add_cap(img):
    out = img.copy()
    x1, y1, x2, y2 = _CAP_RECT
    color = random.choice([(40, 40, 40), (120, 40, 40), (30, 60, 30)])
    cv2.rectangle(out, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(out, (x1, y2), (x2, y2 + 6), tuple(int(c * 0.7) for c in color), -1)
    return out


def random_erase(img, max_frac=0.3):
    out = img.copy()
    h, w = img.shape[:2]
    ew, eh = int(w * random.uniform(0.1, max_frac)), int(h * random.uniform(0.1, max_frac))
    x, y = random.randint(0, w - ew), random.randint(0, h - eh)
    out[y:y + eh, x:x + ew] = np.random.randint(0, 255, (eh, ew, 3), np.uint8)
    return out


def motion_blur(img, k=None):
    k = k or random.choice([5, 7, 9])
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    if random.random() < 0.5:
        kernel = kernel.T
    return cv2.filter2D(img, -1, kernel)


def low_light(img):
    g = random.uniform(0.35, 0.7)
    out = np.clip((img / 255.0) ** (1.0 / g) * 255.0 * g, 0, 255).astype(np.uint8)
    noise = np.random.normal(0, random.uniform(4, 12), img.shape)
    return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg(img, q=None):
    q = q or random.randint(25, 60)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def downscale(img, factor=None):
    factor = factor or random.uniform(0.3, 0.6)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * factor)), max(8, int(h * factor))))
    return cv2.resize(small, (w, h))


_OCCLUDERS = [add_mask, add_sunglasses, add_cap, add_scarf, random_erase]
_PHOTOMETRIC = [motion_blur, low_light, jpeg, downscale]


class OcclusionAugmentor:

    def __init__(self, occlude_prob: float = 0.5, photometric_prob: float = 0.5, seed: Optional[int] = None):
        self.occlude_prob = occlude_prob
        self.photometric_prob = photometric_prob
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        out = img
        if random.random() < self.occlude_prob:
            out = random.choice(_OCCLUDERS)(out)
        if random.random() < self.photometric_prob:
            out = random.choice(_PHOTOMETRIC)(out)
        return out

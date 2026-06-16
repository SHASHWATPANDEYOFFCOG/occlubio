"""5-point similarity-transform alignment to the canonical ArcFace template.

Alignment is the single highest-leverage step for occluded FR — keep this routine
*identical* between training and inference, or accuracy silently collapses.
"""
from __future__ import annotations

import cv2
import numpy as np

# Canonical 5-point destination template for a 112x112 aligned face
# (left eye, right eye, nose, left mouth corner, right mouth corner).
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation+uniform scale+translation).

    This is the Umeyama (1991) solution used by skimage.SimilarityTransform / InsightFace.
    Returns a 3x3 homogeneous matrix. No RANSAC, no OpenCV version dependence.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.full((dim + 1, dim + 1), np.nan)
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ Vt
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt

    scale = (S @ d) / src_demean.var(axis=0).sum()
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale
    return T


def estimate_norm(kps: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Estimate the 2x3 affine matrix mapping the 5 landmarks to the template."""
    assert kps.shape == (5, 2), f"expected 5x2 landmarks, got {kps.shape}"
    dst = ARCFACE_DST.copy()
    if image_size != 112:
        dst = dst * (image_size / 112.0)
    T = _umeyama(kps.astype(np.float64), dst.astype(np.float64))
    return T[:2, :].astype(np.float32)


def norm_crop(img: np.ndarray, kps: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Return an aligned BGR crop of `image_size` x `image_size`."""
    M = estimate_norm(kps, image_size)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

"""Aligned-face dataset with on-the-fly occlusion augmentation.

Expects an ImageFolder layout of *aligned* 112x112 crops:
    root/<identity_id>/<image>.jpg

(Most FR training sets — MS1MV3, WebFace4M — ship pre-aligned. If yours are not aligned,
run them through occlubio.pipeline.aligner.norm_crop first.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _scan(root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for c in classes:
        for p in (root / c).iterdir():
            if p.suffix.lower() in _IMG_EXT:
                samples.append((p, cls_to_idx[c]))
    if not samples:
        raise RuntimeError(f"no images under {root} (expected root/<id>/<img>)")
    return samples, classes


class AlignedFaceDataset(Dataset):
    def __init__(self, root: str, image_size: int = 112, augmentor: Optional[Callable] = None, train: bool = True):
        self.root = Path(root)
        self.image_size = image_size
        self.augmentor = augmentor
        self.train = train
        self.samples, self.classes = _scan(self.root)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read {path}")
        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size))

        if self.train:
            if self.augmentor is not None:
                img = self.augmentor(img)
            if np.random.rand() < 0.5:        # horizontal flip
                img = img[:, ::-1]

        x = img.astype(np.float32)
        x = (x - 127.5) / 128.0               # MUST match inference preprocessing
        x = np.ascontiguousarray(np.transpose(x, (2, 0, 1)))
        return torch.from_numpy(x), label

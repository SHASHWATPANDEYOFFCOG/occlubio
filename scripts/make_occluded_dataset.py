"""Generate a synthetically-occluded copy of a face dataset (for training/eval).

Aligns each image (if landmarks are found) then applies occlusion + photometric augmentation.
Preserves the root/<identity>/<img> structure so it drops into training/eval directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from occlubio import load_config
from occlubio.data import OcclusionAugmentor
from occlubio.pipeline.aligner import norm_crop
from occlubio.pipeline.face_analyzer import FaceAnalyzer
from occlubio.utils import ensure_dir, get_logger

log = get_logger("make_occluded")
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root/<identity>/<img>")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--occlude-prob", type=float, default=1.0)
    ap.add_argument("--photometric-prob", type=float, default=0.5)
    ap.add_argument("--no-align", action="store_true", help="assume images are already aligned 112x112")
    args = ap.parse_args()

    cfg = load_config(args.config)
    aug = OcclusionAugmentor(occlude_prob=args.occlude_prob, photometric_prob=args.photometric_prob)
    analyzer = None if args.no_align else FaceAnalyzer(
        model_name=cfg.detection.model_name, providers=list(cfg.device.providers),
        ctx_id=cfg.device.ctx_id, det_size=tuple(cfg.detection.det_size), detection_only=True,
    )

    src, dst = Path(args.src), Path(args.dst)
    n = 0
    for img_path in src.rglob("*"):
        if img_path.suffix.lower() not in _IMG_EXT:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if analyzer is not None:
            faces = analyzer.analyze(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            crop = norm_crop(img, face.kps)
        else:
            crop = cv2.resize(img, (112, 112))
        out_crop = aug(crop)
        rel = img_path.relative_to(src)
        out_path = dst / rel
        ensure_dir(out_path.parent)
        cv2.imwrite(str(out_path), out_crop)
        n += 1
        if n % 200 == 0:
            log.info("processed %d", n)

    log.info("done: %d occluded images -> %s", n, dst)


if __name__ == "__main__":
    main()

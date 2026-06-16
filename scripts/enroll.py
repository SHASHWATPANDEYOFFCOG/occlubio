"""Enrollment (registration) phase.

Register people from one or more face images, with FACE QUALITY ASSESSMENT so only clear,
valid samples are stored. Multiple accepted images per person are averaged into one robust
template (set-based enrollment), then written to the searchable FAISS gallery.

Layout:  data/enroll/<person_name>/*.jpg   (one folder per identity, 1+ images each)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from occlubio import load_config
from occlubio.gallery import FaissGallery
from occlubio.pipeline import RecognitionEngine
from occlubio.pipeline.aligner import norm_crop
from occlubio.utils import get_logger, l2_normalize

log = get_logger("enroll")
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="root/<person>/<img> reference folder")
    ap.add_argument("--gallery", default=None, help="output gallery dir (default: config gallery.path)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--min-quality", type=float, default=None, help="override config quality.min_score")
    ap.add_argument("--allow-low-quality", action="store_true", help="enroll even if all samples fail the gate")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.min_quality is not None:
        cfg.quality.min_score = args.min_quality
    gallery_path = args.gallery or cfg.gallery.path
    engine = RecognitionEngine(cfg, gallery=FaissGallery.load_or_new(gallery_path, cfg.recognition.embedding_dim))

    root = Path(args.images)
    # per person -> list of (embedding, quality, passed)
    per_person = defaultdict(list)
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img_path in person_dir.iterdir():
            if img_path.suffix.lower() not in _IMG_EXT:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("unreadable: %s", img_path)
                continue
            faces = engine.analyzer.analyze(img)
            if not faces:
                log.warning("no face detected: %s", img_path)
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            aligned = norm_crop(img, face.kps)
            passed, q = engine.quality.passes(aligned, face.det_score)
            emb = engine.embed_face(img, face)
            per_person[person_dir.name].append((emb, q, passed))
            if not passed:
                log.warning("low quality (%.2f) skipped: %s", q, img_path.name)

    n_enrolled = 0
    for person, items in per_person.items():
        accepted = [emb for emb, q, ok in items if ok]
        if not accepted:
            if args.allow_low_quality:
                best = max(items, key=lambda it: it[1])
                accepted = [best[0]]
                log.warning("%s: no sample passed quality gate; enrolling best (q=%.2f)", person, best[1])
            else:
                log.error("%s: NO sample passed quality gate -> NOT enrolled (use --allow-low-quality to force)", person)
                continue
        template = l2_normalize(np.mean(np.stack(accepted, 0), 0))
        engine.gallery.add(template, person, meta={"n_images": len(items), "n_accepted": len(accepted)})
        n_enrolled += 1
        log.info("enrolled %-20s  accepted %d/%d images", person, len(accepted), len(items))

    engine.gallery.save(gallery_path)
    log.info("done: %d identities enrolled, gallery=%s (total entries: %d)",
             n_enrolled, gallery_path, len(engine.gallery))


if __name__ == "__main__":
    main()

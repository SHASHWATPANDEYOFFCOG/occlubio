"""Run 1:N recognition on a single image and (optionally) save an annotated copy."""
from __future__ import annotations

import argparse

import cv2

from occlubio import load_config
from occlubio.gallery import FaissGallery
from occlubio.pipeline import RecognitionEngine
from occlubio.utils import draw_results, get_logger

log = get_logger("recognize_image")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="path to save annotated image")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gallery_path = args.gallery or cfg.gallery.path
    engine = RecognitionEngine(cfg, gallery=FaissGallery.load_or_new(gallery_path, cfg.recognition.embedding_dim))

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read image: {args.image}")

    results = engine.recognize_image(img)
    for f in results:
        log.info("id=%s score=%.3f quality=%.2f occ=%s live=%s",
                 f.identity, f.score, f.quality or 0.0,
                 (f.occlusion or {}).get("type"), f.live)

    if args.out:
        cv2.imwrite(args.out, draw_results(img, results))
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()

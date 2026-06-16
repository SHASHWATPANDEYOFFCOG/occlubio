"""Latency/throughput benchmark on YOUR hardware (always measure on target, not on the trainer).

Reports per-frame p50/p95/p99 and FPS for the full pipeline.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from occlubio import load_config
from occlubio.gallery import FaissGallery
from occlubio.pipeline import RecognitionEngine
from occlubio.utils import get_logger

log = get_logger("benchmark")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="video/rtsp/0; omit to use synthetic frames")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--config", default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    cfg = load_config(args.config)
    engine = RecognitionEngine(cfg, gallery=FaissGallery.load_or_new(cfg.gallery.path, cfg.recognition.embedding_dim))

    cap = None
    if args.source is not None:
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)

    def next_frame():
        if cap is not None:
            ok, f = cap.read()
            return f if ok else None
        return np.random.randint(0, 255, (args.height, args.width, 3), np.uint8)

    times = []
    for i in range(args.frames + args.warmup):
        frame = next_frame()
        if frame is None:
            break
        t = time.perf_counter()
        engine.process_frame(frame)
        dt = (time.perf_counter() - t) * 1000.0
        if i >= args.warmup:
            times.append(dt)

    if cap is not None:
        cap.release()
    if not times:
        raise SystemExit("no frames processed")

    arr = np.array(times)
    log.info("frames=%d  p50=%.1f ms  p95=%.1f ms  p99=%.1f ms  mean=%.1f ms  -> %.1f FPS",
             len(arr), np.percentile(arr, 50), np.percentile(arr, 95),
             np.percentile(arr, 99), arr.mean(), 1000.0 / arr.mean())


if __name__ == "__main__":
    main()

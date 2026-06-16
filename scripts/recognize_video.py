"""Real-time 1:N recognition over webcam / RTSP / video file.

  --source 0                          webcam
  --source rtsp://user:pass@ip/stream IP camera
  --source clip.mp4                   file
"""
from __future__ import annotations

import argparse
import time

import cv2

from occlubio import load_config
from occlubio.gallery import FaissGallery
from occlubio.pipeline import RecognitionEngine
from occlubio.utils import draw_results, get_logger

log = get_logger("recognize_video")


def _open(source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="0 | rtsp://... | path.mp4")
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="optional output video path")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (CPU speedup)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gallery_path = args.gallery or cfg.gallery.path
    engine = RecognitionEngine(cfg, gallery=FaissGallery.load_or_new(gallery_path, cfg.recognition.embedding_dim))

    cap = _open(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")

    writer = None
    if args.out:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n, t0, ema_ms = 0, time.time(), None
    seen_ids: set[str] = set()
    frame_no = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % max(1, args.stride) != 0:   # skip frames for CPU speedup
            continue
        t = time.time()
        results = engine.process_frame(frame)
        seen_ids.update(f.identity for f in results if f.identity != "unknown")
        dt = (time.time() - t) * 1000.0
        ema_ms = dt if ema_ms is None else 0.9 * ema_ms + 0.1 * dt
        n += 1

        if cfg.engine.draw or writer or not args.no_display:
            vis = draw_results(frame, results)
            cv2.putText(vis, f"{ema_ms:.1f} ms/frame  {1000.0/max(ema_ms,1e-3):.1f} FPS",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if writer:
                writer.write(vis)
            if not args.no_display:
                cv2.imshow("occlubio", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    elapsed = max(time.time() - t0, 1e-6)
    log.info("processed %d frames in %.1fs (avg %.1f FPS)", n, elapsed, n / elapsed)
    log.info("identities recognized: %s", sorted(seen_ids) if seen_ids else "(none — empty gallery or all unknown)")
    if args.out:
        log.info("annotated video written -> %s", args.out)


if __name__ == "__main__":
    main()

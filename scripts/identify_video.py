"""Video-based identification (inference) phase — the production deliverable.

Input:  a video file (one or many people).
Output: an annotated MP4 (boxes + names + confidence + track IDs) AND a JSON report with,
per person: id, match confidence, first/last-seen timestamp, number of appearances; plus
unknown-person alerts.

Usage:
  python scripts/identify_video.py --source data/my_video.mp4 \
      --gallery gallery_store --out data/my_output.mp4 --report data/my_report.json --stride 2
"""
from __future__ import annotations

import argparse
import time

import cv2

from occlubio import load_config
from occlubio.analytics import IdentityLog
from occlubio.gallery import FaissGallery
from occlubio.pipeline import RecognitionEngine
from occlubio.utils import draw_results, get_logger

log = get_logger("identify_video")


def _open(source: str):
    return cv2.VideoCapture(int(source) if source.isdigit() else source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file / rtsp / webcam index")
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="annotated output video path")
    ap.add_argument("--report", default=None, help="JSON report path")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (CPU speedup)")
    ap.add_argument("--min-track-frames", type=int, default=3, help="drop tracks shorter than this")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gallery_path = args.gallery or cfg.gallery.path
    engine = RecognitionEngine(cfg, gallery=FaissGallery.load_or_new(gallery_path, cfg.recognition.embedding_dim))
    log.info("gallery '%s' has %d enrolled identities", gallery_path, len(engine.gallery))

    cap = _open(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if args.out:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / max(1, args.stride), (w, h))

    an = getattr(cfg, "analytics", None)
    logbook = IdentityLog(
        min_track_frames=args.min_track_frames,
        merge_gap_s=float(getattr(an, "merge_gap_s", 1.5)) if an else 1.5,
        unknown_min_seconds=float(getattr(an, "unknown_min_seconds", 0.6)) if an else 0.6,
        unknown_merge_sim=float(getattr(an, "unknown_merge_sim", 0.45)) if an else 0.45,
    )
    frame_no, processed, t0 = -1, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % max(1, args.stride) != 0:
            continue
        t = frame_no / fps  # appearance timestamp (seconds)
        results = engine.process_frame(frame)
        logbook.update(t, results)
        processed += 1

        if writer or not args.no_display:
            vis = draw_results(frame, results)
            cv2.putText(vis, f"t={t:6.2f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if writer:
                writer.write(vis)
            if not args.no_display:
                cv2.imshow("identify_video", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    report_path = args.report or "identification_report.json"
    report = logbook.save(report_path)

    # ---- pretty console summary (what you show your professor) ----
    elapsed = time.time() - t0
    log.info("processed %d frames in %.1fs (%.2f FPS)", processed, elapsed, processed / max(elapsed, 1e-6))
    log.info("=" * 64)
    log.info("IDENTIFICATION REPORT  (known=%d  unknown_alerts=%d)",
             report["summary"]["known_people"], report["summary"]["unknown_alerts"])
    log.info("%-16s %5s %10s %10s %8s", "PERSON", "APPS", "FIRST", "LAST", "CONF")
    for p in report["people"]:
        log.info("%-16s %5d %10s %10s %8.3f",
                 p["id"], p["appearances"], p["first_seen"], p["last_seen"], p["max_confidence"])
    for a in report["unknown_alerts"]:
        log.info("[ALERT] %s  appearances=%d  %s -> %s (%.2fs)",
                 a["alert"], a["appearances"], a["first_seen"], a["last_seen"], a["visible_s"])
    log.info("=" * 64)
    log.info("JSON report -> %s", report_path)
    if args.out:
        log.info("annotated video -> %s", args.out)


if __name__ == "__main__":
    main()

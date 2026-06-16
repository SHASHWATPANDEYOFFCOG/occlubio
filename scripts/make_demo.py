"""Generate a SELF-CONTAINED demo so you can verify the full video pipeline with zero
external assets:

  - data/demo_input.mp4         a short video (a panning shot of insightface's sample faces)
  - data/enroll/person_{0,1,2}/ enrollment crops of the 3 largest faces

Then run enroll.py + recognize_video.py on these. For the REAL professor demo, replace
data/demo_input.mp4 with your own clip and data/enroll/<name>/ with your own photos.
"""
from __future__ import annotations

import cv2
import numpy as np
from insightface.data import get_image

from occlubio.pipeline.face_analyzer import FaceAnalyzer
from occlubio.utils import ensure_dir, get_logger

log = get_logger("make_demo")


def main():
    img = get_image("t1")  # bundled BGR sample with several faces
    H, W = img.shape[:2]
    fa = FaceAnalyzer(model_name="buffalo_l", ctx_id=-1)
    faces = fa.analyze(img)
    faces.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    log.info("sample image %dx%d, %d faces detected", W, H, len(faces))

    # 1) enrollment crops for the 3 largest faces (with margin so they re-detect cleanly)
    enroll_root = ensure_dir("data/enroll")
    for i, f in enumerate(faces[:3]):
        x1, y1, x2, y2 = f.bbox
        m = 0.4 * max(x2 - x1, y2 - y1)
        cx1, cy1 = max(0, int(x1 - m)), max(0, int(y1 - m))
        cx2, cy2 = min(W, int(x2 + m)), min(H, int(y2 + m))
        d = ensure_dir(enroll_root / f"person_{i}")
        cv2.imwrite(str(d / "0.jpg"), img[cy1:cy2, cx1:cx2])
    log.info("wrote enrollment crops -> %s/person_0..2", enroll_root)

    # 2) input video: pan the frame so the tracker has motion to follow
    out_path = "data/demo_input.mp4"
    fps, frames = 15, 60
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(frames):
        dx = int(25 * np.sin(t / frames * 2 * np.pi))
        dy = int(10 * np.cos(t / frames * 2 * np.pi))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        writer.write(cv2.warpAffine(img, M, (W, H), borderValue=0))
    writer.release()
    log.info("wrote demo video -> %s (%d frames @ %d fps)", out_path, frames, fps)
    log.info("next: enroll then recognize (see README / professor-demo steps)")


if __name__ == "__main__":
    main()

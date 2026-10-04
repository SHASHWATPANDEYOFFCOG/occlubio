from __future__ import annotations

import os
import shutil
import sys
import tempfile

HOME = tempfile.mkdtemp(prefix="occlubio-smoke-")
os.environ["OCCLUBIO_HOME"] = HOME
os.environ.pop("OCCLUBIO_DB", None)
os.environ["OCCLUBIO_AUTHORITY_CODE"] = "smoke-code"

import cv2
import numpy as np
from fastapi.testclient import TestClient
from insightface.data import get_image

from occlubio.api.app import app
from occlubio.pipeline.face_analyzer import FaceAnalyzer
from occlubio.platform_support import open_video_writer, os_name


def check(cond: bool, what: str) -> None:
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        raise SystemExit(1)


def biggest_face_crop(img: np.ndarray) -> np.ndarray:
    faces = FaceAnalyzer(model_name="buffalo_l", ctx_id=-1, detection_only=True).analyze(img)
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = f.bbox
    m = 0.4 * max(x2 - x1, y2 - y1)
    h, w = img.shape[:2]
    return img[max(0, int(y1 - m)):min(h, int(y2 + m)), max(0, int(x1 - m)):min(w, int(x2 + m))]


def make_clip(img: np.ndarray, path: str, frames: int = 20) -> None:
    h, w = img.shape[:2]
    writer, _ = open_video_writer(path, 10.0, (w, h))
    for _ in range(frames):
        writer.write(img)
    writer.release()


def main() -> None:
    print(f"platform={os_name()} home={HOME}")
    img = get_image("t1")
    crop = biggest_face_crop(img)
    clip = os.path.join(HOME, "smoke clip.mp4")
    make_clip(img, clip)

    with TestClient(app) as c:
        r = c.post("/api/register", json={
            "email": "admin@example.com", "password": "smoke-pass-1", "full_name": "Smoke Admin",
            "role": "authority", "username": "smokeadmin", "authority_code": "smoke-code"})
        check(r.status_code == 200, f"register authority ({r.status_code})")
        auth = {"Authorization": f"Bearer {r.json()['token']}"}

        r = c.post("/api/register", json={
            "email": "student@example.com", "password": "smoke-pass-2", "full_name": "Smoke Student",
            "role": "user", "roll_number": "SMOKE-001"})
        check(r.status_code == 200, f"register student ({r.status_code})")
        student_id = r.json()["user_id"]

        ok, jpg = cv2.imencode(".jpg", crop)
        r = c.post(f"/api/users/{student_id}/enroll", headers=auth,
                   files=[("files", ("face.jpg", jpg.tobytes(), "image/jpeg"))])
        check(r.status_code == 200, f"enroll student ({r.status_code} {r.text[:120]})")

        with open(clip, "rb") as fh:
            r = c.post("/api/identify", headers=auth,
                       data={"stride": "2", "camera": "smoke-cam", "target_user_id": str(student_id)},
                       files={"file": ("../../evil name?.mp4", fh, "video/mp4")})
        check(r.status_code == 200, f"submit identify job ({r.status_code} {r.text[:120]})")
        job_id = r.json()["job_id"]

        job = c.get(f"/api/jobs/{job_id}", headers=auth).json()
        check(job["status"] == "done", f"job finished ({job['status']}: {job.get('message')})")
        report = job["report"]
        check(bool(report["target"]["found"]), "enrolled target found in clip")
        proc = report["processing"]
        print(f"     output codec={proc['video_codec']} browser_playable={proc['browser_playable']}")

        r = c.get(f"/api/jobs/{job_id}/video", headers=auth)
        check(r.status_code == 200 and len(r.content) > 0, "annotated video served")
        r = c.get(f"/api/jobs/{job_id}/source", headers=auth)
        check(r.status_code == 200 and len(r.content) > 0, "source clip served")
        uploads = os.listdir(os.path.join(HOME, "data", "uploads"))
        check(uploads == [f"job_{job_id}.mp4"], f"upload name sanitised ({uploads})")
        check(c.get("/api/alerts", headers=auth).status_code == 200, "alerts endpoint")

    print("smoke test passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        if "--keep" not in sys.argv:
            shutil.rmtree(HOME, ignore_errors=True)

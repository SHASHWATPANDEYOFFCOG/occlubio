import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
psutil = pytest.importorskip("psutil")
pytest.importorskip("uvicorn")

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_MB = 200

SERVER = r"""
import sys
import numpy as np
import uvicorn
import occlubio.api.app as api
from occlubio.db import init_db
init_db()

class StubService:
    def template_for_user(self, db, uid):
        return np.ones(512, np.float32)
    def identify_video(self, *a, **k):
        raise ValueError("stub: no processing in this test")

api.service = StubService()
uvicorn.run(api.app, host="127.0.0.1", port=int(sys.argv[1]), lifespan="off", log_level="warning")
"""


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_large_upload_is_streamed_to_disk_not_buffered_in_memory(tmp_path):
    port = _free_port()
    env = dict(os.environ, OCCLUBIO_HOME=str(tmp_path / "home"), OCCLUBIO_AUTHORITY_CODE="m-code")
    env.pop("OCCLUBIO_DB", None)
    proc = subprocess.Popen([sys.executable, "-c", SERVER, str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(200):
            try:
                if httpx.get(base + "/login", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        procs = [psutil.Process(proc.pid)] + psutil.Process(proc.pid).children(recursive=True)

        def rss():
            return sum(p.memory_info().rss for p in procs if p.is_running())

        c = httpx.Client(base_url=base, timeout=300)
        op = c.post("/api/register", json={"email": "o@example.com", "password": "pass-word-1", "full_name": "Op One",
                                           "role": "authority", "username": "opone", "authority_code": "m-code"}).json()
        st = c.post("/api/register", json={"email": "s@example.com", "password": "pass-word-2", "full_name": "Stu Dent",
                                           "role": "user", "roll_number": "S-1"}).json()
        big = tmp_path / "big.mp4"
        with open(big, "wb") as fh:
            chunk = os.urandom(1 << 20)
            for _ in range(UPLOAD_MB):
                fh.write(chunk)

        baseline = rss()
        peak = [baseline]
        done = threading.Event()

        def watch():
            while not done.is_set():
                peak[0] = max(peak[0], rss())
                time.sleep(0.01)

        t = threading.Thread(target=watch)
        t.start()
        with open(big, "rb") as fh:
            r = c.post("/api/identify", headers={"Authorization": "Bearer " + op["token"]},
                       data={"target_user_id": str(st["user_id"])}, files={"file": ("big.mp4", fh, "video/mp4")})
        time.sleep(0.5)
        done.set()
        t.join()
        assert r.status_code == 200, r.text
        stored = tmp_path / "home" / "data" / "uploads" / f"job_{r.json()['job_id']}.mp4"
        assert stored.stat().st_size == UPLOAD_MB << 20
        grew_mb = (peak[0] - baseline) / 2**20
        assert grew_mb < UPLOAD_MB / 4, f"server memory grew by {grew_mb:.0f} MB for a {UPLOAD_MB} MB upload"
    finally:
        proc.terminate()
        proc.wait(timeout=20)

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import json, os
import numpy as np
from datetime import datetime
from fastapi.testclient import TestClient
import occlubio.api.app as api
from occlubio.db import init_db
init_db()

class StubService:
    def template_for_user(self, db, uid):
        return np.ones(512, np.float32)
    def identify_video(self, *a, **k):
        return {"people": [], "unknown_alerts": [], "summary": {"known_people": 0, "unknown_alerts": 0},
                "processing": {"processing_fps": 0}}
api.service = StubService()
c = TestClient(api.app)
def reg(**b):
    return c.post("/api/register", json=b).json()
op = reg(email="o@example.com", password="pass-word-1", full_name="Op One", role="authority",
         username="opone", authority_code=os.environ["OCCLUBIO_AUTHORITY_CODE"])
st = reg(email="s@example.com", password="pass-word-2", full_name="Stu Dent", role="user", roll_number="S-1")
A = {"Authorization": "Bearer " + op["token"]}
before = datetime.now().isoformat(timespec="seconds")
c.post("/api/identify", headers=A, data={"target_user_id": str(st["user_id"])},
       files={"file": ("c.mp4", b"x", "video/mp4")})
c.post("/api/messages", headers=A, json={"body": "hello", "recipient_ids": None})
from occlubio.db import SessionLocal
from occlubio.db.models import Job
with SessionLocal() as s:
    job = s.query(Job).first()
    captured = job.captured_at.isoformat(timespec="seconds")
msg = c.get("/api/messages", headers=A).json()[0]["created_at"]
print(json.dumps({"before": before, "captured": captured, "message": msg}))
"""


def test_server_timestamps_use_the_same_local_clock_as_user_input(tmp_path):
    env = dict(os.environ, OCCLUBIO_HOME=str(tmp_path), OCCLUBIO_AUTHORITY_CODE="t-code")
    if not sys.platform.startswith("win"):
        env["TZ"] = "Asia/Kolkata"
    env.pop("OCCLUBIO_DB", None)
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    before = datetime.fromisoformat(r["before"])
    for key in ("captured", "message"):
        got = datetime.fromisoformat(r[key][:19])
        assert abs((got - before).total_seconds()) < 60, f"{key} stamped {got} but local time was {before}"


def test_offset_is_observable_on_this_machine():
    offset = time.localtime().tm_gmtoff
    if offset == 0:
        pytest.skip("machine clock is UTC; the local/UTC mix-up cannot be observed here")
    assert offset != 0

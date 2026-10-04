import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import json, os
from fastapi.testclient import TestClient
from occlubio.api.app import app
from occlubio.db import init_db
init_db()
c = TestClient(app)
def reg(code, n):
    return c.post("/api/register", json={"email": f"a{n}@example.com", "password": "pass-word-1",
        "full_name": "Auth User", "role": "authority", "username": f"auth{n}", "authority_code": code}).status_code
code_file = os.path.join(os.environ["OCCLUBIO_HOME"], "authority_code.txt")
generated = open(code_file, encoding="utf-8").read().strip() if os.path.exists(code_file) else None
print(json.dumps({"public_default": reg("occlubio-authority", 1), "wrong": reg("nope", 2),
                  "generated": generated, "with_generated": reg(generated, 3) if generated else None,
                  "with_env": reg(os.environ.get("OCCLUBIO_AUTHORITY_CODE"), 4) if os.environ.get("OCCLUBIO_AUTHORITY_CODE") else None}))
"""


def _probe(tmp_path, env_code=None):
    env = dict(os.environ, OCCLUBIO_HOME=str(tmp_path))
    env.pop("OCCLUBIO_DB", None)
    env.pop("OCCLUBIO_AUTHORITY_CODE", None)
    if env_code:
        env["OCCLUBIO_AUTHORITY_CODE"] = env_code
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_public_default_code_is_rejected_and_a_private_one_is_generated(tmp_path):
    r = _probe(tmp_path)
    assert r["public_default"] == 403
    assert r["wrong"] == 403
    assert r["generated"] and r["generated"] != "occlubio-authority" and len(r["generated"]) >= 12
    assert r["with_generated"] == 200


def test_generated_code_is_stable_across_restarts(tmp_path):
    first = _probe(tmp_path)["generated"]
    assert first
    again = _probe(tmp_path)
    assert again["generated"] == first
    assert again["with_generated"] != 403, "stored code was not accepted after restart"


def test_environment_code_takes_precedence(tmp_path):
    r = _probe(tmp_path, env_code="site-specific-code")
    assert r["with_env"] == 200
    assert r["public_default"] == 403
    assert r["generated"] is None

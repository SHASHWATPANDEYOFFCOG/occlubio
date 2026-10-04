import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from occlubio.api.app import _safe_suffix, app

client = TestClient(app)


def test_manifest_and_icons_served():
    r = client.get("/static/manifest.webmanifest")
    assert r.status_code == 200
    manifest = json.loads(r.text)
    for icon in manifest["icons"]:
        assert client.get(icon["src"]).status_code == 200
    assert client.get("/static/icons/apple-touch-icon.png").status_code == 200


@pytest.mark.parametrize("page", ["/", "/login"])
def test_pages_are_ios_ready(page):
    html = client.get(page).text
    assert "viewport-fit=cover" in html
    assert 'rel="manifest"' in html
    assert 'rel="apple-touch-icon"' in html
    assert "100dvh" in html
    assert "safe-area-inset" in html


@pytest.mark.parametrize("name,expected", [
    ("clip.MP4", ".mp4"),
    ("IMG_0001.MOV", ".mov"),
    ("../../evil name?.mp4", ".mp4"),
    ("a:b|c.avi", ".avi"),
    (None, ".mp4"),
    ("noext", ".mp4"),
    ("x.waytoolongext", ".mp4"),
])
def test_upload_suffix_is_sanitised(name, expected):
    assert _safe_suffix(name) == expected

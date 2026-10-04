import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from occlubio import platform_support as ps

ROOT = Path(__file__).resolve().parents[1]
ODD_NAMES = ["with space", "ünïcødé", "हिन्दी", "日本語"]


@pytest.mark.parametrize("name", ODD_NAMES)
def test_data_path_and_db_url_with_unusual_home(monkeypatch, tmp_path, name):
    home = tmp_path / name
    monkeypatch.setenv("OCCLUBIO_HOME", str(home))
    url = ps.default_db_url()
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    with engine.begin() as c:
        c.execute(text("CREATE TABLE t (v TEXT)"))
        c.execute(text("INSERT INTO t VALUES ('ok')"))
    assert (home / "occlubio.db").exists()
    assert ps.data_path("data/uploads") == home / "data" / "uploads"


@pytest.mark.parametrize("name", ODD_NAMES)
def test_video_write_then_read_back_in_unusual_dir(tmp_path, name):
    cv2 = pytest.importorskip("cv2")
    d = tmp_path / name
    d.mkdir()
    clip = d / "clip.mp4"
    writer, _ = ps.open_video_writer(clip, 10.0, (64, 48))
    for _ in range(5):
        writer.write(np.full((48, 64, 3), 128, np.uint8))
    writer.release()
    assert clip.exists() and clip.stat().st_size > 0
    cap = cv2.VideoCapture(str(clip))
    ok, _ = cap.read()
    cap.release()
    assert ok, f"OpenCV could not read back a video stored under '{name}'"


_FAISS_WIN_NON_ASCII = pytest.mark.xfail(
    sys.platform.startswith("win"), strict=True,
    reason="ISSUE-004: faiss write_index/read_index use ANSI fopen on Windows; non-ASCII dirs fail")


@pytest.mark.parametrize("name", [n if n.isascii() else pytest.param(n, marks=_FAISS_WIN_NON_ASCII)
                                  for n in ODD_NAMES])
def test_faiss_gallery_save_load_in_unusual_dir(tmp_path, name):
    pytest.importorskip("faiss")
    from occlubio.gallery import FaissGallery

    g = FaissGallery(dim=4)
    g.add(np.array([1, 0, 0, 0], np.float32), "Zoë 李")
    g.save(tmp_path / name / "gal")
    g2 = FaissGallery.load(tmp_path / name / "gal")
    assert g2.identify(np.array([1, 0, 0, 0], np.float32), 0.5)[0] == "Zoë 李"


@pytest.mark.parametrize("name", ODD_NAMES)
def test_identity_report_saved_in_unusual_dir_is_utf8(tmp_path, name):
    from occlubio.analytics import IdentityLog

    out = tmp_path / name / "report.json"
    out.parent.mkdir()
    IdentityLog(min_track_frames=1).save(out)
    assert "summary" in json.loads(out.read_text(encoding="utf-8"))


def test_config_loads_regardless_of_locale_encoding(monkeypatch):
    import locale

    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a, **k: "cp1252")
    from occlubio import load_config

    assert load_config().recognition.embedding_dim == 512


def _git_ls(pattern):
    out = subprocess.run(["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True)
    return [ROOT / p for p in out.stdout.split()]


@pytest.mark.parametrize("pattern", ["*.sh", "*.py", "*.html", "*.yml", "*.spec", "*.plist"])
def test_unix_files_have_lf_line_endings_in_checkout(pattern):
    files = _git_ls(pattern)
    if not files:
        pytest.skip("not a git checkout")
    crlf = [str(f.relative_to(ROOT)) for f in files if b"\r\n" in f.read_bytes()]
    assert not crlf, f"CRLF line endings would break these on macOS/iOS tooling: {crlf}"


def test_shell_scripts_are_executable_in_git():
    out = subprocess.run(["git", "ls-files", "-s", "*.sh"], cwd=ROOT, capture_output=True, text=True).stdout
    if not out:
        pytest.skip("not a git checkout")
    modes = {line.split()[3]: line.split()[0] for line in out.splitlines()}
    assert all(m == "100755" for m in modes.values()), modes


def test_upload_dest_never_escapes_upload_dir():
    pytest.importorskip("fastapi")
    from occlubio.api.app import UPLOAD_DIR, _safe_suffix

    for name in ["../../x.mp4", "..\\..\\x.mp4", "/etc/passwd", "C:\\Windows\\x.mp4", "a/b/c.mov", "\u202ecod.mp4"]:
        dest = (UPLOAD_DIR / f"job_1{_safe_suffix(name)}").resolve()
        assert dest.parent == UPLOAD_DIR.resolve()


def test_parse_dt_accepts_browser_datetime_local_and_utc_z():
    pytest.importorskip("fastapi")
    from occlubio.api.app import _parse_dt

    assert _parse_dt("2026-10-01T09:00") == datetime(2026, 10, 1, 9, 0)
    assert _parse_dt("2026-10-01T09:00:30") == datetime(2026, 10, 1, 9, 0, 30)
    assert _parse_dt("2026-10-01T09:00:00Z") == datetime(2026, 10, 1, 9, 0)


def test_platform_layer_is_the_only_os_switch():
    offenders = []
    for f in (ROOT / "occlubio").rglob("*.py"):
        if f.name == "platform_support.py":
            continue
        s = f.read_text(encoding="utf-8")
        if "sys.platform" in s or "os.name ==" in s or "platform.system()" in s:
            offenders.append(str(f.relative_to(ROOT)))
    assert not offenders, offenders

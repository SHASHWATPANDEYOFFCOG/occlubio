import sys
from pathlib import Path

import numpy as np
import pytest

from occlubio import platform_support as ps


def test_source_checkout_keeps_cwd_relative_layout(monkeypatch):
    monkeypatch.delenv("OCCLUBIO_HOME", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert ps.data_root() is None
    assert ps.data_path("data/uploads") == Path("data/uploads")
    assert ps.default_db_url() == "sqlite:///occlubio.db"


def test_occlubio_home_relocates_state(monkeypatch, tmp_path):
    monkeypatch.setenv("OCCLUBIO_HOME", str(tmp_path))
    assert ps.data_path("data/uploads") == tmp_path / "data" / "uploads"
    assert ps.default_db_url() == f"sqlite:///{(tmp_path / 'occlubio.db').as_posix()}"
    absolute = tmp_path / "elsewhere"
    assert ps.data_path(absolute) == absolute


def test_frozen_build_uses_user_data_dir(monkeypatch):
    monkeypatch.delenv("OCCLUBIO_HOME", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert ps.data_root() == ps.user_data_dir()


@pytest.mark.parametrize("platform,expected", [
    ("darwin", Path("Library") / "Application Support" / "occlubio"),
    ("linux", Path(".local") / "share" / "occlubio"),
])
def test_user_data_dir_per_os(monkeypatch, tmp_path, platform, expected):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert ps.user_data_dir() == tmp_path / expected


def test_user_data_dir_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert ps.user_data_dir() == tmp_path / "occlubio"


def test_macos_logs_live_in_library_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert ps.user_log_dir() == tmp_path / "Library" / "Logs" / "occlubio"


def test_resource_dir_contains_web_and_configs():
    root = ps.resource_dir()
    assert (root / "web" / "index.html").exists()
    assert (root / "configs" / "default.yaml").exists()


def test_onnx_providers_drops_unavailable():
    pytest.importorskip("onnxruntime")
    chosen = ps.onnx_providers(["NoSuchExecutionProvider", "CPUExecutionProvider"])
    assert chosen == ["CPUExecutionProvider"]
    assert ps.onnx_providers(["NoSuchExecutionProvider"]) == ["CPUExecutionProvider"]


def test_codec_order_per_os(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert ps.video_codecs() == ["mp4v"]
    monkeypatch.setattr(sys, "platform", "darwin")
    assert ps.video_codecs()[0] == "avc1"


def test_video_writer_opens_and_writes(tmp_path):
    out = tmp_path / "clip.mp4"
    writer, codec = ps.open_video_writer(out, 10.0, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    assert codec in ps.video_codecs()
    assert out.stat().st_size > 0

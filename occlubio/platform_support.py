from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

APP_NAME = "occlubio"
DEFAULT_ONNX_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def os_name() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "ios":
        return "ios"
    return "linux"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> Path:
    bundle = getattr(sys, "_MEIPASS", None)
    if is_frozen() and bundle:
        return Path(bundle)
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    home = Path.home()
    name = os_name()
    if name == "windows":
        base = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    elif name == "macos":
        base = home / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    return base / APP_NAME


def user_log_dir() -> Path:
    name = os_name()
    if name == "macos":
        return Path.home() / "Library" / "Logs" / APP_NAME
    if name == "windows":
        return user_data_dir() / "logs"
    state = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return state / APP_NAME / "logs"


def data_root() -> Optional[Path]:
    override = os.environ.get("OCCLUBIO_HOME")
    if override:
        return Path(override).expanduser()
    if is_frozen():
        return user_data_dir()
    return None


def data_path(relative: str | Path) -> Path:
    p = Path(relative)
    root = data_root()
    if p.is_absolute() or root is None:
        return p
    return root / p


def default_db_url() -> str:
    root = data_root()
    if root is None:
        return "sqlite:///occlubio.db"
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(root / 'occlubio.db').as_posix()}"


def onnx_providers(requested: Optional[Sequence[str]] = None) -> List[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    chosen = [p for p in (requested or DEFAULT_ONNX_PROVIDERS) if p in available]
    return chosen or ["CPUExecutionProvider"]


def torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def video_codecs() -> List[str]:
    if os_name() == "macos":
        return ["avc1", "mp4v"]
    return ["mp4v"]


def browser_playable(codec: str) -> bool:
    return codec == "avc1"


def open_video_writer(path: str | Path, fps: float, size: Tuple[int, int]):
    import cv2

    tried = video_codecs()
    for codec in tried:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise ValueError(f"could not open a video writer (tried {', '.join(tried)}) in this OpenCV build")


def open_url(url: str) -> bool:
    return webbrowser.open(url)

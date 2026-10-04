# PyInstaller spec for the occlubio desktop app.
#   macOS : produces dist/occlubio.app (one arch per build: arm64 or x86_64)
#   other : produces a onedir build, used only to smoke-test the frozen code paths
# Build with packaging/macos/build_macos.sh rather than calling this directly.
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from occlubio import __version__  # noqa: E402

IS_MAC = sys.platform == "darwin"
BUNDLE_ID = os.environ.get("OCCLUBIO_BUNDLE_ID", "io.github.shashwatpandeyoffcog.occlubio")
SIGN_ID = os.environ.get("APPLE_SIGNING_IDENTITY") or None
ENTITLEMENTS = str(ROOT / "packaging" / "macos" / "entitlements.plist") if IS_MAC else None

datas = [
    (str(ROOT / "web"), "web"),
    (str(ROOT / "configs"), "configs"),
]
datas += collect_data_files("insightface")
# insightface.data.get_object() resolves sys._MEIPASS/objects when frozen, not the package path.
import insightface.data as _ins_data  # noqa: E402

datas.append((str(Path(_ins_data.__file__).parent / "objects"), "objects"))
binaries = collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("faiss")
hiddenimports = (
    collect_submodules("uvicorn")
    + collect_submodules("occlubio", filter=lambda name: not name.startswith("occlubio.training"))
    + ["sqlalchemy.dialects.sqlite", "multipart", "python_multipart"]
)
if IS_MAC:
    hiddenimports += collect_submodules("webview")

excludes = ["torch", "torchvision", "timm", "albumentations", "matplotlib", "tkinter",
            "IPython", "pytest", "scalebench"]

a = Analysis(
    [str(ROOT / "packaging" / "macos" / "launch.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="occlubio",
    console=not IS_MAC,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=SIGN_ID,
    entitlements_file=ENTITLEMENTS,
)
coll = COLLECT(exe, a.binaries, a.datas, name="occlubio")

if IS_MAC:
    app = BUNDLE(
        coll,
        name="occlubio.app",
        icon=str(ROOT / "packaging" / "macos" / "occlubio.icns"),
        bundle_identifier=BUNDLE_ID,
        version=__version__,
        info_plist={
            "CFBundleName": "occlubio",
            "CFBundleDisplayName": "occlubio",
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": os.environ.get("OCCLUBIO_BUILD_NUMBER", __version__),
            "LSMinimumSystemVersion": "11.0",
            "LSApplicationCategoryType": "public.app-category.utilities",
            "NSHighResolutionCapable": True,
            "NSCameraUsageDescription": "occlubio uses the camera to capture enrollment photos of your face.",
            "NSHumanReadableCopyright": "Copyright © Shashwat Kumar Pandey",
        },
    )

#!/usr/bin/env bash
# Build occlubio.app + a .dmg for the current Mac's architecture (arm64 or x86_64),
# then optionally sign, smoke-test, notarize and staple.
#
# Usage:  packaging/macos/build_macos.sh [--skip-smoke] [--python python3.12]
#
# Signing / notarization are driven by environment variables. Never commit them.
#   APPLE_SIGNING_IDENTITY   "Developer ID Application: Your Name (TEAMID)"  (unset -> ad-hoc signature)
#   APPLE_NOTARY_PROFILE     keychain profile created with `xcrun notarytool store-credentials`
#     -- or --
#   APPLE_ID, APPLE_TEAM_ID, APPLE_APP_PASSWORD   (app-specific password)
#   OCCLUBIO_BUNDLE_ID       override the bundle identifier
#
# A universal2 build is not produced: onnxruntime, faiss-cpu and opencv ship
# single-arch wheels, so run this once on Apple Silicon and once on Intel
# (or on the macos-latest and macos-15-intel CI runners).
set -euo pipefail

SKIP_SMOKE=0
PYTHON="${PYTHON:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-smoke) SKIP_SMOKE=1; shift ;;
        --python)     PYTHON="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ "$(uname -s)" == "Darwin" ]] || { echo "build_macos.sh must run on macOS" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ARCH="$(uname -m)"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' occlubio/__init__.py)"
DIST="$ROOT/dist/macos-$ARCH"
APP="$DIST/occlubio.app"
DMG="$ROOT/dist/occlubio-$VERSION-macos-$ARCH.dmg"
BUILD_VENV="$ROOT/.venv-build-$ARCH"

if [[ -z "$PYTHON" ]]; then
    for cand in python3.12 python3.11 python3.10; do
        if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
    done
fi
[[ -n "$PYTHON" ]] || { echo "need Python 3.10-3.12 (brew install python@3.12)" >&2; exit 1; }

echo "==> occlubio $VERSION for $ARCH using $($PYTHON --version)"
if [[ ! -x "$BUILD_VENV/bin/python" ]]; then
    "$PYTHON" -m venv "$BUILD_VENV"
fi
PY="$BUILD_VENV/bin/python"
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install -e ".[infer,api,desktop]" "pyinstaller>=6.10"

echo "==> PyInstaller"
rm -rf "$DIST" "$ROOT/build/macos-$ARCH"
"$PY" -m PyInstaller --noconfirm --clean \
    --distpath "$DIST" --workpath "$ROOT/build/macos-$ARCH" \
    packaging/macos/occlubio.spec

echo "==> code signing"
ENTITLEMENTS="$ROOT/packaging/macos/entitlements.plist"
if [[ -n "${APPLE_SIGNING_IDENTITY:-}" ]]; then
    codesign --force --deep --options runtime --timestamp \
        --entitlements "$ENTITLEMENTS" --sign "$APPLE_SIGNING_IDENTITY" "$APP"
else
    echo "    APPLE_SIGNING_IDENTITY not set -> ad-hoc signature (runs locally; Gatekeeper will warn elsewhere)"
    codesign --force --deep --entitlements "$ENTITLEMENTS" --sign - "$APP"
fi
codesign --verify --deep --strict --verbose=2 "$APP"

if [[ $SKIP_SMOKE -eq 0 ]]; then
    echo "==> smoke test (headless app binary)"
    SMOKE_HOME="$(mktemp -d)"
    PORT=8765
    OCCLUBIO_HOME="$SMOKE_HOME" OCCLUBIO_AUTHORITY_CODE=smoke-code \
        "$APP/Contents/MacOS/occlubio" --headless --port "$PORT" > "$DIST/smoke-server.log" 2>&1 &
    APP_PID=$!
    trap 'kill $APP_PID 2>/dev/null || true' EXIT
    for _ in $(seq 1 600); do
        curl -fs -o /dev/null "http://127.0.0.1:$PORT/login" && break
        kill -0 "$APP_PID" 2>/dev/null || { cat "$DIST/smoke-server.log"; exit 1; }
        sleep 1
    done
    "$PY" -m pip install -q httpx
    "$PY" scripts/smoke_api.py --url "http://127.0.0.1:$PORT" --home "$SMOKE_HOME" --authority-code smoke-code
    kill "$APP_PID" 2>/dev/null || true
    trap - EXIT
fi

echo "==> dmg"
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "occlubio $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
rm -rf "$STAGE"
if [[ -n "${APPLE_SIGNING_IDENTITY:-}" ]]; then
    codesign --force --timestamp --sign "$APPLE_SIGNING_IDENTITY" "$DMG"
fi

if [[ -n "${APPLE_SIGNING_IDENTITY:-}" && ( -n "${APPLE_NOTARY_PROFILE:-}" || -n "${APPLE_ID:-}" ) ]]; then
    echo "==> notarization"
    if [[ -n "${APPLE_NOTARY_PROFILE:-}" ]]; then
        xcrun notarytool submit "$DMG" --keychain-profile "$APPLE_NOTARY_PROFILE" --wait
    else
        : "${APPLE_TEAM_ID:?APPLE_TEAM_ID is required with APPLE_ID}"
        : "${APPLE_APP_PASSWORD:?APPLE_APP_PASSWORD is required with APPLE_ID}"
        xcrun notarytool submit "$DMG" --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" \
            --password "$APPLE_APP_PASSWORD" --wait
    fi
    xcrun stapler staple "$DMG"
    spctl --assess --type open --context context:primary-signature --verbose "$DMG" || true
else
    echo "    notarization skipped (needs APPLE_SIGNING_IDENTITY plus APPLE_NOTARY_PROFILE or APPLE_ID/APPLE_TEAM_ID/APPLE_APP_PASSWORD)"
fi

echo "==> done"
echo "    app: $APP"
echo "    dmg: $DMG"

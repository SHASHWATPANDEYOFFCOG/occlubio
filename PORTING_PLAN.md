# Porting plan: Windows → Windows + macOS + iOS

Status: **implemented** on branch `feature/cross-platform` with Option A (iOS client), a full macOS
`.app`/`.dmg`, and opt-in HTTPS for the LAN. Platform behaviour and open items: PLATFORM_NOTES.md.

Found during implementation:
- `.gitignore`'s `data/` rule also hid the `occlubio/data` package, so every fresh clone was broken
  on every OS. Fixed by anchoring it to `/data/`.
- #6 is smaller than expected: insightface **1.0.1** (what Windows already used) is a pure-Python
  wheel, so macOS needs no compiler. The floor is now `insightface>=1.0.1`.
Baseline on Windows (Python 3.12.6, `.venv`): `pytest -q` → 5 passed, 1 skipped.

## 1. Stack summary

| Aspect | What it is |
|---|---|
| Language | Python 3.10–3.12 (backend + CLI); vanilla HTML/CSS/JS (UI, no build step) |
| Framework | FastAPI + Uvicorn (HTTP API + serves `web/*.html`); SQLAlchemy + SQLite; FAISS; insightface + onnxruntime + OpenCV; optional PyTorch (training) |
| Build / packaging | setuptools via `pyproject.toml` (`pip install -e ".[infer,api]"`). No installer, no frozen app, no CI |
| Entry points | `uvicorn occlubio.api.app:app --port 8001` (web console); `scripts/*.py` (CLI); `python -m occlubio.training.train`; `scalebench/run.py` |
| Launchers | `run_server.bat` → `run_server.ps1` (untracked, Windows-only) |
| UI | Browser-based. Runs in any browser; already has a mobile breakpoint (`@media (max-width:780px)`) |

**Key point:** this is not a desktop GUI app. It is a local web server plus a browser UI. The
Python code already uses `pathlib` and explicit UTF-8 almost everywhere, and every tracked file
uses LF line endings. So most of the macOS work is launchers, data locations, dependency install
and packaging. The OS-specific code is a small part of it.

## 2. Issues found

Severity: **B** = blocks macOS/iOS · **M** = wrong/degraded behaviour · **L** = polish / hygiene.

| # | File:line | What it is | Why it breaks on macOS / iOS | Proposed fix | Sev |
|---|---|---|---|---|---|
| 1 | `run_server.bat`, `run_server.ps1` (whole files) | Windows launcher: `.venv\Scripts\python.exe`, `Get-NetTCPConnection`, `Start-Process`, PowerShell jobs | No PowerShell/cmd on macOS by default; venv layout is `.venv/bin/python` | Add `run_server.sh` (bash/zsh) with the same flags (port, bind, authority code, db, reload, no-browser, dep check, port-busy check, `open` URL). Keep `.bat`/`.ps1` unchanged | B |
| 2 | `occlubio/db/database.py:10` | Default DB `sqlite:///occlubio.db` relative to CWD | Works from a source checkout. In a packaged `.app` the CWD is `/` (read-only), so the DB can't be created | Platform layer `app_data_dir()` → `%LOCALAPPDATA%\occlubio` / `~/Library/Application Support/occlubio` / XDG. Use it **only when frozen/packaged** or when `OCCLUBIO_HOME` is set; source checkouts keep today's CWD-relative paths so Windows behaviour doesn't change | B (packaged) |
| 3 | `occlubio/api/app.py:28-29` | `ensure_dir("data/uploads")`, `ensure_dir("data/outputs")` at import time | Same CWD problem as #2 when packaged | Route through the platform layer (same rule as #2) | B (packaged) |
| 4 | `configs/default.yaml:50` (`gallery.path: gallery_store`) | Relative gallery path | Same as #2 when packaged | Resolve relative paths against the platform data dir when packaged | B (packaged) |
| 5 | `occlubio/api/app.py:495` | Upload saved as `f"{UPLOAD_DIR}/job_{id}_{file.filename}"`. Path built by string concatenation from a client-supplied name | Characters valid on macOS/iOS names (`:`, `?`, `*`, `"`, `|`) fail on Windows. iOS uploads arrive as `IMG_1234.MOV` or with no name at all. A crafted name containing `../` is a path traversal on every OS | Sanitize to `job_<id><safe-suffix>` with `pathlib`. Keep the original name in the DB only | M (+security) |
| 6 | `pyproject.toml:21` (`insightface>=0.7.3`) | insightface ships wheels for Windows only; on macOS it builds a Cython extension | `pip install` fails on a Mac without the Xcode Command Line Tools | Document `xcode-select --install`. `run_server.sh` checks for it and stops with a clear message. CI installs it on `macos-latest` | B |
| 7 | `configs/default.yaml:6-8`, `occlubio/pipeline/antispoof.py:24-26` | Provider list is CUDA→CPU. `antispoof.py` passes the list straight to ORT without filtering by availability | No CUDA on macOS. The anti-spoof session logs warnings for the missing CUDA provider. Apple's GPU/ANE goes unused | Platform layer `onnx_providers()`: filter to available providers. On macOS, optionally prepend `CoreMLExecutionProvider` behind a config flag (off by default until accuracy is verified) | M |
| 8 | `occlubio/training/train.py:67` | Device default `cuda` else `cpu` | Apple Silicon's `mps` backend is ignored, so training is very slow | Default to `cuda` → `mps` → `cpu` | L |
| 9 | `occlubio/service/face_service.py:125`, `scripts/identify_video.py:47`, `scripts/recognize_video.py:45`, `scripts/make_demo.py:33` | Output video encoded as `mp4v` | Already can't play in browsers on Windows (UI tells the user to use VLC). Same on Safari/iOS. On iOS there is no VLC by default, so annotated output can't be viewed in-app | Try `avc1` (H.264) first and fall back to `mp4v`. opencv-python wheels on macOS can usually write H.264; on Windows it depends on the OpenH264 DLL, so keep the fallback and the existing message. Not a regression either way | M |
| 10 | `occlubio/service/face_service.py:115` (decode side) | iPhone clips are HEVC `.mov` | opencv-python's bundled FFmpeg decodes HEVC on macOS/Linux. On Windows decoding depends on the build | Verify with a real iPhone clip. If it fails, show a clear error ("re-export as H.264 / Most Compatible") instead of an empty report | M |
| 11 | `scalebench/report.py:599-622` (untracked) | Hardcoded `C:\Program Files\Microsoft\Edge\...\msedge.exe` + `file:///` + backslash replace | No Edge at that path on macOS, so PDF generation is skipped | Look up the browser per OS (Edge/Chrome on Windows; `/Applications/Google Chrome.app/...` or `Microsoft Edge.app` on macOS; `chromium` on Linux). Build the file URL with `Path.as_uri()` | L |
| 12 | `scalebench/report.py:51` | Matplotlib font `"Segoe UI"` first | Not on macOS. Matplotlib falls back to DejaVu with a warning | Add `"Helvetica Neue"`/`"Arial"` before the DejaVu fallback | L |
| 13 | `web/index.html:25`, `web/login.html:25` | Font stack `system-ui,-apple-system,"Segoe UI",…` | Already correct (`-apple-system` resolves to SF) | None | — |
| 14 | `web/index.html:50`, `web/login.html:41` | `min-height:100vh` | On iOS Safari, `100vh` includes the area under the collapsing toolbar, so content jumps or gets cut off | Add `min-height:100dvh` after the `100vh` line (progressive enhancement) | M (iOS) |
| 15 | `web/*.html` `<meta name="viewport">`, fixed bars (`index.html:83`, toast at `:574`) | No `viewport-fit=cover` / `env(safe-area-inset-*)` | Content under the notch and Dynamic Island; the home indicator overlaps controls in standalone/PWA/Capacitor mode | Add `viewport-fit=cover` and safe-area padding on `.mobile-bar`, `.side`, modals and toasts | M (iOS) |
| 16 | `web/index.html` ~20 `:hover` rules with `transform` | Hover-lift effects | On touch devices hover "sticks" after a tap, so buttons stay raised | Wrap hover effects in `@media (hover:hover)` | L (iOS) |
| 17 | `web/index.html:774` (`getUserMedia`) | Webcam enrol | iOS Safari only allows the camera in a **secure context** (HTTPS or `localhost`). A phone hitting `http://<lan-ip>:8001` gets `navigator.mediaDevices === undefined` and the code throws | Feature-detect it and show "Camera needs HTTPS — use photo upload instead". The photo upload input (`accept="image/*"`) already opens the iOS camera, so that path keeps working. Document HTTPS for LAN use | B (iOS webcam) |
| 18 | `web/index.html:866`, `:918` | "Open in VLC" guidance and `<a download>` | iOS can't run VLC from the browser by default. `download` saves to Files | Becomes moot for output video if #9 lands (H.264 plays inline). Otherwise reword per platform | L |
| 19 | No `.gitattributes` | Line endings are LF today but nothing enforces it | A Windows checkout with `core.autocrlf=true` would give `run_server.sh` / `dgx_spark_setup.sh` CRLF, and bash fails with `$'\r': command not found` | `.gitattributes`: `* text=auto eol=lf`, `*.bat *.ps1 eol=crlf`, binaries `-text` | M |
| 20 | No CI | Nothing builds on any OS | — | GitHub Actions: pytest on `windows-latest` + `macos-latest` (arm64) + `macos-13` (x86_64); iOS job depends on the strategy in §3 | M |
| 21 | `training/train.py:65` `--workers 4` | DataLoader multiprocessing | macOS (like Windows) uses `spawn`. `train.py` is already under `if __name__ == "__main__"`, so it's fine | None (verified) | — |
| 22 | `deepstream/`, `requirements-edge.txt`, `scalebench/dgx_spark_setup.sh` | Jetson / DGX (Linux + CUDA) | Not Windows-specific and not meant for macOS/iOS | Out of scope. Note in PLATFORM_NOTES | — |

**Searched for and not found:** registry/`winreg`, `ctypes.windll`, pywin32, COM/WMI,
`os.startfile`, `msvcrt`, `cmd /c`/`subprocess` with Windows shells (except #11), `%APPDATA%`/`%TEMP%`
literals, `os.sep` or backslash path joins in the package, file locking, CRLF in tracked files, cp1252
I/O (every `open()` passes `encoding="utf-8"`), and import vs filename case mismatches. Desktop-GUI
items (app menu, Cmd+Q, tray, native dialogs, DPI) **don't apply** because the UI is a browser page.
The only keyboard shortcut is `Escape` to close the clip modal, and it is OS-neutral.

## 3. iOS strategy: decision needed

The core of the system (insightface/SCRFD + ArcFace via onnxruntime, OpenCV video decode, FAISS,
SQLite, a FastAPI server doing background video jobs) **cannot run as-is inside an iOS app**. There
are no pip wheels for these on iOS, an iOS app can't host a long-running background server, and
background execution is limited. Realistic options:

| Option | What iOS gets | Effort | Trade-offs |
|---|---|---|---|
| **A. iOS as a client of the server (recommended)**: make the existing web UI Safari/touch-correct (#14–#18), add a PWA manifest + icons ("Add to Home Screen"), and optionally wrap it in a **Capacitor** shell for a real App Store / Simulator app with a configurable server URL | Full console on iPhone/iPad. Processing runs on the Windows/Mac/DGX server | PWA: ~0.5 day. Capacitor shell + Info.plist + icons + Simulator CI: ~1.5–2 days | Needs a reachable server, and **HTTPS** for the camera. The Capacitor shell adds a Node + Xcode toolchain (a new framework in the repo) |
| B. On-device iOS app (Swift + Core ML) | Offline face recognition on the phone | Weeks: convert SCRFD/ArcFace to Core ML, rewrite alignment, quality gate and 1:N search in Swift, new UI | A full rewrite of the pipeline and UI. Two codebases to keep in sync. Big on-device biometric privacy and legal surface |
| C. BeeWare/Toga or Kivy (Python on iOS) | — | — | **Not viable**: insightface, onnxruntime-python, faiss and opencv-python have no iOS builds |

Option A keeps a single codebase and matches how the system is deployed (cameras → server).
Option B is a rewrite, so I'm pausing here for your choice, as the ground rules ask.

Also needs your call:
- **Network exposure.** An iPhone can only reach the server if it binds beyond `127.0.0.1`. The
  launcher already warns about this for a biometric system. For LAN use I'd add an opt-in
  `--https` mode (self-signed cert via `mkcert`, documented) and keep the default as localhost-only.
- **macOS packaging depth.** The "app" is a local server. Choices:
  (i) `run_server.sh` + docs only (fast, honest, matches how Windows is run today, ~0.5 day); or
  (ii) a real `.app`/`.dmg` (PyInstaller or Briefcase) that starts the server and opens the browser,
  with `.icns`, bundle ID, and env-var-driven `codesign` + `notarytool` steps (~2–3 days). The bundle
  will be large (~400–600 MB with onnxruntime/opencv/faiss; models still download on first run), and
  insightface's native extension must be built per architecture (arm64 and x86_64 separately; a
  universal2 build isn't practical with these wheels). For parity, Windows currently has **no**
  installer either.

## 4. Work order (assuming Option A + packaging choice)

| Step | Item(s) | Est. |
|---|---|---|
| 1 | `.gitattributes` (#19) | 0.1 d |
| 2 | Platform layer `occlubio/platform.py`: data/config/cache/log dirs, frozen detection, `onnx_providers()`, `open_url()`, torch device pick. Wire up #2–#5, #7, #8 with source-checkout behaviour unchanged. Add tests | 0.5 d |
| 3 | `run_server.sh` (#1, #6 checks) | 0.3 d |
| 4 | Video codec fallback + HEVC error message (#9, #10) | 0.3 d |
| 5 | scalebench browser/font lookup (#11, #12) | 0.2 d |
| 6 | Web UI: dvh, safe areas, hover media query, camera feature-detect, PWA manifest + icons (#14–#18) | 0.5 d |
| 7 | macOS packaging (choice i or ii) | 0.5 d / 2–3 d |
| 8 | iOS Capacitor shell (`ios-client/`), Info.plist (camera + photo library usage strings), icons, launch screen, server-URL setting | 1.5 d |
| 9 | CI: Windows + macOS arm64 + macOS x86_64 pytest; iOS Simulator `xcodebuild` | 0.5 d |
| 10 | README + PLATFORM_NOTES.md | 0.3 d |

After each step: `pytest -q` on Windows, and a manual server smoke test on port 8001.

## 5. What I can and can't verify from this machine

This is a Windows machine. I can run the Windows tests and smoke test the server here. I **can't**
run macOS or the iOS Simulator locally. Those get verified through the GitHub Actions
`macos-latest` runners, which needs the repo pushed to GitHub (your call). A real-device run
needs your Apple Developer account.

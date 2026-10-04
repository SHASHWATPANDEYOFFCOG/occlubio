# occlubio — Occlusion-Robust Biometric Face Recognition

A modular, edge-oriented pipeline for **face recognition under occlusion** (masks, sunglasses,
caps, helmets, scarves, side-profiles, low light, motion blur). The full design rationale,
papers, datasets, and deployment plan live in
[OCCLUSION_ROBUST_FR_ARCHITECTURE.md](OCCLUSION_ROBUST_FR_ARCHITECTURE.md).

This repo is the **working implementation scaffold** of that architecture.

> The baseline path (detect → align → embed → 1:N search) runs out-of-the-box with pretrained
> models. The occlusion-aware extensions (synthetic occlusion augmentation, AdaFace training,
> quality gate, anti-spoof/occlusion hooks, DeepStream config) are wired in as modular pieces you
> enable as you progress through the roadmap.

---

## Quickstart (CPU works; GPU if available)

> Use **Python 3.10–3.12** — some CV dependencies (opencv / insightface / onnxruntime / faiss)
> do not ship wheels for newer interpreters yet.

```bash
# from the repo root
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Linux/macOS:         source .venv/bin/activate

pip install -e ".[infer]"     # core + inference (insightface, onnxruntime, faiss-cpu)
# GPU box / DGX:  pip install onnxruntime-gpu faiss-gpu   (replace the cpu wheels)
```

First run downloads the pretrained model pack (`buffalo_l`) automatically (~300 MB).

### 1. Enroll a gallery
Put reference images in `data/enroll/<person_name>/*.jpg` (one folder per identity), then:
```bash
python scripts/enroll.py --images data/enroll --gallery gallery_store
```

### 2. Recognize from an image
```bash
python scripts/recognize_image.py --image path/to/test.jpg --gallery gallery_store --out out.jpg
```

### 3. Recognize from webcam / RTSP / video file
```bash
python scripts/recognize_video.py --source 0                       # webcam
python scripts/recognize_video.py --source rtsp://user:pass@cam/stream
python scripts/recognize_video.py --source clip.mp4 --gallery gallery_store
```

### 4. Build an occluded test set (synthetic masks/sunglasses/caps)
```bash
python scripts/make_occluded_dataset.py --src data/enroll --dst data/enroll_occluded
```

### 5. Benchmark latency on your hardware
```bash
python scripts/benchmark.py --source clip.mp4 --frames 300
```

---

## Web platform (surveillance console)

A full account-based web console on top of the pipeline — FastAPI + SQLite + FAISS:

```bash
pip install -e ".[infer,api]"
uvicorn occlubio.api.app:app --port 8001
# open http://localhost:8001  (API docs at /docs)
```

Or use the launcher, which creates the venv, installs dependencies, checks the port and opens
the browser: `.\run_server.ps1` on Windows, `./run_server.sh` on macOS/Linux (details below).

- **Two roles.** Students sign up with their **roll number** (used as the login username) and
  enroll their own face from photos or a webcam burst. Authorities sign up with a username plus
  an access code. Set it with `OCCLUBIO_AUTHORITY_CODE` (or the launcher's `-AuthorityCode` /
  `--authority-code`); if unset, the server generates a private code on first start, stores it in
  `authority_code.txt` in its data directory and prints it in the startup log. In the macOS app
  use **occlubio → Show Authority Sign-up Code**.
- **Registration is validated end-to-end** — email/password/roll-number format checks, live
  "already taken" availability feedback, and per-field duplicate errors.
- **Targeted video identification.** An authority uploads a clip and searches it for **one
  person** — an enrolled subject or an uploaded photo. Appearance windows are mapped from clip
  time to real gate time via the clip's recorded start/end.
- **Unknown-person alerts.** Every un-enrolled face found in a processed clip raises an alert.
  Each alert can be replayed in the browser: the player jumps to the moment the person appears,
  with a timeline band marking exactly when they are present, so the operator can scrub before
  and after the event.
- **Gate activity & notices.** Per-subject daily first-in/last-out records per camera, plus
  direct/broadcast notices from authorities to participants.
- The SQLite database (`occlubio.db`) is the source of truth; the FAISS index is rebuilt from it
  on startup and after every enrollment change.

## Platforms: Windows, macOS, iOS

| Platform | What you get | Status in CI |
|---|---|---|
| Windows 10/11 (x64) | Full system (server + console) | `windows-latest`: launcher, unit tests, end-to-end smoke |
| macOS 11+ (Apple Silicon and Intel) | Full system, plus a packaged `occlubio.app` / `.dmg` | `macos-latest` (arm64) + `macos-15-intel`: same as Windows, plus `.app` build and smoke test |
| iOS / iPadOS 15+ | The console as an app or Home Screen web app; **processing runs on a server** | iOS Simulator build, launched against a live server |

Platform differences, disabled features and the items that need your Apple account are in
[PLATFORM_NOTES.md](PLATFORM_NOTES.md). The porting rationale is in [PORTING_PLAN.md](PORTING_PLAN.md).

### Windows
Prerequisites: Python 3.10–3.12 from python.org (the default `python` must not be 3.13+).
```powershell
.\run_server.ps1                      # or double-click run_server.bat
.\run_server.ps1 -Port 8080 -Reload -AuthorityCode "my-secret"
.\run_server.ps1 -Bind 0.0.0.0 -CertFile lan.pem -KeyFile lan-key.pem   # HTTPS on the LAN for phones
```

### macOS
Prerequisites: Python 3.10–3.12 (`brew install python@3.12`) and the Xcode Command Line Tools
(`xcode-select --install`).
```bash
./run_server.sh                       # creates .venv, installs, serves on :8001, opens the browser
./run_server.sh --port 8080 --reload --authority-code my-secret
./run_server.sh --bind 0.0.0.0 --cert lan.pem --key lan-key.pem         # HTTPS on the LAN for phones
```
**Desktop app.** `pip install -e ".[infer,api,desktop]"` then `occlubio-desktop` opens the console
in a native window with the standard menu (Cmd+Q to quit). `--browser` uses your browser instead;
`--headless` serves only.

**Packaging `.app` + `.dmg`.** Run on the Mac whose architecture you are targeting:
```bash
packaging/macos/build_macos.sh        # -> dist/occlubio-<ver>-macos-<arm64|x86_64>.dmg
```
It builds with PyInstaller, signs (Developer ID if `APPLE_SIGNING_IDENTITY` is set, otherwise
ad-hoc), runs the end-to-end smoke test against the bundled binary, creates the `.dmg`, and
notarizes and staples it when the Apple credentials are set (see PLATFORM_NOTES.md). The app
stores its data in `~/Library/Application Support/occlubio` and downloads the face models
(~300 MB) on first launch.

### iOS / iPadOS
The phone runs the console. Detection, recognition and storage stay on a Windows/macOS/Linux
server (they can't run inside an iOS app; see PLATFORM_NOTES.md).

- **No build needed.** Start the server with `--bind 0.0.0.0` (with HTTPS for the camera), open it
  in Safari, then *Share → Add to Home Screen*.
- **Native app (Capacitor).** Prerequisites: a Mac with Xcode 16+ and Node.js 22+.
  ```bash
  cd ios-client
  npm ci
  OCCLUBIO_SERVER_URL=https://192.168.1.20:8001 npm run sync   # your server's URL
  npm run open                                                  # Xcode: pick your team, Run
  npm run build:sim                                             # command-line Simulator build
  ```
  Without `OCCLUBIO_SERVER_URL` the app points to `http://localhost:8001`, which suits the
  Simulator with a server running on the same Mac.

### Tests on any platform
```bash
pip install -e ".[infer,api,dev]"
pytest -q                     # unit tests (no model download)
python scripts/smoke_api.py   # register -> enroll -> identify end-to-end in a temp data dir
```

## What maps to which roadmap phase

| Roadmap phase (in the architecture doc) | Code here |
|---|---|
| P1 detection + tracking + alignment | `occlubio/pipeline/face_analyzer.py`, `occlubio/tracking/`, `occlubio/pipeline/aligner.py` |
| P2 baseline recognition + gallery | `occlubio/pipeline/engine.py`, `occlubio/gallery/`, `scripts/enroll.py` |
| P3 occlusion robustness | `occlubio/data/occlusion_aug.py`, `occlubio/training/`, `occlubio/pipeline/quality.py`, `occlubio/pipeline/occlusion.py` |
| P4 security & multimodal | `occlubio/pipeline/antispoof.py` (+ template-protection / gait hooks noted inline) |
| P5 optimize & deploy | `deepstream/`, `scripts/benchmark.py`, training export-to-ONNX |

## Training your own occlusion-aware model (Phase 3)
```bash
pip install -e ".[train]"     # adds torch, timm
python -m occlubio.training.train --data /path/to/aligned_faces --epochs 20 --out runs/edge_adaface
# then plug the exported ONNX back into inference:
#   set recognition.custom_onnx in configs/default.yaml to runs/edge_adaface/model.onnx
```

## Layout
```
occlubio/            # the package
  pipeline/          # detector/aligner/quality/occlusion/antispoof/engine
  tracking/          # lightweight IoU tracker (ByteTrack hook noted)
  gallery/           # FAISS 1:N enroll + search + persistence
  data/              # synthetic occlusion + photometric augmentation
  training/          # AdaFace head, dataset, train+ONNX export
  analytics/         # per-identity appearance windows + report building
  service/           # face service used by the web platform
  db/                # SQLAlchemy models + SQLite migrations
  api/               # FastAPI app (auth, enroll, identify, alerts, messages)
  platform_support.py # the only OS-specific code: data dirs, providers, devices, codecs
  desktop.py         # native-window / headless launcher used by the macOS app
web/                 # console UI (login + role-based dashboard) + PWA manifest/icons
packaging/macos/     # PyInstaller spec, entitlements, build_macos.sh (.app + .dmg)
ios-client/          # Capacitor iOS/iPadOS shell around the console
run_server.ps1/.bat/.sh  # launchers for Windows and macOS/Linux
scripts/             # enroll / recognize / identify_video / benchmark / make_demo / build_report
deepstream/          # sample DeepStream/TensorRT edge-deployment config
tests/               # smoke tests (no network/model download required)
configs/default.yaml # single source of truth for all knobs
```

## Status
- **Runs today:** detection, alignment, embedding, FAISS enroll/search, video pipeline, occlusion
  augmentation, quality gate, AdaFace training skeleton, benchmarking, and the full web platform
  (accounts, enrollment, targeted video identification, alerts with clip review, notices).
- **Hooks (you must supply weights/integration):** trained anti-spoof (MiniFASNet ONNX), trained
  occlusion-type classifier, biometric template protection, gait/body-ReID fusion, DeepStream
  C/Python deployment — integration points are described in the architecture doc.
- Verify an install with the smoke tests: `pip install -e ".[dev]"` then `pytest -q`.
- Processing is CPU-bound without a GPU (roughly 0.4 FPS on video) — raise `--stride` / the
  console's stride field for faster turnaround.

> ⚠️ This is a surveillance/biometric system. Read the responsible-use section (§0) of the
> architecture doc before any deployment: legal basis (EU AI Act / India DPDP Act), template
> protection, and bias evaluation are mandatory, not optional.

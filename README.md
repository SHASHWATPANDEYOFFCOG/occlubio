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

- **Two roles.** Students sign up with their **roll number** (used as the login username) and
  enroll their own face from photos or a webcam burst. Authorities sign up with a username plus
  an access code (set `OCCLUBIO_AUTHORITY_CODE` in the environment before starting the server).
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
web/                 # console UI (login + role-based dashboard)
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

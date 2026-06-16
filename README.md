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
scripts/             # enroll / recognize / benchmark / make_occluded_dataset
deepstream/          # sample DeepStream/TensorRT edge-deployment config
tests/               # smoke tests (no network/model download required)
configs/default.yaml # single source of truth for all knobs
```

## Honest status
- **Runs today:** detection, alignment, embedding, FAISS enroll/search, video pipeline, occlusion
  augmentation, quality gate, AdaFace training skeleton, benchmarking.
- **Hooks (you must supply weights/integration):** trained anti-spoof (MiniFASNet ONNX), trained
  occlusion-type classifier, biometric template protection, gait/body-ReID fusion, DeepStream C/Python
  deployment. Each is marked `# HOOK:` in code with what to plug in.
- This code was written but **not executed in your environment** — run the smoke tests first:
  `pip install -e .` then `pytest -q`.

> ⚠️ This is a surveillance/biometric system. Read the responsible-use section (§0) of the
> architecture doc before any deployment: legal basis (EU AI Act / India DPDP Act), template
> protection, and bias evaluation are mandatory, not optional.

# Edge deployment — DeepStream + TensorRT on Jetson Orin

This folder holds the **edge production** path (architecture §1.1, §4, §5). The Python pipeline
in `occlubio/` is your reference/training/eval implementation; on Jetson you run the same models
as **TensorRT engines inside DeepStream** for hardware-decoded, multi-camera, zero-copy inference.

## Why two implementations?
- `occlubio/` (Python + onnxruntime + FAISS): fast to iterate, trains models, validates accuracy,
  runs on your laptop / DGX Spark.
- DeepStream (C/Python + nvinfer + TensorRT): the deployable hot path. NVDEC decode, `nvstreammux`
  batching across cameras, `nvtracker`, and TensorRT engines — keeps frames on the GPU end to end.

Keep **the models identical** between the two; only the runtime changes.

## Bring-up steps (Jetson Orin NX / AGX, JetPack 6.x)
1. **Flash JetPack 6.x** (CUDA + cuDNN + TensorRT) and `sudo apt-get install deepstream-7.0`.
2. **Export models to ONNX** from the Python side:
   - Detector: SCRFD ONNX (from `insightface/detection/scrfd`) — keep the post-proc that emits 5 landmarks.
   - Recognizer: your trained model (`occlubio.training.train` already exports `model.onnx`).
3. **Build TensorRT engines** (do this *on the Jetson*, engines are not portable across devices/TRT versions):
   ```bash
   # detector — INT8 (build a calibration cache from ~1-5k representative frames)
   trtexec --onnx=scrfd_2.5g.onnx --saveEngine=scrfd_2.5g_int8.engine --int8 --fp16 \
           --calib=calib.cache
   # recognizer — keep FP16 (INT8 on FR embeddings inflates false matches; QAT-only)
   trtexec --onnx=model.onnx --saveEngine=recognizer_fp16.engine --fp16
   ```
4. **Wire into DeepStream** using `config_infer_primary_scrfd.txt` (PGIE detector) and
   `deepstream_app_config.txt` (sources + tracker + sinks). Add the recognizer as an SGIE
   (`operate-on-gie-id` = the detector) or run it via `nvinferserver` (Triton) for hot-swap.
5. **1:N search**: keep FAISS (CPU for small galleries, GPU/cuVS for large) in a sidecar process;
   feed it embeddings from the DeepStream metadata probe / message broker.

## Notes
- `nvstreammux batch-size` = number of cameras. Tune `batched-push-timeout`.
- Anti-spoof / occlusion-classifier / FIQA become additional SGIEs or a probe callback.
- Heavy "second-opinion" models (DG anti-spoof, restoration, gait) → serve from a nearby
  DGX/edge server via **Triton**, called over the network; keep only light models on-device.
- Reference apps: `marcoslucianops/DeepStream-Yolo`, `NVIDIA-AI-IOT/deepstream_reference_apps`,
  and the NVIDIA "Implementing Real-Time, Multi-Camera Pipelines with Jetson" blog.

> The two `.txt` files here are **annotated templates** — fill the engine paths, input dims, and
> the SCRFD output-parser library for your build. They are intentionally not turnkey because the
> exact dims/labels depend on which SCRFD variant and TRT version you compile.

# Occlusion-Robust Multimodal Biometric Surveillance — Reference Architecture (2024–2026)

**Target:** Real-time face recognition under masks, sunglasses, caps, helmets, scarves, side-profiles, low light, motion blur, and partial visibility.
**Train/bench:** NVIDIA DGX Spark. **Edge:** Jetson Orin NX / AGX Orin via DeepStream + TensorRT (FP16/INT8) + Triton.
**Author context:** beginner-friendly, production-oriented. Every component lists *what / paper / repo / why-over-alternatives / train / optimize / deploy*.

> ⚠️ **Your hard latency constraint was cut off in the brief** ("End-to-end latency …"). This doc assumes two operating points and you pick one: **(A) ≤ 100 ms glass-to-glass** (comfortable, multi-camera) and **(B) ≤ 30 ms per face** (aggressive). Both are achievable on Orin NX; see the budget tables.

---

## 0. Responsible-use guardrails (read first — this is a surveillance system)

Biometric surveillance is legally and ethically high-risk. Bake these in from day one or the project is not deployable:

- **Legal basis & jurisdiction.** In the EU, real-time remote biometric identification in public spaces is heavily restricted under the **EU AI Act** (high-risk / largely prohibited use). India's **DPDP Act 2023** treats biometrics as sensitive personal data. Get written authorization, a defined deployment scope, and a DPIA before any field test.
- **Restoration ≠ identification.** Generative face restoration (GFPGAN/CodeFormer/DiffBIR) **hallucinates** plausible pixels. Never use a restored face as primary evidence of identity — use it only for gallery cleanup/visualization, and always match on the *original* embedding. Document this.
- **Template protection is mandatory, not optional.** Store protected/cancelable templates, encrypt at rest, prefer on-device matching, support revocation. See §1.12.
- **Bias & failure transparency.** Measure and publish per-demographic TAR@FAR. Occlusion + low light disproportionately raises false matches for under-represented groups. Keep a human in the loop for any consequential decision.

---

## 1. Unified pipeline (the whole system on one page)

```
                          ┌─────────────────────── DeepStream pipeline (per Jetson) ───────────────────────┐
 RTSP/USB cameras ──▶ nvurisrcbin ──▶ nvstreammux (batch=N cams) ──▶ nvvideoconvert
   │
   ├─(PGIE) SCRFD face detector (TensorRT INT8) ──▶ 5-pt landmarks
   │
   ├─ nvtracker (ByteTrack/NvDCF)  ── one identity-stream per track, pick best frame
   │
   ├─(per best-frame, gated) ─▶ FIQA quality gate (CR-FIQA) ─┐
   │                                                          ├─ skip low-quality frames
   ├─ align (similarity transform to 112×112) ◀──────────────┘
   │
   ├─(SGIE-1) Occlusion/parsing head  ─▶ occlusion mask + type {mask,glasses,cap,...}
   ├─(SGIE-2) Anti-spoof (MiniFASNet) ─▶ live/spoof
   ├─(SGIE-3) Occlusion-aware FR (EdgeFace/MobileFaceNet + AdaFace + feature-mask) ─▶ 512-d embedding
   │
   └─ embedding ─▶ template protect ─▶ FAISS/Milvus 1:N search ─▶ candidate IDs
                                                                       │
   (parallel, body-level)  Person-ReID + Gait (OpenGait) ─────────────┤
                                                                       ▼
                                              Multimodal score fusion ─▶ decision + track label
                                                                       ▼
                                              Kafka/Redis msg broker ─▶ VMS / dashboard / alerts
```

**Key design principle:** detection + tracking run on *every* frame (cheap); the expensive recognition stack runs **once per track on the best, highest-quality, live frame** (gated by FIQA + anti-spoof). This is what makes multi-camera real-time feasible.

---

### 1.1 Video ingestion & decode
- **What:** NVIDIA **DeepStream SDK 7.x** on JetPack 6.x. Hardware-decode RTSP/H.264/H.265 via NVDEC; zero-copy NVMM buffers; `nvstreammux` batches all cameras into one inference batch.
- **Why DeepStream over OpenCV/GStreamer-by-hand:** NVDEC hardware decode + zero-copy keeps frames on the GPU end-to-end (no PCIe/host round-trips), gives you free multi-stream batching, built-in tracker, and a message broker. OpenCV decode on CPU will bottleneck you at 2–3 cameras.
- **Deploy:** `nvurisrcbin` (auto-reconnect for flaky RTSP) → `nvstreammux` `batch-size = #cameras`, `width/height` at detector input res. Tune `batched-push-timeout`.
- **Resource:** NVIDIA blog "Implementing Real-Time, Multi-Camera Pipelines with NVIDIA Jetson"; `marcoslucianops/DeepStream-Yolo` for clean config examples.

### 1.2 Preprocessing & alignment
- **What:** color convert + resize on GPU (`nvvideoconvert`), then **5-point similarity-transform alignment** to canonical 112×112 (the ArcFace/insightface standard) using SCRFD's landmarks.
- **Why it matters most:** alignment is the single highest-leverage step for occluded FR. Mis-alignment from a single occluded landmark wrecks the embedding. For side-profiles where only 3 landmarks are reliable, fall back to a **partial-affine** estimate or a pose-bucketed template.
- **Tip:** keep alignment math identical between training and inference. Mismatched alignment is the #1 silent accuracy killer.

### 1.3 Face detection
- **Recommendation:** **SCRFD** (Sample-and-Computation Redistribution for Face Detection), from InsightFace. Variants: SCRFD-500M / 2.5G / 10G — pick **SCRFD-2.5G** for Orin NX, 10G for AGX.
- **Paper:** Guo et al., *Sample and Computation Redistribution for Efficient Face Detection* (ICLR 2022). **Repo:** `deepinsight/insightface` → `detection/scrfd`.
- **Why over alternatives:** SCRFD gives **bbox + 5 landmarks in one shot**, is explicitly compute-redistributed for efficiency, and has battle-tested TensorRT exports. **YOLOv5-Face** (`deepcam-cn/yolov5-face`) and **YOLOv8-face** are strong alternatives and slightly easier to fine-tune for *helmet/cap-heavy* crowds, but SCRFD has the best accuracy-per-FLOP for landmark-aligned FR. **RetinaFace** is heavier with no edge advantage.
- **Occlusion angle:** fine-tune SCRFD on **WIDER FACE** + your masked/helmet imagery so heavily occluded heads still produce a box. Add a small **head/helmet detector** (YOLOv8n) in parallel so a fully helmeted person still yields a track for gait/body ReID even when no face is found.
- **Optimize:** export ONNX → TensorRT INT8 (detection tolerates INT8 well). Run as DeepStream PGIE via `nvinfer`.

### 1.4 Multi-object tracking
- **Recommendation:** **ByteTrack** (or **BoT-SORT** when you want appearance ReID in the tracker). In DeepStream use `nvtracker` with NvDCF or the ByteTrack low-confidence-association logic.
- **Paper:** Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* (ECCV 2022). **Repo:** `FoundationVision/ByteTrack`; BoT-SORT `NirAharon/BoT-SORT`.
- **Why:** ByteTrack keeps low-confidence (occluded/blurred) detections alive by associating them too — exactly the occlusion case. Tracking lets you (a) run FR once per track, (b) accumulate a *set* of embeddings per person and fuse them (set-based recognition is far more robust than single-shot under occlusion).

### 1.5 Occlusion detection & face parsing
- **Recommendation:** lightweight **occlusion-type classifier** (MobileNetV3-small head, classes: clear/mask/sunglasses/cap-shadow/scarf/hand/profile) **+** a face-parsing/segmentation map to produce a per-pixel **visibility mask** fed into the FR feature-masking module.
- **Parsing models:** **BiSeNet face-parsing** (`zllrunning/face-parsing.PyTorch`, trained on CelebAMask-HQ) for speed; **FaRL** (`FacePerceiver/FaRL`, CVPR 2022) for accuracy when you can afford it.
- **Why:** the occlusion mask is the bridge to occlusion-aware recognition (§1.8) and tells the pipeline whether to route to **periocular** matching (mask/scarf → eyes still visible) vs **lower-face** matching (sunglasses → mouth/jaw visible).

### 1.6 Face anti-spoofing (presentation-attack detection)
- **Edge recommendation:** **Silent-Face-Anti-Spoofing / MiniFASNet** (`minivision-ai/Silent-Face-Anti-Spoofing`) — passive RGB, ~1–2 ms, runs comfortably on Jetson; great default to block printed photos / replay on a screen.
- **SOTA for generalization (2024–2025):** prompt/ domain-generalization methods — **CFPL-FAS** (CVPR 2024), **BUDPT** (bottom-up domain prompt tuning, ECCV 2025), gradient-alignment & hyperbolic prototype methods (CVPR 2024). Use these as a **teacher** to distill into MiniFASNet, or run server-side in Triton for a second opinion.
- **Strongest option if hardware allows:** add an **IR / depth (stereo or ToF) sensor**. Multimodal FAS (RGB-D/IR) crushes RGB-only PAD. *Denoising and Alignment* (2025) is a good multimodal reference.
- **Why anti-spoof before FR:** in surveillance, a printed mask or phone-replay is the cheapest attack; gate recognition on a liveness pass.

### 1.7 Face restoration (optional, gallery/forensic only — NOT inline)
- **Recommendation:** **GFPGAN** (`TencentARC/GFPGAN`) or **CodeFormer** (`sczhou/CodeFormer`) for fast restoration; **DiffBIR** (`XPixelGroup/DiffBIR`, ECCV 2024) or **PMRF/ELIR** when quality > latency.
- **Why NOT in the realtime loop:** GAN/diffusion restoration is 50–500 ms/face and **hallucinates identity-bearing detail**. Use it to clean enrollment/gallery images and for human-review visualization only. Always recognize on the *original* crop. (See §0.)
- **2024–2025 note:** the NTIRE 2025 Real-World Face Restoration challenge and **FaceMe** (identity-conditioned restoration, 2025) show the field moving to identity-preserving restoration — relevant if you later want restoration that *helps* rather than misleads recognition.

### 1.8 Occlusion-aware face recognition (the core)
This is where you win or lose. Use a **three-part recipe**: strong loss + edge backbone + feature-masking.

1. **Loss — AdaFace** (CVPR 2022, `mk-minchul/AdaFace`): quality-adaptive margin that *de-emphasizes* unidentifiable (occluded/blurry) samples instead of forcing them. Empirically the best loss for low-quality/occluded/low-res. Alternative: **ArcFace** (still a fine, simpler baseline) or **AdaFace+CR-FIQA** coupling.
2. **Backbone (edge) — EdgeFace** (Idiap, *EdgeFace: Efficient Face Recognition Model for Edge Devices*, IEEE T-BIOM 2024, `otroshi/edgeface`): hybrid CNN-Transformer, won the compact track of EFaR-2023; superb accuracy-per-param. Alternatives: **MobileFaceNet** (the safe baseline), **GhostFaceNets** (`HamadYA/GhostFaceNets`, 2023), **PocketNet/MixFaceNet**. Server/teacher backbone: **ResNet-100 / IResNet-100**.
3. **Occlusion mechanism — feature masking (FROM-style):** *End2End Occluded Face Recognition by Masking Corrupted Features* (Qiu et al., TPAMI 2021, `haibo-qiu/FROM`) learns to mask feature-map regions corrupted by occlusion using the §1.5 mask. Plus **MaskNet**-style adaptive feature weighting. For masks specifically, *Seeing through the Mask: Multi-task Generative Mask Decoupling FR* (2023) is a good reference.
- **Periocular sub-model (high value when masked):** when occlusion-type = mask/scarf, also compute a **periocular (eye-region) embedding** and fuse. Eyes-visible-only is the dominant real-world surveillance case and periocular recognition closes most of the masked-FR gap.
- **Why this combo:** AdaFace handles *quality*, the feature-mask handles *spatial occlusion*, periocular handles *what's left visible*, and EdgeFace keeps it edge-deployable. No single trick is enough.
- **Output:** L2-normalized **512-d** (or 256-d, see §1.10) embedding.

### 1.9 Face image quality assessment (FIQA) — the gate
- **Recommendation:** **CR-FIQA** (CVPR 2023, `fdbtrs/CR-FIQA`) — predicts a scalar usability score from "sample relative classifiability". 2024–2025 SOTA: **CLIB-FIQA** (CVPR 2024), **MR-FIQA** (ICCV 2025), **GraFIQs** (training-free, 2024). Standard: **ISO/IEC 29794-5:2025**.
- **Use it three ways:** (1) gate frames into recognition (skip junk → saves compute), (2) weight per-frame embeddings in set-based fusion, (3) decide when to update a person's enrolled template. Cheap, huge ROI.

### 1.10 Feature compression
- **Train-time:** output **256-d** embeddings directly (train the head at 256-d) — ~half the storage/bandwidth of 512-d with negligible accuracy loss. Or learn a 512→128 projection by **distillation**.
- **Search-time:** **OPQ + IVF-PQ** (in FAISS) compresses each vector to ~16–64 bytes for billion-scale galleries. For ≤1M gallery on Jetson, keep flat FP16 and skip PQ.
- **Why:** at multi-camera, 24/7 scale, embedding storage + 1:N bandwidth dominate; compress early.

### 1.11 Retrieval / similarity search (1:N)
- **Edge / small gallery (≤1M):** **FAISS** (`facebookresearch/faiss`) `IndexFlatIP` (exact cosine) or `IndexHNSWFlat`. Runs on Jetson GPU.
- **Server / large gallery (1M–1B):** **Milvus** (`milvus-io/milvus`, GPU via NVIDIA RAFT/cuVS) or FAISS-GPU `IVF-PQ`. *Billion-Scale Similarity Search with GPUs* (Johnson et al.) is the foundational reference.
- **Why:** cosine/IP on normalized embeddings is the right metric; IVF-PQ/HNSW gives sub-ms 1:N at scale. Keep a **re-rank** step (exact cosine on top-k candidates) to recover PQ's approximation loss.

### 1.12 Biometric template protection
- **Recommendation:** **IronMask** (CVPR 2021) — modular cancelable protection that bolts onto any cosine-similarity FR model; or recent **cancelable/secure-sketch** methods (*Embedding Non-Distortive Cancelable Face Template Generation*, 2024; *IDFace*, 2025 for efficient protected 1:N). Standard: **ISO/IEC 24745**.
- **Minimum viable:** even before research-grade protection — AES-256 encryption at rest, on-device matching, salted/keyed transforms, and a revocation path. Never store raw embeddings in the clear.
- **Why:** biometrics can't be reissued. A leaked unprotected embedding is a permanent compromise; many embeddings are partially invertible back to a face.

### 1.13 Multimodal fusion
- **Add modalities that survive face occlusion:**
  - **Person-ReID (body/clothing):** `JDAI-CV/fast-reid` or `KaiyangZhou/deep-person-reid`. Works when the whole face is gone (helmet).
  - **Gait recognition:** **OpenGait** (`ShiqiYu/OpenGait`) with **DeepGaitV2 / SkeletonGait++** (TPAMI 2025). Identity from walking pattern; complements face at distance.
  - **Periocular** (§1.8) for masked faces.
- **Fusion strategy:** start with **score-level fusion** (weighted sum of cosine similarities, weights ∝ per-modality quality/availability) — simple, robust, debuggable. Move to learned **feature-level / attention fusion** only after the score-level baseline works. Always gate each modality by its own quality/availability flag (don't fuse a gait score from a standing person).
- **Why:** the entire premise — "recognize when the face is occluded" — is partly *unsolvable from the face alone* for helmets/full masks. Multimodal is how you stay robust in the worst cases.

---

## 2. Datasets by pipeline stage

| Stage | Datasets | Notes |
|---|---|---|
| FR backbone training | **WebFace4M / WebFace12M / WebFace260M**, MS1MV3, Glint360K | WebFace4M is the practical sweet spot; clean, large, license-gated. |
| FR evaluation | **IJB-B / IJB-C** (template, mixed quality), **TinyFace** (low-res), **LFW/CFP-FP/AgeDB** (sanity) | IJB-C TAR@FAR=1e-4/1e-5 is the headline metric. |
| Occluded FR | **O-LFW / LFW-OCC**, **MFR2**, occluded variants via **MaskTheFace** augmentation | Generate occlusions synthetically (masks/glasses/caps) for training. |
| Masked FR | **RMFRD/SMFRD**, **MFR2**, **MFV**, **LFW-SM** | RMFRD = 90k unmasked + 5k masked / 525 IDs. |
| Cross-pose / profile | **CFP-FP**, **CPLFW**, Multi-PIE | Profile is its own failure mode; weight it in eval. |
| Low-res / surveillance | **TinyFace**, **QMUL-SurvFace**, **SCFace** | Match your real camera resolution/standoff. |
| Face detection | **WIDER FACE**, **MAFA** (masked faces) | MAFA is the masked-face detection set. |
| Face parsing | **CelebAMask-HQ**, **LaPa**, **Helen** | For BiSeNet/FaRL parsing. |
| Anti-spoofing | **CelebA-Spoof**, **CASIA-SURF (CeFA)**, **OULU-NPU**, **SiW-Mv2**, **WMCA** (RGB-D/IR) | Use cross-dataset protocol to prove generalization. |
| Helmet / PPE | **GDUT-HWD**, hardhat/PPE sets (Roboflow) | For the parallel helmet detector. |
| Gait | **CASIA-B**, **OU-MVLP**, **GREW**, **Gait3D**, **SUSTech1K** | GREW/Gait3D are the in-the-wild ones. |
| Multi-camera tracking / ReID | **Market-1501**, **MSMT17**, **MOT17/20**, **MTMC (AIC City)** | For ReID + cross-camera association. |

**Licensing reality check:** WebFace260M, MS-Celeb-derived sets, and several face datasets have research-only / withdrawn licenses. For a *production* system, confirm you have rights for commercial use or build a consented enrollment gallery. This is a legal blocker, not a footnote.

---

## 3. Train → distill → quantize → benchmark (on DGX Spark)

> **DGX Spark reality:** the GB10 Grace-Blackwell desktop (128 GB unified LPDDR5X, ~1 PFLOP FP4) is excellent for **fine-tuning, distillation, INT8 calibration, and benchmarking**, and for prototyping the whole pipeline. Training a ResNet-100 from scratch on WebFace12M is heavy for a single Spark — either (a) rent A100/H100 hours for the teacher, or (b) train an EdgeFace/MobileFaceNet-scale model directly, which Spark handles well. Use Spark as your daily driver and the cloud only for the big teacher run.

**Step 1 — Teacher.** Train (or download) a strong teacher: IResNet-100 + AdaFace on WebFace12M. This defines your accuracy ceiling.

**Step 2 — Occlusion-aware student.** Train EdgeFace/MobileFaceNet with AdaFace + the feature-masking head. **Augment aggressively**: MaskTheFace masks, synthetic sunglasses/cap/scarf overlays, random erasing, motion blur, low-light gamma/noise, JPEG, downscale-to-surveillance-res. Curriculum: start clean, ramp occlusion probability to ~50%.

**Step 3 — Distillation.** Distill teacher → student: cosine/L2 **feature distillation** on embeddings (+ optional logit/relation KD). This recovers most of the accuracy lost to the small backbone. Methods to read: margin-based KD for FR, ARoFace (augmentation-robust). Train periocular and (optionally) gait models in parallel.

**Step 4 — Quantization.**
- **Detector, parser, anti-spoof, FIQA → INT8** (PTQ via TensorRT with a ~1–5k representative-image calibration set). These tolerate INT8 well.
- **FR embedding model → FP16 first.** Embeddings are sensitive; INT8 can shift cosine distances and inflate false matches. If you need INT8 FR, use **QAT** (quantization-aware training) and re-verify TAR@FAR, don't just PTQ.
- Tooling: TensorRT `trtexec`, `polygraphy`, NVIDIA **Model Optimizer** (PTQ/QAT/sparsity).

**Step 5 — Benchmark (accuracy + systems together).**
- Accuracy: IJB-C TAR@FAR, TinyFace rank-1, masked/occluded sets, **per-demographic** breakdown, and a **DET/ROC at your operating FAR**. Track FP16-vs-INT8 delta explicitly.
- Systems: latency (p50/p95/p99), throughput (FPS × #cameras), VRAM, power (`tegrastats`), thermals. Measure **on the actual Jetson**, not the Spark — Spark numbers don't transfer.

---

## 4. Hardware / latency / memory budgets

Per-face stage budget (INT8 detector, FP16 FR), order-of-magnitude on **Orin NX 16GB**; AGX Orin is ~2–3× faster:

| Stage | Model | Precision | ~Latency (Orin NX) | Notes |
|---|---|---|---|---|
| Detection | SCRFD-2.5G | INT8 | 2–5 ms/frame | whole frame, all faces |
| Tracking | ByteTrack/NvDCF | — | <1 ms | per frame |
| Align | similarity transform | — | ~0.5 ms | GPU |
| FIQA gate | CR-FIQA (small) | INT8 | ~1 ms | optional |
| Anti-spoof | MiniFASNet | INT8 | 1–2 ms | per candidate |
| Parsing/occ | BiSeNet-small | INT8 | 2–4 ms | run only when needed |
| **FR embed** | EdgeFace/MobileFaceNet | FP16 | 1–3 ms/face | the core |
| 1:N search | FAISS HNSW (≤1M) | — | <1 ms | per query |

- **Per-face critical path** ≈ **8–15 ms** → both operating points (≤100 ms, ≤30 ms/face) are met with margin because FR runs once per track, not per frame.
- **Multi-camera:** Orin NX comfortably runs **~4–8 cameras @ 1080p/15–30 fps** with this design; AGX Orin **~12–24+**. Scale by adding Jetsons, not by overloading one.
- **Memory:** keep total engine VRAM < 60% of board (leave headroom for decode surfaces + tracker + FAISS). On Orin NX 16GB you have ample room for this model set.

---

## 5. Serving topology
- **On Jetson:** DeepStream `nvinfer` for the in-graph models (detector, FR, anti-spoof) compiled to TensorRT engines. Keep the hot path in-process.
- **Triton Inference Server:** use via DeepStream `nvinferserver` to (a) hot-swap/version models without rebuilding the pipeline, (b) serve the heavier "second-opinion" models (DG anti-spoof, restoration, gait) from a nearby **DGX/edge server**, (c) dynamic-batch across cameras. Run light models on-device, heavy models on Triton.
- **Messaging:** DeepStream message broker → **Kafka/Redis** → dashboard, alerting, and VMS integration.

---

## 6. Step-by-step roadmap (beginner-friendly, ~6 phases)

**Phase 0 — Foundations (week 1–2).** Stand up JetPack 6.x + DeepStream on the Jetson; run a stock DeepStream sample with one RTSP camera. Set up the DGX Spark Python/CUDA env (PyTorch, insightface, ONNX, TensorRT). Goal: *frames flowing end-to-end with a toy model*.

**Phase 1 — Detection + tracking + alignment (week 3–4).** Deploy SCRFD (TensorRT) as PGIE; add ByteTrack; verify aligned 112×112 crops come out correctly. Goal: *clean aligned faces per track from live video*.

**Phase 2 — Baseline recognition (week 5–7).** Use a **pretrained** AdaFace/ArcFace model first (no training!). Build the FAISS gallery, do 1:N, measure baseline accuracy on your own footage. Goal: *recognition works on easy frames* — this is your control.

**Phase 3 — Occlusion robustness (week 8–12).** On DGX Spark: train the occlusion-aware student (AdaFace + EdgeFace + feature-mask + heavy occlusion augmentation), distill from a teacher, add the FIQA gate, occlusion classifier, and periocular sub-model. Re-measure on masked/occluded/profile/low-light eval sets. Goal: *measurable lift on occluded sets vs Phase 2*.

**Phase 4 — Security & multimodal (week 13–16).** Add anti-spoofing, template protection, and one extra modality (gait or body-ReID) with score-level fusion. Goal: *robust to spoofing and to full-face occlusion*.

**Phase 5 — Optimize & harden (week 17–20).** Quantize (INT8 where safe, QAT for FR if needed), benchmark on-device (latency/FPS/power/thermal), scale to N cameras, add Triton for model management, set up logging/metrics. Run a **field pilot** with the legal/DPIA paperwork from §0. Goal: *production-grade, measured, authorized deployment*.

**Golden rule for a beginner:** always keep a *pretrained baseline* working and benchmark every change against it. Don't train your own model until the pretrained pipeline runs end-to-end on real video.

---

## 7. Common pitfalls (the ones that will actually bite you)
1. **Alignment mismatch** between train and inference → silent accuracy collapse. Pin one alignment routine.
2. **INT8 on the FR embedding** → inflated false matches. Keep FR FP16 unless QAT-verified.
3. **Restoration in the hot loop** → latency blowup + hallucinated identities. Gallery-only.
4. **Benchmarking on Spark, deploying on Jetson** → numbers don't transfer. Always measure on target.
5. **Single-frame recognition** → fragile under occlusion. Use set-based, track-level fusion.
6. **Ignoring dataset licenses & deployment law** → the project can't ship. Resolve early.
7. **No per-demographic eval** → undetected bias, especially under occlusion/low-light.

---

## 8. Curated link index
- InsightFace (SCRFD, ArcFace, partial-fc): https://github.com/deepinsight/insightface
- AdaFace (CVPR 2022): https://github.com/mk-minchul/AdaFace
- EdgeFace (IEEE T-BIOM 2024): https://github.com/otroshi/edgeface
- GhostFaceNets: https://github.com/HamadYA/GhostFaceNets
- FROM — occluded FR (TPAMI 2021): https://github.com/haibo-qiu/FROM
- YOLOv5-Face: https://github.com/deepcam-cn/yolov5-face
- ByteTrack (ECCV 2022): https://github.com/FoundationVision/ByteTrack ; BoT-SORT: https://github.com/NirAharon/BoT-SORT
- Face parsing (BiSeNet): https://github.com/zllrunning/face-parsing.PyTorch ; FaRL: https://github.com/FacePerceiver/FaRL
- Silent-Face Anti-Spoofing (MiniFASNet): https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- GFPGAN: https://github.com/TencentARC/GFPGAN ; CodeFormer: https://github.com/sczhou/CodeFormer ; DiffBIR (ECCV 2024): https://github.com/XPixelGroup/DiffBIR
- CR-FIQA (CVPR 2023): https://github.com/fdbtrs/CR-FIQA
- OpenGait (DeepGaitV2/SkeletonGait++, TPAMI 2025): https://github.com/ShiqiYu/OpenGait
- FAISS: https://github.com/facebookresearch/faiss ; Milvus: https://github.com/milvus-io/milvus
- FastReID: https://github.com/JDAI-CV/fast-reid ; deep-person-reid: https://github.com/KaiyangZhou/deep-person-reid
- MaskTheFace (occlusion augmentation): https://github.com/aqeelanwar/MaskTheFace
- DeepStream YOLO configs: https://github.com/marcoslucianops/DeepStream-Yolo
- NVIDIA Jetson multi-camera pipelines: https://developer.nvidia.com/blog/implementing-real-time-multi-camera-pipelines-with-nvidia-jetson/

*Verify exact repo paths/weights before depending on them; a few URLs above are reconstructed from memory and should be confirmed.*

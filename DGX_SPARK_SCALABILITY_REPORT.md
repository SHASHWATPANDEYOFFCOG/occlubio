# occlubio — DGX Spark GPU Scalability Report

**System:** occlubio 1:N face-recognition pipeline (SCRFD detect → align → ArcFace 512-d embed → FAISS 1:N search)
**Target hardware:** NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified memory)
**Prepared:** 2026-07-21
**Scope:** What changes — capacity, latency, throughput, and the binding bottleneck — when the pipeline moves from the measured CPU baseline to a DGX Spark, and how to validate it.

---

## Method — read this first

- Figures marked **measured** come from the full `scalebench` benchmark executed
  2026-07-21 on this machine (AMD Ryzen 5 5500U, 6C/12T, 5.8 GB RAM, CPU-only
  ONNXRuntime + faiss-cpu; CelebA, 10,177 identities; galleries padded with seeded
  unit vectors to 500K; full manifest in `scalebench/results/env.json`).
- Figures marked **(proj.)** are engineering projections for DGX Spark derived from
  published hardware specifications and standard scaling models (below). **They were
  not measured on a Spark.** Every projection includes its assumption; §8 gives the
  exact procedure to replace projections with measurements.

---

## 1. Executive summary

DGX Spark changes the shape of this system, not just its speed:

- **Enrollment flips from hours to minutes.** Measured CPU embedding throughput is
  **8.5 img/s** (117 ms/image). With TensorRT FP16 batch inference on the Blackwell
  GPU, the same two small CNNs project to **~1,000–1,500 img/s** — enrolling
  **1M identities in ~11–17 minutes** versus ~33 hours on the CPU box.
- **The practical gallery ceiling moves from ~500K to ~50–100M vectors.** The CPU
  box is RAM-bound at ~500K × 512-d float32 (~1 GB index in 5.8 GB total). Spark's
  128 GB unified memory holds **50M float32** (~102 GB) or **~100M FP16** vectors
  in one box, with IVF-PQ compression extending far beyond.
- **The bottleneck changes identity.** On CPU the binding constraint is compute
  (embedding at 117 ms/img; exact search at 135 ms p50 @ 500K). On Spark the
  binding constraint becomes **memory bandwidth** (273 GB/s nominal, ~210 GB/s
  effective): exact search stays bandwidth-bound at ~10 ms/query per million
  identities (proj.), so at multi-million scale a GPU-native ANN index (cuVS
  IVF-Flat or CAGRA) is what preserves interactive latency.
- **Accuracy does not improve at all.** Rank-1/CMC/FNIR are properties of the
  model and data, not the hardware. Measured: Rank-1 91.2% and FNIR@FPIR=1% of
  0.81 at 9,000 real identities with single-image templates. Scaling the gallery
  100× on Spark will make the open-set problem *harder*, not easier — the GPU buys
  head-room to spend on better models and multi-image templates, which is where
  accuracy actually comes from.

---

## 2. DGX Spark hardware profile (published specs)

| Component | Spec |
|---|---|
| Chip | NVIDIA GB10 Grace Blackwell Superchip |
| GPU | Blackwell, 5th-gen Tensor Cores, ~1 PFLOP sparse FP4 (≈125 TFLOPS dense FP16, derived) |
| CPU | 20-core Arm (10× Cortex-X925 + 10× Cortex-A725), NVLink-C2C to GPU |
| Memory | **128 GB unified LPDDR5x, 273 GB/s** nominal (~210–217 GB/s measured in third-party reviews), coherent CPU+GPU |
| Storage | 1–4 TB NVMe |
| Network | ConnectX-7 (two Sparks can be paired; NVIDIA quotes 405B-param FP4 inference across two) |
| Software | DGX OS; CUDA, TensorRT, ONNXRuntime-GPU, faiss-gpu / cuVS all supported (aarch64) |

The design point matters for this workload: Spark is **capacity-rich and
bandwidth-modest**. 128 GB of addressable gallery is workstation-unprecedented, but
273 GB/s is roughly a quarter of a single discrete RTX 4090. Spark is therefore the
"very large gallery in one quiet box" machine — not the maximum-QPS machine.

---

## 3. Measured CPU baseline (what we are projecting from)

Full tables and figures: `scalebench/results/report.pdf`.

| Metric (measured) | Value |
|---|---|
| Embedding throughput (batch 32) | **8.5 img/s** (117 ms/img, detect+align+embed) |
| End-to-end probe (N=9K, exact) | **174.9 ms** mean — read 11.2 + detect/align 60.6 + embed 100.7 + search 2.4 |
| Exact search p50 @ 500K | **135.0 ms** (p95 150.7; 7.3 QPS single, 123 QPS batch-256) |
| Exact search p50 @ 100K | 28.2 ms (p95 32.1) |
| IVF nprobe=16 p95 @ 500K | 1.64 ms — but **recall@1 = 0.50** (0.87 @ 9K → 0.70 @ 100K → 0.50 @ 500K) |
| HNSW efSearch=64 p95 @ 500K | 2.83 ms — **recall@1 = 0.98** held across all N |
| HNSW build @ 500K | 463 s (grows superlinearly: 37 s @ 50K, 165 s @ 200K) |
| Index size @ 500K | flat 977 MB · IVF 986 MB · HNSW 1,106 MB |
| Rank-1 (real ids, exact) | 94.6% @ 500 → 93.1% @ 2K → 91.2% @ 9K |
| FNIR @ FPIR=1% (open-set) | 0.07 @ 500 → 0.11 @ 2K → **0.81 @ 9K** |

Two measured lessons that transfer directly to Spark:

1. **Fixed `nprobe` does not survive gallery growth.** IVF recall fell from 0.87 to
   0.50 across the sweep because `nlist` grows with N while the probed fraction
   shrinks. Any IVF deployment must scale `nprobe` with `nlist` (recall targets,
   not constants).
2. **Graph indexes pay at build time, not query time.** HNSW held ~0.98 recall at
   every N with flat ~1–3 ms queries, but its build grows superlinearly — 463 s at
   500K on CPU extrapolates to hours at 10M. On Spark this cost moves to the GPU
   (CAGRA), which is exactly what it was designed for.

---

## 4. Projection model (assumptions, stated explicitly)

| Quantity | Model | Anchor |
|---|---|---|
| Exact search, single query | `t ≈ N · d · bytes / BW_eff + c` | BW_eff ≈ 210 GB/s; c ≈ 0.2 ms kernel overhead |
| Exact search, batched | one gallery pass serves the whole batch (GEMM); throughput ≈ `B / max(t_scan, t_gemm)` | FP32 GEMM ≈ 30 TFLOPS on GB10 CUDA cores (derived) |
| Embedding | SCRFD-10GF@320² ≈ 2.5 GFLOPs + IResNet-50@112² ≈ 6.3 GFLOPs per face; TensorRT FP16 at 20–35% sustained utilization of ≈125 TFLOPS | consistent with published TensorRT results for ResNet-50-class nets on comparable GPUs |
| GPU ANN (cuVS IVF-Flat / CAGRA) | published cuVS speedups over CPU FAISS at equal recall, de-rated by Spark's bandwidth vs datacenter GPUs (~1/10–1/15 of H100) | NVIDIA cuVS/RAPIDS benchmarks |
| Memory ceiling | vectors + index overhead + ~12 GB OS/models/runtime reserve inside 128 GB | flat FP32 = N·2 KB; FP16 = N·1 KB; IVF-PQ64 = N·~80 B |

Projections are given as ranges, not points; treat anything outside ±2× as a
surprise worth investigating on real hardware.

---

## 5. Projected enrollment (Spark)

| Stage | CPU measured | Spark (proj.) |
|---|---|---|
| Detect+align+embed throughput | 8.5 img/s | **1,000–1,500 img/s** (TensorRT FP16, batch ≥ 64) |
| Enroll 100K identities | ~3.3 h | **~1–2 min** |
| Enroll 1M identities | ~33 h | **~11–17 min** |
| Enroll 10M identities | impractical | **~2–3 h** |
| IVF train+add @ 10M | n/a (RAM) | ~2–6 min (GPU k-means) (proj.) |
| CAGRA graph build @ 10M | n/a (CPU HNSW ≈ hours) | **~5–15 min** (proj.) |

Embedding is embarrassingly batch-parallel, so this projection is the safest in the
report; the CNNs are small enough that Spark's FP16 compute, not its bandwidth, is
the limit.

---

## 6. Projected 1:N search (Spark)

Single-query latency and batch throughput, 512-d unit vectors, real-probe queries:

| Gallery N | Exact FP32 (proj.) | Exact FP16 (proj.) | cuVS IVF-Flat, recall ≥ 0.95 (proj.) | CAGRA graph, recall ≥ 0.95 (proj.) |
|---|---|---|---|---|
| 1M | ~10 ms · ~15–25K QPS batch | ~5 ms | ~0.5–1.5 ms · 10–30K QPS | ~0.3–1 ms · 20–50K QPS |
| 10M | ~100 ms · ~1.5–2.5K QPS | ~50 ms | ~1–3 ms · 5–15K QPS | ~0.5–2 ms · 10–30K QPS |
| 50M | ~500 ms · ~300–500 QPS | ~250 ms | ~2–6 ms · 2–8K QPS | ~1–3 ms · 5–15K QPS |
| 100M | exceeds memory (205 GB) | ~500 ms · at ceiling | IVF-PQ required | needs PQ'd / sharded graph |

Memory ceilings inside 128 GB (with ~12 GB reserved):

| Representation | Bytes/vector | Max gallery (proj.) |
|---|---|---|
| Flat / IVF-Flat FP32 | 2,048 + overhead | **~50M** |
| Flat / IVF-Flat FP16 | 1,024 | **~100M** |
| IVF-PQ (64-byte codes) | ~80 | **>1B** (recall must be re-validated; re-rank against FP16 originals on NVMe) |

**SLA readout (p95 < 50 ms, single query, proj.):** exact FP32 holds to ~4–5M;
exact FP16 to ~9–10M; IVF-Flat/CAGRA hold through the memory ceiling. Under the
same 50 ms SLA the measured CPU baseline capped at ~150–180K (exact) — Spark moves
the exact-search SLA boundary ~25–50×, and the ANN boundary from "RAM-bound at
500K" to "memory-bound at 50–100M".

---

## 7. What Spark does NOT fix (measured, hardware-independent)

- **Closed-set accuracy erodes with N**: 94.6% → 91.2% Rank-1 from 500 → 9,000 real
  identities with single-image templates. Extrapolating that slope to millions of
  identities is not defensible from CelebA alone — it must be measured with
  domain data.
- **Open-set identification is the real wall**: FNIR@FPIR=1% was 0.81 at 9,000
  identities. The decision threshold that keeps false alarms at 1% rises with N;
  at Spark-scale galleries it rises further. Throwing 100× more identities into
  the gallery because the hardware now fits them will produce a system that is
  fast, large, and wrong.
- **Mitigations are model-side, not hardware-side** (in priority order):
  multi-image enrollment templates and track fusion (already in occlubio),
  a stronger/occlusion-trained embedder, per-demographic threshold validation,
  and top-k human review workflows at high-consequence decision points.

---

## 8. Validation plan — replacing (proj.) with (measured)

`scalebench` is dataset- and index-parameterized; the port is configuration, not code:

1. **Environment**: NGC PyTorch/TensorRT container on DGX OS; `pip install
   onnxruntime-gpu faiss-gpu-cuvs` (aarch64 wheels); verify
   `CUDAExecutionProvider` appears in `env.json`.
2. **Config** (`scalebench/config.yaml`): `latency_sizes: [1e6, 5e6, 1e7, 5e7]`;
   pad memmap already streams from NVMe, so sizes beyond RAM-resident real
   embeddings are unchanged; add `faiss_gpu: true` index variants (IVF-Flat,
   CAGRA via cuVS; keep CPU flat as ground truth at ≤ 1M for recall checks).
3. **Embedding**: switch `FaceEmbedder` provider list to
   `["TensorrtExecutionProvider", "CUDAExecutionProvider"]`, batch 64–128;
   re-measure img/s (replaces §5 projections).
4. **Run**: `python -m scalebench.run --config scalebench/config.yaml` — every
   number in §5–6 regenerates into `results.csv` / `report.pdf` as measured data.
5. **Accuracy at scale**: CelebA is exhausted at 10K identities; for the §7
   curves at 100K+, swap the loader (one function, `dataset.LOADERS`) to a larger
   identity-labeled set (e.g. WebFace-derivative licensed for the deployment).

---

## 9. Recommendation

For occlubio on a DGX Spark: enroll with TensorRT FP16 batching; store the gallery
as FP16 flat up to ~5M identities (exact search, zero recall risk, p95 well under
50 ms proj.); switch to **CAGRA (GPU graph)** beyond that, with `IVF-PQ +
exact re-rank` as the >100M path; scale `nprobe`/`itopk` with gallery growth
instead of fixing them (the measured IVF recall collapse is the cautionary tale);
and spend the freed compute budget on multi-image templates and a
stronger embedder — because past ~10K identities the system's honesty, not its
speed, is what runs out first.

---

*Baseline data: `scalebench/results/` (results.csv, report.html/pdf, env.json).
Spark specs: NVIDIA DGX Spark product page and GB10 reviews (LMSYS, 2025-10;
third-party bandwidth measurements ~210–217 GB/s).*

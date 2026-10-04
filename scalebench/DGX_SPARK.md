# Running the CelebA scalability test on DGX Spark

The projection report ([../DGX_SPARK_SCALABILITY_REPORT.md](../DGX_SPARK_SCALABILITY_REPORT.md))
predicts what a Spark should do; this runbook is how you **measure** it. One
command, from the repo root, on the Spark:

```bash
bash scalebench/dgx_spark_setup.sh
```

That installs the GPU stack (onnxruntime-gpu, faiss-gpu-cuvs), downloads CelebA
(202,599 images via kagglehub + the MD5-verified `identity_CelebA.txt`), and runs
`python -m scalebench.run --config scalebench/config.dgx-spark.yaml`.

## What the Spark config changes

| Aspect | CPU baseline (`config.yaml`) | Spark (`config.dgx-spark.yaml`) |
|---|---|---|
| Embedding | CPU ONNX, batch 32 (8.5 img/s measured) | TensorRT/CUDA, batch 128 |
| Sweep max N | 500K (5.8 GB RAM ceiling) | **50M** padded (~102 GB in 128 GB unified) |
| Flat / IVF | CPU | **GPU** (`gpu: true` → `index_cpu_to_gpu`) |
| HNSW | product path | CPU reference, capped at N ≤ 1M |
| CAGRA | n/a | **enabled** (GPU graph, cuVS build) |
| IVF tuning | nprobe 1–64, nlist ≤ 4,096 | nprobe 8–512, nlist ≤ 32,768 |

Accuracy sweeps are identical in both configs (500 → 9,000 real identities) —
accuracy is model-bound, not hardware-bound, and CelebA caps at 10,177 identities.

## Expected runtime (from the projection report; verify)

- Embedding ~18.5K images: **~20–60 s** on TensorRT (vs 36 min on the CPU box).
  First TensorRT run adds a few minutes of engine build — the embedding cache
  absorbs it on reruns.
- Pad memmap generation to 50M × 512-d float32: ~102 GB written to NVMe once
  (tens of minutes, reused across all indexes and reruns).
- Index sweeps: GPU flat and IVF are minutes; CAGRA builds at 50M are the long
  pole (projected ~15–45 min).

## Reading the results against the projections

`scalebench/results-dgx-spark/report.html` regenerates every figure/table from
measured data. Compare directly against §5–6 of the projection report:

- **Embedding img/s** vs the projected 1,000–1,500.
- **GPU-flat p95 vs N** vs the ~10 ms-per-million bandwidth model (`gpu=1` rows
  in `results.csv`).
- **CAGRA recall@1 + p95** vs the CPU HNSW reference at N ≤ 1M, then alone
  above it.
- **The SLA table**: largest N with p95 < 50 ms per index — the projection says
  ~4–5M (exact FP32) and memory-bound for CAGRA/IVF.

Anything more than ~2× off the projections is worth a look (thermals, provider
fallback to CPU, cuVS wheel not actually GPU-enabled — check the `[2/5]` sanity
output and `env.json`).

## Troubleshooting

- `faiss GPU: False` in the sanity check → the cuVS wheel didn't match this
  CUDA; use NVIDIA's NGC RAPIDS/pytorch container and `pip install -e .` the
  repo inside it, or conda `faiss-gpu-cuvs`.
- `TensorrtExecutionProvider` missing → onnxruntime-gpu without TRT is fine
  (CUDA EP is ~70–80% of TRT for these small CNNs); the provider list falls
  through automatically.
- OOM near 50M: lower `latency_sizes` to 25M — flat FP32 at 50M plus CAGRA
  working memory is close to the 128 GB ceiling with the OS resident.
- CAGRA rows missing from the report: the guard skipped it (no `GpuIndexCagra`
  in the installed faiss) — see the run log.

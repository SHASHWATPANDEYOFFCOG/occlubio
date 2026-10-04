# scalebench — 1:N face-recognition scalability benchmark

Measures how a detect → embed → 1:N-search pipeline scales with enrolled
gallery size N, on CelebA, comparing FAISS `IndexFlatIP` (exact) against
`IndexIVFFlat` and `IndexHNSWFlat` (approximate).

## Run

```bash
# full benchmark (config-driven), writes scalebench/results/
python -m scalebench.run --config scalebench/config.yaml

# smoke test (~2 min)
python -m scalebench.run --config scalebench/config.yaml --quick

# DGX Spark (GPU) variant — see DGX_SPARK.md
bash scalebench/dgx_spark_setup.sh

# individual stages
python -m scalebench.run --stages accuracy,report
```

Everything is controlled by `config.yaml` (model, det size, index parameters,
sweep sizes, dataset paths, SLA targets). Embeddings are cached in
`results/cache/embeddings.npz`, so reruns are fast.

## Outputs (`scalebench/results/`)

| File | Contents |
|---|---|
| `results.csv` | one row per (stage, N, index, params, metric) |
| `report.html` / `report.pdf` | figures + tables + written analysis |
| `env.json` | hardware, versions, full config, seed |
| `split.json` / `settled_split.json` | the exact reproducible split |
| `skipped_images.txt` | images with no detectable face |
| `curves/acc_N*.npz` | CMC + DET raw curves per gallery size |

## Design notes

- **Accuracy vs latency scaling are separated**: Rank-k / CMC / DET sweeps use
  only real identities (CelebA caps at 10,177); latency/throughput/memory
  sweeps pad the gallery with seeded unit-normalized random vectors (search
  cost is independent of vector provenance; probes stay real).
- **Dataset-agnostic**: the only CelebA-specific code is
  `dataset.load_identity_map_celeba`; register a new loader in
  `dataset.LOADERS` to benchmark LFW etc.
- Galleries are nested (N=500 ⊂ N=1000 ⊂ …) and fully seeded, so runs are
  reproducible and per-identity results comparable across N.

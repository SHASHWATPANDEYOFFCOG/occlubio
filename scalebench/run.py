"""End-to-end benchmark orchestrator.

Single command:
    python -m scalebench.run --config scalebench/config.yaml
Optional:
    --stages embed,accuracy,scale,e2e,report   (default: all)
    --quick                                    (tiny smoke-test sweep)

Stages:
  split     seeded split, logged to results/split.json
  embed     detect+align+embed with disk cache -> real gallery/probe matrices
  accuracy  Rank-k / CMC / open-set DET on REAL identities (exact index)
  scale     latency / throughput / memory / ANN-recall sweep on padded galleries
  e2e       full detect+embed+search probe latency
  report    figures + HTML + PDF (scalebench.report)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import sys
import time
import importlib.metadata as im

import numpy as np
import yaml

from . import bench, dataset, faiss_indexes, metrics

log = logging.getLogger("scalebench")

STAGE_ORDER = ["split", "embed", "accuracy", "scale", "e2e", "report"]


# ---------------------------------------------------------------- utilities

def load_config(path: str, quick: bool) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Paths may start with ~ (e.g. kagglehub's ~/.cache) on every OS.
    for section, key in (("dataset", "images_dir"), ("dataset", "identity_file"),
                         ("paths", "results_dir")):
        if cfg.get(section, {}).get(key):
            cfg[section][key] = os.path.expanduser(cfg[section][key])
    if quick:
        cfg["split"].update(n_nonmated=100, max_gallery=300, max_mated_probes=100)
        cfg["sweep"] = {"accuracy_sizes": [200], "latency_sizes": [200, 2000]}
        cfg["measure"].update(max_single_queries=50, single_query_budget_s=5,
                              e2e_probe_images=5)
    return cfg


def collect_env(cfg: dict) -> dict:
    import psutil
    vers = {}
    for p in ["numpy", "faiss-cpu", "insightface", "onnxruntime",
              "opencv-python", "matplotlib", "pandas", "psutil", "pyyaml"]:
        try:
            vers[p] = im.version(p)
        except Exception:
            vers[p] = "?"
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "gpu": "none (CPU-only ONNXRuntime + faiss-cpu)",
        "python": sys.version.split()[0],
        "versions": vers,
        "embedding_dim": cfg["embedding"]["dim"],
        "seed": cfg["seed"],
        "config": cfg,
    }


class Results:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, stage, N, index, params, metric, value, unit):
        self.rows.append(dict(stage=stage, N=N, index=index, params=params,
                              metric=metric, value=value, unit=unit))

    def save(self, path: str):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["stage", "N", "index", "params",
                                               "metric", "value", "unit"])
            w.writeheader()
            w.writerows(self.rows)
        log.info("wrote %d result rows -> %s", len(self.rows), path)


# ---------------------------------------------------------------- embed stage

def settle_embeddings(cfg, split, id_map, cache, res: Results, rdir: str):
    """Embed candidates with detection-failure fallback; build real matrices.

    Returns (gallery_embs [G,512], probe_embs [P,512], nonmated_embs [Q,512],
    usable_identities list). Row i of gallery/probe belongs to identity i of
    the settled enrollment order, so galleries of any size N are embs[:N].
    """
    from .embedder import FaceEmbedder

    max_g = cfg["split"]["max_gallery"]
    max_p = cfg["split"]["max_mated_probes"]
    dim = cfg["embedding"]["dim"]

    wanted: list[str] = []
    for ident in split.enroll_order:
        wanted.extend(split.candidates[ident][:2])
    by_nm_id: dict[str, list[str]] = {}
    for p in split.nonmated_pool:
        by_nm_id.setdefault(id_map[p], []).append(p)
    wanted.extend(v[0] for v in by_nm_id.values())

    embedder = None
    stats_path = os.path.join(rdir, "embed_stats.json")
    acc = {"images": 0, "seconds": 0.0, "skipped": 0}  # cumulative this run

    def ensure(paths: list[str]):
        nonlocal embedder
        todo = cache.missing(paths)
        if not todo:
            return
        if embedder is None:
            e = cfg["embedding"]
            embedder = FaceEmbedder(e["model"], e["det_size"], e["batch_size"],
                                    e.get("providers"))
        t0 = time.perf_counter()
        embs, skipped = embedder.embed_paths(todo)
        dt = time.perf_counter() - t0
        cache.add(embs, skipped)
        cache.save()
        acc["images"] += len(todo)
        acc["seconds"] += dt
        acc["skipped"] += len(skipped)
        stats = {"images": acc["images"], "seconds": acc["seconds"],
                 "img_per_s": acc["images"] / acc["seconds"] if acc["seconds"] else 0,
                 "ms_per_img": 1000 * acc["seconds"] / acc["images"] if acc["images"] else 0,
                 "skipped": acc["skipped"]}
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=1)
        log.info("embedded %d fresh images in %.1fs (%.1f img/s, %d no-face)",
                 len(todo), dt, len(todo) / dt if dt else 0, len(skipped))

    ensure(wanted)

    # Fallback pass: identities whose first two candidates did not both embed.
    retry: list[str] = []
    for ident in split.enroll_order:
        cands = split.candidates[ident]
        ok = [c for c in cands[:2] if c in cache.embs]
        if len(ok) < 2:
            retry.extend(c for c in cands[2:4])
    for ident, paths in by_nm_id.items():
        if paths[0] not in cache.embs and len(paths) > 1:
            retry.append(paths[1])
    if retry:
        log.info("fallback candidates after detection failures: %d", len(retry))
        ensure(retry)

    usable, g_rows, p_rows = [], [], []
    dropped = 0
    for ident in split.enroll_order:
        oks = [c for c in split.candidates[ident] if c in cache.embs]
        if len(oks) >= 2:
            usable.append(ident)
            g_rows.append(cache.embs[oks[0]])
            p_rows.append(cache.embs[oks[1]])
        else:
            dropped += 1
        if len(usable) >= max_g:
            break

    nm_rows = []
    for ident, paths in by_nm_id.items():
        for p in paths:
            if p in cache.embs:
                nm_rows.append(cache.embs[p])
                break
        if len(nm_rows) >= cfg["split"]["n_nonmated"]:
            break

    gallery = np.stack(g_rows).astype(np.float32) if g_rows else np.zeros((0, dim), np.float32)
    probes = np.stack(p_rows[:max_p]).astype(np.float32)
    nonmated = np.stack(nm_rows).astype(np.float32)

    skipped_file = os.path.join(rdir, "skipped_images.txt")
    with open(skipped_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(cache.skipped)))

    sel = {"usable_identities": usable, "dropped_identities": dropped,
           "n_gallery": len(gallery), "n_mated_probes": len(probes),
           "n_nonmated_probes": len(nonmated),
           "n_skipped_images": len(cache.skipped)}
    with open(os.path.join(rdir, "settled_split.json"), "w", encoding="utf-8") as fh:
        json.dump(sel, fh, indent=1)

    if os.path.exists(stats_path):
        st = json.load(open(stats_path, encoding="utf-8"))
        res.add("embed", len(gallery), "embedder", "", "images_embedded", st["images"], "count")
        res.add("embed", len(gallery), "embedder", "", "embed_total_s", st["seconds"], "s")
        res.add("embed", len(gallery), "embedder", "", "embed_ms_per_img", st["ms_per_img"], "ms")
        res.add("embed", len(gallery), "embedder", "", "embed_img_per_s", st["img_per_s"], "img/s")
    res.add("embed", len(gallery), "embedder", "", "no_face_skipped", len(cache.skipped), "count")
    res.add("embed", len(gallery), "embedder", "", "identities_dropped", dropped, "count")

    log.info("settled: gallery=%d mated=%d nonmated=%d dropped=%d skipped=%d",
             len(gallery), len(probes), len(nonmated), dropped, len(cache.skipped))
    return gallery, probes, nonmated, usable


# ------------------------------------------------------------- accuracy stage

def run_accuracy(cfg, gallery, probes, nonmated, res: Results, rdir: str):
    import faiss
    curves_dir = os.path.join(rdir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    max_rank = cfg["measure"]["cmc_max_rank"]

    for N in cfg["sweep"]["accuracy_sizes"]:
        N = min(N, len(gallery))
        sub = np.ascontiguousarray(gallery[:N])
        index = faiss.IndexFlatIP(sub.shape[1])
        index.add(sub)

        p = np.ascontiguousarray(probes[:min(N, len(probes))])
        true_rows = np.arange(len(p))
        D, I = index.search(p, max_rank)
        ranks = metrics.ranks_of_truth(I, true_rows)
        cmc = metrics.cmc_curve(ranks, max_rank)

        mated_top1 = D[:, 0].astype(np.float64)
        mated_correct = I[:, 0] == true_rows
        Dn, _ = index.search(np.ascontiguousarray(nonmated), 1)
        nm_top1 = Dn[:, 0].astype(np.float64)

        taus, fpir, fnir = metrics.openset_det(mated_top1, mated_correct, nm_top1)
        fnir_t, tau_t = metrics.fnir_at_fpir(taus, fpir, fnir,
                                             cfg["measure"]["target_fpir"])

        np.savez(os.path.join(curves_dir, f"acc_N{N}.npz"),
                 cmc=cmc, taus=taus, fpir=fpir, fnir=fnir,
                 mated_top1=mated_top1, nonmated_top1=nm_top1,
                 mated_correct=mated_correct)

        res.add("accuracy", N, "flat", "", "rank1", float(cmc[0]), "frac")
        res.add("accuracy", N, "flat", "", "rank5", float(cmc[4]), "frac")
        res.add("accuracy", N, "flat", "", "rank10", float(cmc[9]), "frac")
        res.add("accuracy", N, "flat", "",
                f"fnir_at_fpir{cfg['measure']['target_fpir']}", fnir_t, "frac")
        res.add("accuracy", N, "flat", "", "fnir_threshold", tau_t, "cosine")
        res.add("accuracy", N, "flat", "", "n_mated_probes", len(p), "count")
        log.info("accuracy N=%d rank1=%.4f rank5=%.4f FNIR@FPIR=%.4f",
                 N, cmc[0], cmc[4], fnir_t)
        del index
        gc.collect()


# ---------------------------------------------------------------- scale stage

def run_scale(cfg, gallery, probes, res: Results, rdir: str):
    import faiss
    m = cfg["measure"]
    faiss.omp_set_num_threads(m["faiss_threads"])
    cache_dir = os.path.join(rdir, "cache")
    tmp_index = os.path.join(cache_dir, "tmp_index.bin")
    k = m["k"]
    dim = gallery.shape[1]

    sizes = [s for s in cfg["sweep"]["latency_sizes"]]
    max_pad = max(0, max(sizes) - len(gallery))
    pad = None
    if max_pad > 0:
        pad = faiss_indexes.build_pad_memmap(
            os.path.join(cache_dir, f"pad_{max_pad}x{dim}.npy"),
            max_pad, dim, cfg["seed"])

    q = np.ascontiguousarray(probes)

    def measure_params(index, kind, N, params_str, I_exact):
        lat = bench.single_query_latency(index, q, k, m["warmup_queries"],
                                         m["max_single_queries"],
                                         m["single_query_budget_s"])
        for name, unit in [("mean_ms", "ms"), ("p50_ms", "ms"), ("p95_ms", "ms"),
                           ("p99_ms", "ms"), ("qps_single", "qps"),
                           ("n_queries", "count")]:
            res.add("scale", N, kind, params_str, name, lat[name], unit)
        for b in m["batch_sizes"]:
            qps = bench.batched_qps(index, q, k, b)
            res.add("scale", N, kind, params_str, f"qps_batch{b}", qps, "qps")
        _, I_ann = index.search(q, k)
        if I_exact is not None:
            rec = metrics.ann_recall(I_ann, I_exact)
            res.add("scale", N, kind, params_str, "recall@1", rec["recall@1"], "frac")
            res.add("scale", N, kind, params_str, "recall@10", rec["recall@10"], "frac")
        log.info("  %s[%s] N=%d p50=%.3fms p95=%.3fms qps1=%.0f",
                 kind, params_str, N, lat["p50_ms"], lat["p95_ms"], lat["qps_single"])
        return I_ann

    for N in sizes:
        if N < len(gallery):
            provider = faiss_indexes.GalleryVectors(gallery[:N], None, N)
        else:
            provider = faiss_indexes.GalleryVectors(gallery, pad, N)
        log.info("scale sweep N=%d (real=%d pad=%d)",
                 N, min(N, len(gallery)), max(0, N - len(gallery)))
        I_exact = None

        for kind in ["flat", "ivf", "hnsw", "cagra"]:
            icfg = cfg["indexes"].get(kind, {})
            if not icfg.get("enabled", False):
                continue
            if N > icfg.get("max_n", float("inf")):
                log.info("  skipping %s at N=%d (max_n=%d)", kind, N, icfg["max_n"])
                continue
            if kind == "cagra" and not faiss_indexes.cagra_available():
                log.info("  skipping cagra (faiss build has no GpuIndexCagra)")
                continue
            index, params = faiss_indexes.make_index(kind, dim, N, icfg)
            if icfg.get("gpu", False) and kind in ("flat", "ivf"):
                index, on_gpu = faiss_indexes.to_gpu(index)
                if on_gpu:
                    params["gpu"] = 1
            with bench.RSSSampler() as rss:
                _, build_s = bench.timed(
                    faiss_indexes.train_and_add, index, provider, cfg["seed"],
                    kind, params.get("nlist"))
            base = ",".join(f"{a}={b}" for a, b in params.items())
            res.add("scale", N, kind, base, "build_s", build_s, "s")
            res.add("scale", N, kind, base, "rss_peak_build_mb",
                    rss.peak / 2**20, "MB")
            res.add("scale", N, kind, base, "index_disk_mb",
                    faiss_indexes.index_disk_bytes(index, tmp_index) / 2**20, "MB")
            log.info("  built %s N=%d in %.1fs (peak RSS %.0f MB)",
                     kind, N, build_s, rss.peak / 2**20)

            with bench.RSSSampler() as rss_s:
                if kind == "flat":
                    I_exact = measure_params(index, kind, N, base, None)[:, :k]
                elif kind == "ivf":
                    nlist = params["nlist"]
                    done = set()
                    for nprobe in icfg["nprobe_sweep"]:
                        nprobe = min(nprobe, nlist)
                        if nprobe in done:
                            continue
                        done.add(nprobe)
                        faiss_indexes.set_nprobe(index, nprobe)
                        ps = base + f",nprobe={nprobe}"
                        measure_params(index, kind, N, ps, I_exact)
                elif kind == "hnsw":
                    for ef in icfg["ef_search_sweep"]:
                        index.hnsw.efSearch = ef
                        ps = base + f",efSearch={ef}"
                        measure_params(index, kind, N, ps, I_exact)
                elif kind == "cagra":
                    measure_params(index, kind, N, base, I_exact)
            res.add("scale", N, kind, base, "rss_peak_search_mb",
                    rss_s.peak / 2**20, "MB")
            del index
            gc.collect()


# ------------------------------------------------------------------ e2e stage

def run_e2e(cfg, gallery, split, cache, res: Results):
    """Full-path probe latency: imread + detect/align + embed + search."""
    import faiss
    from .embedder import FaceEmbedder

    e = cfg["embedding"]
    embedder = FaceEmbedder(e["model"], e["det_size"], e["batch_size"],
                            e.get("providers"))
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(np.ascontiguousarray(gallery))

    paths = []
    for ident in split.enroll_order:
        cands = [c for c in split.candidates[ident] if c in cache.embs]
        if len(cands) >= 2:
            paths.append(cands[1])
        if len(paths) >= cfg["measure"]["e2e_probe_images"]:
            break

    totals, stage_sums = [], {"read": 0.0, "detect_align": 0.0, "embed": 0.0,
                              "search": 0.0}
    for p in paths:
        emb, t = embedder.embed_single_timed(p)
        if emb is None:
            continue
        t0 = time.perf_counter()
        index.search(emb[None, :], cfg["measure"]["k"])
        t["search"] = time.perf_counter() - t0
        totals.append(sum(t.values()))
        for s in stage_sums:
            stage_sums[s] += t.get(s, 0.0)

    if totals:
        ms = np.array(totals) * 1000
        N = index.ntotal
        res.add("e2e", N, "flat", "", "e2e_mean_ms", float(ms.mean()), "ms")
        res.add("e2e", N, "flat", "", "e2e_p50_ms", float(np.percentile(ms, 50)), "ms")
        res.add("e2e", N, "flat", "", "e2e_p95_ms", float(np.percentile(ms, 95)), "ms")
        for s, v in stage_sums.items():
            res.add("e2e", N, "flat", "", f"e2e_{s}_mean_ms",
                    1000 * v / len(totals), "ms")
        log.info("e2e over %d probes: mean=%.1fms p95=%.1fms",
                 len(totals), ms.mean(), np.percentile(ms, 95))


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="1:N FR scalability benchmark")
    ap.add_argument("--config", default="scalebench/config.yaml")
    ap.add_argument("--stages", default="all")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")

    cfg = load_config(args.config, args.quick)
    stages = STAGE_ORDER if args.stages == "all" else args.stages.split(",")

    rdir = cfg["paths"]["results_dir"]
    os.makedirs(os.path.join(rdir, "cache"), exist_ok=True)

    np.random.seed(cfg["seed"])
    env = collect_env(cfg)
    with open(os.path.join(rdir, "env.json"), "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=1)
    log.info("env: %s | %d threads | %.1f GB RAM", env["processor"],
             env["cpu_threads"], env["ram_total_gb"])

    ds = cfg["dataset"]
    id_map = dataset.LOADERS[ds["name"]](ds["images_dir"], ds["identity_file"])
    split = dataset.build_split(id_map, cfg["seed"], cfg["split"]["n_nonmated"],
                                cfg["split"]["max_gallery"])
    split.save(os.path.join(rdir, "split.json"))
    log.info("split: %s", split.stats)

    from .embedder import EmbeddingCache
    cache = EmbeddingCache(os.path.join(rdir, "cache", "embeddings.npz"))
    res = Results()

    gallery = probes = nonmated = None
    if {"embed", "accuracy", "scale", "e2e"} & set(stages):
        gallery, probes, nonmated, _ = settle_embeddings(
            cfg, split, id_map, cache, res, rdir)

    if "accuracy" in stages:
        run_accuracy(cfg, gallery, probes, nonmated, res, rdir)
    if "scale" in stages:
        run_scale(cfg, gallery, probes, res, rdir)
    if "e2e" in stages:
        run_e2e(cfg, gallery, split, cache, res)

    if res.rows:  # never clobber results.csv on a report-only invocation
        res.save(os.path.join(rdir, "results.csv"))

    if "report" in stages:
        from . import report
        report.build(cfg, rdir)


if __name__ == "__main__":
    main()

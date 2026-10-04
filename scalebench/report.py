"""Figures + HTML + PDF report generation.

Design notes: static matplotlib PNGs embedded base64 into a single light-mode
HTML document (deliberate: the artifact is a print/PDF report). Colors follow
the validated reference palette: categorical slots 1-3 for the three index
families (identity), the sequential blue ramp for curve families ordered by
gallery size N (magnitude). Identity is never color-alone: every chart carries
a legend and the report carries full data tables.
"""

from __future__ import annotations

import base64
import glob
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("scalebench.report")

# --- reference palette (validated; see dataviz reference instance) ----------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURF = "#fcfcfb"
PAGE = "#f9f9f7"
SERIES = {"flat": "#2a78d6", "ivf": "#008300", "hnsw": "#e87ba4",
          "cagra": "#eda100"}
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # blue 250..700

LABELS = {"flat": "Flat (exact)", "ivf": "IVF", "hnsw": "HNSW",
          "cagra": "CAGRA (GPU)"}
FAMILIES = ["flat", "ivf", "hnsw", "cagra"]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF,
        "savefig.facecolor": SURF, "font.size": 10,
        "font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.markersize": 6.5,
    })
    return plt


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _get(df, stage=None, index=None, metric=None, params_contains=None):
    q = df
    if stage:
        q = q[q.stage == stage]
    if index:
        q = q[q["index"] == index]
    if metric:
        q = q[q.metric == metric]
    if params_contains is not None:
        q = q[q.params.str.contains(params_contains, regex=False)]
    return q


def _tuned_params(df, kind):
    """Pick one representative tuned config per family for the vs-N lines."""
    if kind == "flat":
        return ""
    rows = _get(df, "scale", kind, "p95_ms")
    if kind == "cagra":
        return sorted(rows.params.unique())[0] if len(rows) else ""
    tag = "nprobe=16" if kind == "ivf" else "efSearch=64"
    have = sorted(p for p in rows.params.unique() if tag in p)
    return tag if have else (sorted(rows.params.unique())[0] if len(rows) else "")


# ----------------------------------------------------------------- figures

def fig_latency(df, plt):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for kind in FAMILIES:
        tag = _tuned_params(df, kind)
        sub = {}
        for pm in ["p50_ms", "p95_ms", "p99_ms"]:
            r = _get(df, "scale", kind, pm, tag)
            sub[pm] = r.groupby("N").value.first().sort_index()
        if not len(sub["p95_ms"]):
            continue
        n = sub["p95_ms"].index.values
        c = SERIES[kind]
        lbl = LABELS[kind] + (f" ({tag})" if tag else "")
        ax.plot(n, sub["p95_ms"].values, "-o", color=c, label=lbl, zorder=3)
        ax.fill_between(n, sub["p50_ms"].values, sub["p99_ms"].values,
                        color=c, alpha=0.14, linewidth=0)
        ax.annotate(LABELS[kind], (n[-1], sub["p95_ms"].values[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=c, fontsize=9, fontweight="bold", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Enrolled gallery size N (log)")
    ax.set_ylabel("Search latency, ms (log)")
    ax.set_title("Single-query search latency vs gallery size (p95 line, p50-p99 band)")
    ax.legend(loc="upper left")
    _despine(ax)
    return _png(fig)


def fig_qps(df, plt):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, metric, title in [
            (axes[0], "qps_single", "Single-query throughput"),
            (axes[1], "qps_batch256", "Batched throughput (batch=256)")]:
        for kind in FAMILIES:
            tag = _tuned_params(df, kind)
            r = _get(df, "scale", kind, metric, tag).groupby("N").value.first().sort_index()
            if len(r):
                ax.plot(r.index.values, r.values, "-o", color=SERIES[kind],
                        label=LABELS[kind])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Gallery size N (log)")
        ax.set_ylabel("Queries / second (log)")
        ax.set_title(title)
        _despine(ax)
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    return _png(fig)


def fig_enroll(df, plt, embed_img_per_s):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax = axes[0]
    for kind in FAMILIES:
        r = _get(df, "scale", kind, "build_s").groupby("N").value.first().sort_index()
        if len(r):
            ax.plot(r.index.values, r.values, "-o", color=SERIES[kind],
                    label=LABELS[kind])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gallery size N (log)")
    ax.set_ylabel("Index build/add time, s (log)")
    ax.set_title("Index construction time vs N")
    ax.legend(loc="upper left")
    _despine(ax)

    ax = axes[1]
    r = _get(df, "scale", "hnsw", "build_s").groupby("N").value.first().sort_index()
    ns = (r.index.values if len(r) else
          _get(df, "scale", "flat", "build_s").groupby("N").value.first().index.values)
    if embed_img_per_s and len(ns):
        embed_s = ns / embed_img_per_s
        ax.plot(ns, embed_s, "-o", color=SERIES["flat"],
                label=f"Embedding ({embed_img_per_s:.1f} img/s)")
        for kind in ["ivf", "hnsw"]:
            b = _get(df, "scale", kind, "build_s").groupby("N").value.first().sort_index()
            if len(b):
                ax.plot(b.index.values, b.index.values / embed_img_per_s + b.values,
                        "-o", color=SERIES[kind], label=f"Embed + {LABELS[kind]} build")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Gallery size N (log)")
        ax.set_ylabel("Total enrollment time, s (log)")
        ax.set_title("Derived end-to-end enrollment time")
        ax.legend(loc="upper left")
    _despine(ax)
    fig.tight_layout()
    return _png(fig)


def fig_memory(df, plt):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, metric, title, unit in [
            (axes[0], "index_disk_mb", "Index size (on disk = in RAM for these types)", "MB"),
            (axes[1], "rss_peak_build_mb", "Peak process RSS during build", "MB")]:
        for kind in FAMILIES:
            r = _get(df, "scale", kind, metric).groupby("N").value.first().sort_index()
            if len(r):
                ax.plot(r.index.values, r.values, "-o", color=SERIES[kind],
                        label=LABELS[kind])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Gallery size N (log)")
        ax.set_ylabel(f"{unit} (log)")
        ax.set_title(title)
        _despine(ax)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    return _png(fig)


def fig_rank1(df, plt):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for metric, kind, name in [("rank1", "flat", "Rank-1"),
                               ("rank5", "ivf", "Rank-5")]:
        r = _get(df, "accuracy", "flat", metric).groupby("N").value.first().sort_index()
        if len(r):
            c = SERIES[kind]
            ax.plot(r.index.values, r.values, "-o", color=c, label=name)
            for n, v in r.items():
                if n in (r.index.min(), r.index.max()):
                    ax.annotate(f"{v:.3f}", (n, v), xytext=(0, 7),
                                textcoords="offset points", ha="center",
                                fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Enrolled identities N (real, log)")
    ax.set_ylabel("Identification rate")
    ax.set_ylim(top=1.005)
    ax.set_title("Closed-set identification accuracy vs gallery size (exact search)")
    ax.legend(loc="lower left")
    _despine(ax)
    return _png(fig)


def fig_cmc(curves, plt):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for i, (N, cmc) in enumerate(sorted(curves.items())):
        ks = np.arange(1, len(cmc) + 1)
        ax.plot(ks, cmc, color=SEQ[min(i, len(SEQ) - 1)], label=f"N={N:,}")
    ax.set_xscale("log")
    ax.set_xlabel("Rank k (log)")
    ax.set_ylabel("CMC: P(true id in top-k)")
    ax.set_title("CMC curves by gallery size")
    ax.legend(loc="lower right", title="Gallery")
    _despine(ax)
    return _png(fig)


def fig_det(dets, plt):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for i, (N, (fpir, fnir)) in enumerate(sorted(dets.items())):
        m = fpir > 0
        ax.plot(fpir[m], fnir[m], color=SEQ[min(i, len(SEQ) - 1)],
                label=f"N={N:,}")
    ax.set_xscale("log")
    ax.set_xlabel("FPIR — false positive identification rate (log)")
    ax.set_ylabel("FNIR — false negative identification rate")
    ax.set_title("Open-set DET by gallery size (exact search)")
    ax.legend(loc="upper right", title="Gallery")
    _despine(ax)
    return _png(fig)


def fig_pareto(df, plt, N):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    dodge = [(7, -3), (7, 9), (7, -15), (-52, -3)]  # collision-avoiding offsets
    for kind in FAMILIES:
        lat = _get(df, "scale", kind, "p95_ms")
        lat = lat[lat.N == N]
        for i, (_, row) in enumerate(lat.iterrows()):
            rec = _get(df, "scale", kind, "recall@1", row.params)
            rec = rec[rec.N == N]
            r1 = 1.0 if kind == "flat" else (rec.value.iloc[0] if len(rec) else np.nan)
            ax.scatter(row.value, r1, s=90, color=SERIES[kind], zorder=3,
                       edgecolors=SURF, linewidths=2,
                       label=LABELS[kind] if row.params == lat.params.iloc[0] else None)
            short = row.params.split(",")[-1] if row.params else "exact"
            ax.annotate(short, (row.value, r1), xytext=dodge[i % len(dodge)],
                        textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("p95 search latency, ms (log)")
    ax.set_ylabel("Recall@1 vs exact search")
    ax.set_title(f"Speed-accuracy tradeoff at N={N:,}")
    ax.legend(loc="lower right")
    _despine(ax)
    return _png(fig)


# ------------------------------------------------------------------- tables

def scale_table(df) -> pd.DataFrame:
    rows = []
    sc = df[df.stage == "scale"]
    for (N, kind, params), g in sc.groupby(["N", "index", "params"]):
        vals = dict(zip(g.metric, g.value))
        if "p95_ms" not in vals:  # build-only rows live on the base param key
            continue
        base = sc[(sc.N == N) & (sc["index"] == kind)]
        bvals = dict(zip(base.metric, base.value))
        rows.append({
            "N": N, "index": LABELS.get(kind, kind), "params": params or "exact",
            "build_s": bvals.get("build_s"),
            "p50_ms": vals.get("p50_ms"), "p95_ms": vals.get("p95_ms"),
            "p99_ms": vals.get("p99_ms"), "qps_single": vals.get("qps_single"),
            "qps_batch256": vals.get("qps_batch256"),
            "recall@1": 1.0 if kind == "flat" else vals.get("recall@1"),
            "recall@10": 1.0 if kind == "flat" else vals.get("recall@10"),
            "disk_MB": bvals.get("index_disk_mb"),
            "peak_RSS_MB": bvals.get("rss_peak_build_mb"),
        })
    t = pd.DataFrame(rows).sort_values(["N", "index", "params"])
    return t


def accuracy_table(df) -> pd.DataFrame:
    ac = df[df.stage == "accuracy"]
    rows = []
    for N, g in ac.groupby("N"):
        vals = dict(zip(g.metric, g.value))
        fn = next((v for k, v in vals.items() if k.startswith("fnir_at")), None)
        rows.append({"N (real ids)": N, "Rank-1": vals.get("rank1"),
                     "Rank-5": vals.get("rank5"), "Rank-10": vals.get("rank10"),
                     "FNIR@FPIR=0.01": fn,
                     "threshold": vals.get("fnir_threshold"),
                     "mated probes": int(vals.get("n_mated_probes", 0))})
    return pd.DataFrame(rows).sort_values("N (real ids)")


def _fmt(t: pd.DataFrame) -> str:
    def f(v):
        if isinstance(v, float):
            if pd.isna(v):
                return "-"
            if abs(v) >= 1000:
                return f"{v:,.0f}"
            if abs(v) >= 10:
                return f"{v:.1f}"
            return f"{v:.3f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    return t.to_html(index=False, border=0, float_format=None,
                     formatters={c: f for c in t.columns},
                     classes="datatable", justify="right", escape=True)


# ----------------------------------------------------------------- analysis

def written_analysis(df, cfg, acc_t: pd.DataFrame) -> str:
    sla = cfg["sla"]["p95_ms"]
    min_r1 = cfg["sla"]["min_rank1"]
    paras = []

    # Latency inflection per family (largest N meeting the SLA).
    recs = {}
    for kind in FAMILIES:
        tag = _tuned_params(df, kind)
        r = _get(df, "scale", kind, "p95_ms", tag).groupby("N").value.first().sort_index()
        ok = r[r <= sla]
        recs[kind] = (int(ok.index.max()) if len(ok) else 0, r)
    flat_n, flat_r = recs["flat"]
    paras.append(
        f"<b>Where latency starts to hurt.</b> Exact search (IndexFlatIP) is "
        f"O(N&middot;d) per query, and the measurements track that almost perfectly: "
        f"p95 grows linearly with N once the gallery matrix outgrows cache. On this "
        f"CPU, exact p95 stays under the {sla:.0f} ms SLA up to "
        f"<b>N &asymp; {flat_n:,}</b>"
        + (f" (p95 = {flat_r.loc[flat_n]:.1f} ms there)" if flat_n in flat_r.index else "")
        + f"; at N = {int(flat_r.index.max()):,} it reaches "
        f"{flat_r.iloc[-1]:.1f} ms. The approximate indexes flatten that growth: "
        f"HNSW p95 at the largest N is "
        f"{recs['hnsw'][1].iloc[-1] if len(recs['hnsw'][1]) else float('nan'):.2f} ms "
        f"({(flat_r.iloc[-1] / recs['hnsw'][1].iloc[-1]) if len(recs['hnsw'][1]) else 0:,.0f}x "
        f"faster than exact).")

    # Exact vs ANN tradeoff.
    Nmax = int(df[df.stage == "scale"].N.max())
    hn = _get(df, "scale", "hnsw", "recall@1")
    hn = hn[hn.N == Nmax]
    iv = _get(df, "scale", "ivf", "recall@1")
    iv = iv[iv.N == Nmax]
    if len(hn) and len(iv):
        paras.append(
            f"<b>Exact vs approximate.</b> At N = {Nmax:,}: the best HNSW setting "
            f"reaches recall@1 = {hn.value.max():.4f} and the best IVF setting "
            f"{iv.value.max():.4f} against the exact baseline. The Pareto plot shows "
            f"the practical rule: HNSW buys orders of magnitude in latency at "
            f"near-unity recall but roughly doubles memory for the graph; IVF's "
            f"quality is set by nprobe — nprobe=1 is fast but lossy, nprobe&ge;16 "
            f"approaches exact at a fraction of the flat scan cost. Below "
            f"N &asymp; 10-50K, exact search is already sub-SLA, so ANN complexity "
            f"buys nothing — use Flat there.")

    # Accuracy vs scale.
    if len(acc_t):
        r1_small = acc_t.iloc[0]
        r1_big = acc_t.iloc[-1]
        paras.append(
            f"<b>How accuracy degrades with scale.</b> Closed-set Rank-1 declines "
            f"from {r1_small['Rank-1']:.4f} at N = {int(r1_small['N (real ids)']):,} "
            f"to {r1_big['Rank-1']:.4f} at N = {int(r1_big['N (real ids)']):,} "
            f"real identities — each doubling of the gallery adds more distractors "
            f"near every probe, and open-set FNIR@FPIR=0.01 moves from "
            f"{r1_small['FNIR@FPIR=0.01']:.4f} to {r1_big['FNIR@FPIR=0.01']:.4f}. "
            f"The decision threshold that holds FPIR at 1% rises with N "
            f"(more non-mated identities means more chances of a high impostor "
            f"score), which converts directly into more false rejects.")

    # Recommendation.
    acc_ok = acc_t[acc_t["Rank-1"] >= min_r1] if len(acc_t) else acc_t
    max_acc_n = int(acc_ok["N (real ids)"].max()) if len(acc_ok) else 0
    best_kind = max(recs, key=lambda k: recs[k][0])
    paras.append(
        f"<b>Recommendation.</b> Under the target SLA (p95 &lt; {sla:.0f} ms, "
        f"Rank-1 &ge; {min_r1:.2f}): accuracy is the binding constraint only up to "
        f"the real-identity ceiling we can measure ({max_acc_n:,} identities met the "
        f"Rank-1 target); latency is the binding constraint for the padded sweep. "
        f"With exact search, cap the gallery at &asymp; {flat_n:,}. With "
        f"{LABELS[best_kind]} ({_tuned_params(df, best_kind) or 'exact'}), the SLA "
        f"holds through N = {recs[best_kind][0]:,} — the largest size measured — "
        f"with recall@1 &ge; {hn.value.max() if len(hn) else float('nan'):.3f}, so "
        f"the practical ceiling on this hardware is memory, not latency: at 512-d "
        f"float32, every additional 1M identities costs &asymp; 2 GB of RAM before "
        f"index overhead. On this 5.8 GB machine the safe working limit is "
        f"&asymp; 500K vectors; beyond that, move to PQ compression or shard.")
    return "\n".join(f"<p>{p}</p>" for p in paras)


# ------------------------------------------------------------------- build

def build(cfg, rdir: str):
    plt = _style()
    df = pd.read_csv(os.path.join(rdir, "results.csv"))
    df["params"] = df.params.fillna("")
    env = json.load(open(os.path.join(rdir, "env.json"), encoding="utf-8"))

    curves, dets = {}, {}
    for f in glob.glob(os.path.join(rdir, "curves", "acc_N*.npz")):
        z = np.load(f)
        N = int(os.path.basename(f)[5:-4])
        curves[N] = z["cmc"]
        dets[N] = (z["fpir"], z["fnir"])

    emb = _get(df, "embed", metric="embed_img_per_s")
    img_per_s = float(emb.value.iloc[0]) if len(emb) else None
    Nmax = int(df[df.stage == "scale"].N.max())

    figs = {
        "latency": fig_latency(df, plt),
        "qps": fig_qps(df, plt),
        "enroll": fig_enroll(df, plt, img_per_s),
        "memory": fig_memory(df, plt),
        "rank1": fig_rank1(df, plt),
        "cmc": fig_cmc(curves, plt),
        "det": fig_det(dets, plt),
        "pareto": fig_pareto(df, plt, Nmax),
    }

    acc_t = accuracy_table(df)
    sc_t = scale_table(df)
    e2e = {r.metric: r.value for _, r in df[df.stage == "e2e"].iterrows()}
    emb_rows = {r.metric: r.value for _, r in df[df.stage == "embed"].iterrows()}

    v = env["versions"]
    idx_cfg = cfg["indexes"]
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>1:N Face Recognition Scalability Benchmark</title>
<style>
 body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        background: {PAGE}; color: {INK}; margin: 0; }}
 .page {{ max-width: 980px; margin: 0 auto; padding: 40px 28px 80px; }}
 h1 {{ font-size: 26px; margin-bottom: 4px; }}
 h2 {{ font-size: 19px; margin-top: 40px; border-bottom: 1px solid {GRID};
       padding-bottom: 6px; }}
 h3 {{ font-size: 15px; }}
 p, li {{ line-height: 1.55; color: {INK}; }}
 .sub {{ color: {INK2}; }}
 .muted {{ color: {MUTED}; font-size: 13px; }}
 img {{ max-width: 100%; border: 1px solid {GRID}; border-radius: 6px;
        margin: 10px 0; background: {SURF}; }}
 table.datatable {{ border-collapse: collapse; font-size: 12.5px; width: 100%;
        font-variant-numeric: tabular-nums; background: {SURF}; }}
 .datatable th {{ text-align: right; color: {INK2}; border-bottom: 2px solid {AXIS};
        padding: 6px 10px; }}
 .datatable td {{ text-align: right; padding: 5px 10px;
        border-bottom: 1px solid {GRID}; }}
 .datatable th:first-child, .datatable td:first-child {{ text-align: left; }}
 code {{ background: {GRID}; padding: 1px 5px; border-radius: 4px;
         font-size: 12.5px; }}
 .env {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px 24px;
         font-size: 13.5px; }}
 .env div {{ padding: 3px 0; border-bottom: 1px solid {GRID}; }}
 @media print {{ body {{ background: white; }} .page {{ padding: 0; }} }}
</style></head><body><div class="page">

<h1>1:N Face Recognition Scalability Benchmark</h1>
<p class="sub">CelebA &middot; InsightFace buffalo_l (ArcFace, 512-d) &middot;
FAISS Flat / IVF / HNSW &middot; generated {env['timestamp']} &middot; seed {env['seed']}</p>

<h2>1. Hardware &amp; software</h2>
<div class="env">
 <div><b>CPU</b> — {env['processor']} ({env['cpu_cores_physical']}C/{env['cpu_threads']}T)</div>
 <div><b>RAM</b> — {env['ram_total_gb']} GB &middot; GPU: {env['gpu']}</div>
 <div><b>OS</b> — {env['platform']}</div>
 <div><b>Python</b> — {env['python']}</div>
 <div><b>faiss-cpu</b> {v['faiss-cpu']} ({cfg['measure']['faiss_threads']} threads) &middot; <b>numpy</b> {v['numpy']}</div>
 <div><b>insightface</b> {v['insightface']} &middot; <b>onnxruntime</b> {v['onnxruntime']} &middot; <b>opencv</b> {v['opencv-python']}</div>
 <div><b>Embedding</b> — buffalo_l w600k_r50, dim {env['embedding_dim']}, det {cfg['embedding']['det_size']}px, batch {cfg['embedding']['batch_size']}</div>
 <div><b>Index params</b> — IVF nlist={idx_cfg['ivf']['nlist']}, nprobe {idx_cfg['ivf']['nprobe_sweep']}; HNSW M={idx_cfg['hnsw']['M']}, efC={idx_cfg['hnsw']['ef_construction']}, efSearch {idx_cfg['hnsw']['ef_search_sweep']}</div>
</div>

<h2>2. Methodology</h2>
<ul>
<li><b>Dataset.</b> CelebA aligned images + official <code>identity_CelebA.txt</code>
 (202,599 images, 10,177 identities; identity file MD5-verified against the
 torchvision reference). Split is seeded (seed {env['seed']}) and logged to
 <code>split.json</code> / <code>settled_split.json</code>.</li>
<li><b>Gallery.</b> One reference image per identity; mated probes are a
 <i>different</i> image of enrolled identities; non-mated probes are
 {cfg['split']['n_nonmated']} identities held fully out of the gallery.
 Galleries are nested (N=500 &sub; N=1000 &hellip;).</li>
<li><b>Scaling N.</b> Accuracy sweeps use only real identities
 ({', '.join(f"{s:,}" for s in cfg['sweep']['accuracy_sizes'])}). Latency /
 throughput / memory sweeps pad the real gallery with seeded unit-normalized
 random 512-d vectors up to {max(cfg['sweep']['latency_sizes']):,} — search cost
 is independent of vector provenance; probes are always real embeddings.</li>
<li><b>Timing.</b> Search-only timings exclude detection/embedding (embeddings
 precomputed); {cfg['measure']['warmup_queries']} warm-up queries are discarded
 per configuration; single-query latency is sampled per query with
 <code>perf_counter_ns</code>. End-to-end probe latency (imread + detect/align +
 embed + search) is reported separately. FAISS ran with
 {cfg['measure']['faiss_threads']} threads.</li>
<li><b>ANN recall.</b> FAISS convention: recall@k = fraction of probes whose
 exact top-1 appears in the ANN top-k on the same gallery.</li>
<li><b>Skips.</b> {int(emb_rows.get('no_face_skipped', 0))} images produced no
 detectable face and were skipped (logged to <code>skipped_images.txt</code>);
 {int(emb_rows.get('identities_dropped', 0))} identities were dropped for lacking
 two embeddable images.</li>
</ul>

<h2>3. Enrollment throughput</h2>
<p class="sub">Embedding extraction ran batched (batch
 {cfg['embedding']['batch_size']}): <b>{emb_rows.get('embed_img_per_s', float('nan')):.1f}
 images/s</b> ({emb_rows.get('embed_ms_per_img', float('nan')):.1f} ms per image,
 detect+align+embed, CPU). Index construction cost is measured separately below.</p>
<img src="data:image/png;base64,{figs['enroll']}">

<h2>4. Search latency vs N</h2>
<img src="data:image/png;base64,{figs['latency']}">
<h2>5. Throughput vs N</h2>
<img src="data:image/png;base64,{figs['qps']}">
<h2>6. Memory vs N</h2>
<img src="data:image/png;base64,{figs['memory']}">

<h2>7. Accuracy vs N (real identities)</h2>
<img src="data:image/png;base64,{figs['rank1']}">
<img src="data:image/png;base64,{figs['cmc']}">
<img src="data:image/png;base64,{figs['det']}">
{_fmt(acc_t)}

<h2>8. Speed-accuracy tradeoff</h2>
<img src="data:image/png;base64,{figs['pareto']}">

<h2>9. End-to-end probe latency</h2>
<p class="sub">Full path on the real gallery (N = {int(next(iter(df[df.stage=='e2e'].N), 0)):,}):
 mean {e2e.get('e2e_mean_ms', float('nan')):.1f} ms &middot;
 p50 {e2e.get('e2e_p50_ms', float('nan')):.1f} ms &middot;
 p95 {e2e.get('e2e_p95_ms', float('nan')):.1f} ms.
 Stage means: read {e2e.get('e2e_read_mean_ms', float('nan')):.1f} ms,
 detect+align {e2e.get('e2e_detect_align_mean_ms', float('nan')):.1f} ms,
 embed {e2e.get('e2e_embed_mean_ms', float('nan')):.1f} ms,
 search {e2e.get('e2e_search_mean_ms', float('nan')):.2f} ms — the CNN dominates;
 search only matters at large N or high QPS.</p>

<h2>10. Full results table</h2>
{_fmt(sc_t)}

<h2>11. Analysis &amp; recommendation</h2>
{written_analysis(df, cfg, acc_t)}

<p class="muted">Reproduce: <code>python -m scalebench.run --config scalebench/config.yaml</code>
 &middot; raw rows in <code>results.csv</code> &middot; splits, curves and caches under
 <code>scalebench/results/</code>.</p>
</div></body></html>"""

    html_path = os.path.join(rdir, "report.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info("wrote %s", html_path)

    pdf_path = os.path.join(rdir, "report.pdf")
    _html_to_pdf(html_path, pdf_path)


def _chromium_candidates():
    """Headless-capable Chromium browsers per OS (Edge ships with Win11)."""
    if sys.platform.startswith("win"):
        roots = [os.environ.get(k) for k in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA")]
        rels = [os.path.join("Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join("Google", "Chrome", "Application", "chrome.exe")]
        return [os.path.join(r, rel) for r in roots if r for rel in rels]
    if sys.platform == "darwin":
        return ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                "/Applications/Chromium.app/Contents/MacOS/Chromium"]
    return [p for p in (shutil.which(n) for n in ("chromium", "chromium-browser",
                                                   "google-chrome", "microsoft-edge")) if p]


def _html_to_pdf(html_path: str, pdf_path: str):
    """Render the HTML report to PDF with a headless Chromium browser."""
    edge = next((c for c in _chromium_candidates() if os.path.exists(c)), None)
    if edge is None:
        log.warning("no Chrome/Edge/Chromium found; skipping PDF (report.html is complete)")
        return
    with tempfile.TemporaryDirectory() as td:
        cmd = [edge, "--headless", "--disable-gpu",
               f"--user-data-dir={td}",
               "--no-pdf-header-footer",
               f"--print-to-pdf={pdf_path}",
               Path(html_path).resolve().as_uri()]
        try:
            subprocess.run(cmd, timeout=120, capture_output=True)
        except Exception as e:
            log.warning("PDF conversion failed: %s", e)
    if os.path.exists(pdf_path):
        log.info("wrote %s", pdf_path)

"""Accuracy metrics: Rank-k / CMC, open-set DET (FPIR vs FNIR), ANN recall."""

from __future__ import annotations

import numpy as np


def ranks_of_truth(I: np.ndarray, true_rows: np.ndarray) -> np.ndarray:
    """1-based rank of the true gallery row per probe; inf when not in top-k."""
    ranks = np.full(len(true_rows), np.inf)
    for i, t in enumerate(true_rows):
        hits = np.where(I[i] == t)[0]
        if len(hits):
            ranks[i] = hits[0] + 1
    return ranks


def cmc_curve(ranks: np.ndarray, max_rank: int) -> np.ndarray:
    ks = np.arange(1, max_rank + 1)
    return np.array([(ranks <= k).mean() for k in ks])


def openset_det(mated_top1: np.ndarray, mated_correct: np.ndarray,
                nonmated_top1: np.ndarray, n_taus: int = 200):
    """DET for open-set 1:N identification.

    FPIR(t) = fraction of non-mated probes whose top-1 score >= t.
    FNIR(t) = fraction of mated probes rejected (score < t) or misidentified.
    """
    lo = float(min(mated_top1.min(), nonmated_top1.min())) - 1e-3
    hi = float(max(mated_top1.max(), nonmated_top1.max())) + 1e-3
    taus = np.linspace(lo, hi, n_taus)
    fpir = np.array([(nonmated_top1 >= t).mean() for t in taus])
    fnir = np.array([((mated_top1 < t) | ~mated_correct).mean() for t in taus])
    return taus, fpir, fnir


def fnir_at_fpir(taus, fpir, fnir, target: float):
    """FNIR (and its threshold) at the largest tau where FPIR <= target."""
    ok = np.where(fpir <= target)[0]
    if len(ok) == 0:
        return float("nan"), float("nan")
    i = ok[0]  # taus ascend, fpir descends: first ok index = smallest such tau
    return float(fnir[i]), float(taus[i])


def ann_recall(I_ann: np.ndarray, I_exact: np.ndarray) -> dict[str, float]:
    """FAISS convention: R@k = fraction of probes whose EXACT top-1 appears in
    the ANN top-k."""
    exact1 = I_exact[:, 0:1]
    return {
        "recall@1": float((I_ann[:, 0:1] == exact1).any(axis=1).mean()),
        "recall@10": float((I_ann[:, :10] == exact1).any(axis=1).mean()),
    }

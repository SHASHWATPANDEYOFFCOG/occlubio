"""Measurement harness: timing, throughput, and peak-RSS sampling."""

from __future__ import annotations

import threading
import time

import numpy as np
import psutil


class RSSSampler:
    """Samples process RSS on a background thread; ``peak`` in bytes."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.peak = self._proc.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)


def single_query_latency(index, probes: np.ndarray, k: int, warmup: int,
                         max_queries: int, budget_s: float) -> dict:
    """Per-query latency percentiles (ms) + single-stream QPS.

    Warm-up queries run first and are discarded. Queries cycle through the
    probe set one at a time (true single-query path).
    """
    n = len(probes)
    for i in range(min(warmup, n)):
        index.search(probes[i:i + 1], k)

    times_ns = []
    t_start = time.perf_counter()
    for j in range(max_queries):
        q = probes[j % n:j % n + 1]
        t0 = time.perf_counter_ns()
        index.search(q, k)
        times_ns.append(time.perf_counter_ns() - t0)
        if len(times_ns) >= 50 and time.perf_counter() - t_start > budget_s:
            break

    ms = np.array(times_ns, dtype=np.float64) / 1e6
    return {
        "n_queries": len(ms),
        "mean_ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "qps_single": float(1000.0 / ms.mean()),
    }


def batched_qps(index, probes: np.ndarray, k: int, batch: int,
                repeats: int = 3) -> float:
    """Best-of-N batched throughput (queries/second) at the given batch size."""
    best = 0.0
    for _ in range(repeats):
        done = 0
        t0 = time.perf_counter()
        for start in range(0, len(probes), batch):
            q = probes[start:start + batch]
            index.search(q, k)
            done += len(q)
        qps = done / (time.perf_counter() - t0)
        best = max(best, qps)
    return float(best)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0

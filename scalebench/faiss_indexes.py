"""FAISS index construction and the padded-gallery vector provider.

All vectors are L2-normalized, so inner product == cosine similarity:
  - IndexFlatIP      exact brute-force baseline
  - IndexIVFFlat     inverted file (tune nlist / nprobe)
  - IndexHNSWFlat    graph index (tune M / efSearch)

Galleries larger than the real identity count are padded with unit-normalized
random embeddings streamed from a seeded on-disk memmap, so every index at a
given N sees byte-identical vectors and RAM never holds a second full copy.
"""

from __future__ import annotations

import logging
import math
import os

import faiss
import numpy as np

log = logging.getLogger("scalebench.indexes")

CHUNK = 65536


def build_pad_memmap(path: str, n_rows: int, dim: int, seed: int) -> np.memmap:
    """Create (once) a seeded memmap of unit-normalized random embeddings."""
    if os.path.exists(path):
        mm = np.lib.format.open_memmap(path, mode="r")
        if mm.shape == (n_rows, dim):
            return mm
        del mm
        os.remove(path)
    log.info("generating pad memmap: %d x %d -> %s", n_rows, dim, path)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                   shape=(n_rows, dim))
    for c, start in enumerate(range(0, n_rows, CHUNK)):
        n = min(CHUNK, n_rows - start)
        g = np.random.default_rng(seed * 100_003 + c)
        x = g.standard_normal((n, dim), dtype=np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        mm[start:start + n] = x
    mm.flush()
    return np.lib.format.open_memmap(path, mode="r")


class GalleryVectors:
    """First ``n_real`` real embeddings, then pad rows, totalling N."""

    def __init__(self, real: np.ndarray, pad: np.memmap | None, n_total: int):
        assert n_total >= len(real) or pad is None
        self.real = real
        self.pad = pad
        self.n = n_total

    def chunks(self):
        yield np.ascontiguousarray(self.real[:min(len(self.real), self.n)])
        need = self.n - len(self.real)
        for start in range(0, max(0, need), CHUNK):
            n = min(CHUNK, need - start)
            yield np.ascontiguousarray(self.pad[start:start + n])

    def sample(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(self.n, size=min(n, self.n), replace=False))
        real_part = idx[idx < len(self.real)]
        pad_part = idx[idx >= len(self.real)] - len(self.real)
        parts = []
        if len(real_part):
            parts.append(self.real[real_part])
        if len(pad_part):
            parts.append(self.pad[pad_part])
        return np.ascontiguousarray(np.concatenate(parts).astype(np.float32))


def auto_nlist(n: int, cap: int = 4096) -> int:
    return int(np.clip(4 * math.sqrt(n), 64, cap))


_GPU_RES = None


def gpu_available() -> bool:
    return hasattr(faiss, "StandardGpuResources")


def cagra_available() -> bool:
    return hasattr(faiss, "GpuIndexCagra")


def gpu_resources():
    """Singleton StandardGpuResources (must outlive every GPU index)."""
    global _GPU_RES
    if _GPU_RES is None:
        _GPU_RES = faiss.StandardGpuResources()
    return _GPU_RES


def to_gpu(index):
    """Move a CPU flat/IVF index to GPU 0; no-op on CPU-only builds."""
    if not gpu_available():
        log.warning("faiss GPU support not present; staying on CPU")
        return index, False
    return faiss.index_cpu_to_gpu(gpu_resources(), 0, index), True


def set_nprobe(index, nprobe: int) -> None:
    try:
        faiss.extract_index_ivf(index).nprobe = nprobe
    except Exception:
        index.nprobe = nprobe  # GPU IVF exposes nprobe directly


def make_index(kind: str, dim: int, n_total: int, cfg: dict):
    """Returns (untrained index, params_dict)."""
    if kind == "flat":
        return faiss.IndexFlatIP(dim), {}
    if kind == "ivf":
        nlist = cfg.get("nlist", "auto")
        cap = int(cfg.get("nlist_max", 4096))
        nlist = auto_nlist(n_total, cap) if nlist == "auto" else int(nlist)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                   faiss.METRIC_INNER_PRODUCT)
        return index, {"nlist": nlist}
    if kind == "hnsw":
        m = int(cfg.get("M", 32))
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(cfg.get("ef_construction", 100))
        return index, {"M": m, "ef_construction": index.hnsw.efConstruction}
    if kind == "cagra":
        cc = faiss.GpuIndexCagraConfig()
        cc.graph_degree = int(cfg.get("graph_degree", 64))
        cc.intermediate_graph_degree = int(cfg.get("intermediate_graph_degree", 128))
        index = faiss.GpuIndexCagra(gpu_resources(), dim,
                                    faiss.METRIC_INNER_PRODUCT, cc)
        return index, {"graph_degree": cc.graph_degree}
    raise ValueError(f"unknown index kind: {kind}")


def train_and_add(index, provider: GalleryVectors, seed: int,
                  kind: str = "", nlist: int | None = None) -> None:
    if kind == "cagra":
        # CAGRA builds its graph from the full matrix in one call. Only viable
        # on unified/large-memory boxes (DGX Spark) at large N.
        xb = np.concatenate([c for c in provider.chunks() if len(c)])
        index.train(xb)
        del xb
        return
    if not index.is_trained:
        if nlist is None:
            nlist = faiss.extract_index_ivf(index).nlist
        train = provider.sample(min(provider.n, max(40 * nlist, 10_000)), seed)
        index.train(train)
        del train
    for chunk in provider.chunks():
        if len(chunk):
            index.add(chunk)


def index_disk_bytes(index, tmp_path: str) -> float:
    try:
        cpu = (faiss.index_gpu_to_cpu(index)
               if "Gpu" in type(index).__name__ else index)
        faiss.write_index(cpu, tmp_path)
        size = os.path.getsize(tmp_path)
        os.remove(tmp_path)
        return size
    except Exception as e:  # e.g. CAGRA serialization unsupported in this build
        log.warning("index serialization unsupported (%s); disk size = NaN", e)
        return float("nan")

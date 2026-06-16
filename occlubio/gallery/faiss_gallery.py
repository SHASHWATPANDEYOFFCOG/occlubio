"""FAISS-backed 1:N identity gallery (cosine similarity on L2-normalized embeddings).

Edge / small gallery: keep this flat IndexFlatIP (exact). For 1M-1B identities move to
IVF-PQ or Milvus-GPU and add an exact re-rank on the top-k (architecture §1.11).

HOOK: store *protected* templates here, not raw embeddings (architecture §1.12). The
add/search API is unchanged if you transform embeddings before insertion + query.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from occlubio.utils import ensure_dir, get_logger, l2_normalize

log = get_logger(__name__)


class FaissGallery:
    def __init__(self, dim: int = 512):
        import faiss  # lazy

        self._faiss = faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # inner product == cosine on normalized vecs
        self.labels: List[str] = []
        self.meta: List[dict] = []

    # ---- mutation ----------------------------------------------------------
    def add(self, embedding: np.ndarray, label: str, meta: Optional[dict] = None) -> None:
        emb = l2_normalize(embedding).astype(np.float32).reshape(1, -1)
        if emb.shape[1] != self.dim:
            raise ValueError(f"embedding dim {emb.shape[1]} != gallery dim {self.dim}")
        self.index.add(emb)
        self.labels.append(label)
        self.meta.append(meta or {})

    def add_many(self, embeddings: np.ndarray, labels: List[str], metas=None) -> None:
        embs = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        self.index.add(embs)
        self.labels.extend(labels)
        self.meta.extend(metas or [{} for _ in labels])

    # ---- query -------------------------------------------------------------
    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        if self.index.ntotal == 0:
            return []
        q = l2_normalize(embedding).astype(np.float32).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, idx = self.index.search(q, k)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            out.append((self.labels[i], float(s), self.meta[i]))
        return out

    def identify(self, embedding: np.ndarray, threshold: float) -> Tuple[str, float]:
        hits = self.search(embedding, top_k=1)
        if not hits:
            return "unknown", 0.0
        label, score, _ = hits[0]
        return (label, score) if score >= threshold else ("unknown", score)

    # ---- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = ensure_dir(path)
        self._faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "labels": self.labels, "meta": self.meta}, f, indent=2)
        log.info("Saved gallery (%d entries) -> %s", len(self.labels), path)

    @classmethod
    def load(cls, path: str | Path) -> "FaissGallery":
        import faiss

        path = Path(path)
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        g = cls(dim=data["dim"])
        g.index = faiss.read_index(str(path / "index.faiss"))
        g.labels = data["labels"]
        g.meta = data["meta"]
        log.info("Loaded gallery (%d entries) <- %s", len(g.labels), path)
        return g

    @classmethod
    def load_or_new(cls, path: str | Path, dim: int = 512) -> "FaissGallery":
        path = Path(path)
        if (path / "index.faiss").exists():
            return cls.load(path)
        log.info("No gallery at %s — starting empty (dim=%d)", path, dim)
        return cls(dim=dim)

    def __len__(self) -> int:
        return len(self.labels)

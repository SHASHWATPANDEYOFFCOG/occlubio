"""Dataset loading and reproducible split construction.

The ONLY dataset-specific piece is the identity-label loader. To benchmark a
different identity-labeled dataset (LFW, etc.), write a loader that returns
``{absolute_image_path: identity_label}`` and register it in ``LOADERS``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np


def load_identity_map_celeba(images_dir: str, identity_file: str) -> dict[str, str]:
    """CelebA: ``identity_CelebA.txt`` lines are '<filename> <identity_int>'."""
    id_map: dict[str, str] = {}
    with open(identity_file, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2:
                continue
            fname, ident = parts
            id_map[os.path.join(images_dir, fname)] = ident
    return id_map


LOADERS = {"celeba": load_identity_map_celeba}


@dataclass
class Split:
    """Seeded, disk-logged benchmark split.

    ``enroll_order`` lists enrollable identities (>=2 images) in seeded order;
    galleries of size N are always the first N *usable* identities, so galleries
    are nested (N=500 is a subset of N=1000, etc.) and per-identity results are
    comparable across N. Each identity carries its candidate images in
    deterministic order; the embedding stage settles which candidate actually
    embeds (detection failures fall through to the next candidate).
    """

    seed: int
    enroll_order: list[str] = field(default_factory=list)          # identity ids
    candidates: dict[str, list[str]] = field(default_factory=dict) # id -> image paths
    nonmated_pool: list[str] = field(default_factory=list)         # image paths
    stats: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=1)


def build_split(id_map: dict[str, str], seed: int, n_nonmated: int,
                max_gallery: int, max_candidates: int = 4) -> Split:
    by_id: dict[str, list[str]] = {}
    for path, ident in id_map.items():
        by_id.setdefault(ident, []).append(path)
    for imgs in by_id.values():
        imgs.sort()  # deterministic candidate order

    singles = sorted(i for i, imgs in by_id.items() if len(imgs) == 1)
    multis = sorted(i for i, imgs in by_id.items() if len(imgs) >= 2)

    rng = np.random.default_rng(seed)
    rng.shuffle(singles)
    rng.shuffle(multis)

    # Non-mated identities: prefer single-image identities (useless for
    # gallery+mated anyway); top up from the multi pool if needed.
    nonmated_ids = singles[:n_nonmated]
    if len(nonmated_ids) < n_nonmated:
        take = n_nonmated - len(nonmated_ids)
        nonmated_ids += multis[:take]
        multis = multis[take:]

    enroll_order = multis[:max_gallery + 500]  # slack for detection dropouts

    # A few candidate images per non-mated identity too (detection fallback).
    nonmated_pool = []
    for ident in nonmated_ids:
        nonmated_pool.extend(by_id[ident][:2])

    return Split(
        seed=seed,
        enroll_order=enroll_order,
        candidates={i: by_id[i][:max_candidates] for i in enroll_order},
        nonmated_pool=nonmated_pool,
        stats={
            "total_images": len(id_map),
            "total_identities": len(by_id),
            "single_image_identities": len(singles),
            "multi_image_identities": len(multis),
            "nonmated_identities": len(nonmated_ids),
            "enroll_candidates": len(enroll_order),
        },
    )

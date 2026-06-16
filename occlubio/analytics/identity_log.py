"""Video identity analytics: turn per-frame recognition results into a person-level report.

Two consolidation steps keep the report clean (no fragmented-track noise):
  - KNOWN people: tracks are grouped by identity, and time windows separated by less than
    `merge_gap_s` are merged so "appearances" = real on-screen segments, not track IDs.
  - UNKNOWN people: short blips are dropped, and remaining unknown tracks are clustered by
    face-embedding similarity, so the same un-enrolled person isn't reported 100+ times.

Produces: person id, confidence, first/last-seen timestamp, appearances, unknown-person alerts.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class TrackRecord:
    track_id: int
    first_t: float
    last_t: float
    n_frames: int = 0
    score_sum: float = 0.0
    best_score: float = 0.0
    votes: Counter = field(default_factory=Counter)
    best_emb: Optional[np.ndarray] = None
    best_emb_score: float = -1.0

    def update(self, t: float, identity: str, score: float, emb: Optional[np.ndarray]) -> None:
        self.last_t = t
        self.n_frames += 1
        self.score_sum += float(score)
        self.best_score = max(self.best_score, float(score))
        if identity and identity != "unknown":
            self.votes[identity] += 1
        if emb is not None and float(score) >= self.best_emb_score:
            self.best_emb_score = float(score)
            self.best_emb = np.asarray(emb, dtype=np.float32)

    @property
    def identity(self) -> str:
        return self.votes.most_common(1)[0][0] if self.votes else "unknown"

    @property
    def avg_score(self) -> float:
        return self.score_sum / max(self.n_frames, 1)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_t - self.first_t)


def _ts(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m:02d}:{seconds - 60 * m:05.2f}"


def _merge_windows(intervals, gap):
    merged = []
    for s, e in sorted(intervals):
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class IdentityLog:
    def __init__(self, min_track_frames: int = 3, merge_gap_s: float = 1.5,
                 unknown_min_seconds: float = 0.6, unknown_merge_sim: float = 0.45):
        self.tracks: dict[int, TrackRecord] = {}
        self.min_track_frames = min_track_frames
        self.merge_gap_s = merge_gap_s
        self.unknown_min_seconds = unknown_min_seconds
        self.unknown_merge_sim = unknown_merge_sim

    def update(self, t: float, results) -> None:
        for f in results:
            tid = getattr(f, "track_id", None)
            if tid is None:
                continue
            rec = self.tracks.get(tid)
            if rec is None:
                rec = TrackRecord(tid, first_t=t, last_t=t)
                self.tracks[tid] = rec
            rec.update(t, f.identity, f.score, getattr(f, "embedding", None))

    def _cluster_unknowns(self, recs: List[TrackRecord]) -> List[List[TrackRecord]]:
        clusters: list[dict] = []
        for r in sorted(recs, key=lambda r: -r.duration):
            emb = _unit(r.best_emb) if r.best_emb is not None else None
            placed = False
            if emb is not None:
                for c in clusters:
                    if c["rep"] is not None and float(np.dot(emb, c["rep"])) >= self.unknown_merge_sim:
                        c["members"].append(r)
                        placed = True
                        break
            if not placed:
                clusters.append({"rep": emb, "members": [r]})
        return [c["members"] for c in clusters]

    def report(self) -> dict:
        tracks = [r for r in self.tracks.values() if r.n_frames >= self.min_track_frames]
        persons: dict[str, list[TrackRecord]] = defaultdict(list)
        unknown_tracks: list[TrackRecord] = []
        for r in tracks:
            (unknown_tracks if r.identity == "unknown" else persons[r.identity]).append(r)

        people = []
        for name, recs in sorted(persons.items()):
            windows = _merge_windows([(r.first_t, r.last_t) for r in recs], self.merge_gap_s)
            people.append({
                "id": name,
                "appearances": len(windows),
                "first_seen": _ts(min(w[0] for w in windows)),
                "last_seen": _ts(max(w[1] for w in windows)),
                "total_visible_s": round(sum(b - a for a, b in windows), 2),
                "max_confidence": round(max(r.best_score for r in recs), 3),
                "avg_confidence": round(sum(r.avg_score for r in recs) / len(recs), 3),
                "appearance_windows": [[_ts(a), _ts(b)] for a, b in windows],
            })

        # drop unknown blips, then cluster the rest into distinct unknown people
        unknown_tracks = [r for r in unknown_tracks if r.duration >= self.unknown_min_seconds]
        clusters = self._cluster_unknowns(unknown_tracks)
        alerts = []
        for i, recs in enumerate(sorted(clusters, key=lambda c: min(r.first_t for r in c)), start=1):
            windows = _merge_windows([(r.first_t, r.last_t) for r in recs], self.merge_gap_s)
            alerts.append({
                "alert": f"UNKNOWN-{i}",
                "appearances": len(windows),
                "first_seen": _ts(min(w[0] for w in windows)),
                "last_seen": _ts(max(w[1] for w in windows)),
                "visible_s": round(sum(b - a for a, b in windows), 2),
            })

        return {
            "summary": {"known_people": len(people), "unknown_alerts": len(alerts)},
            "people": people,
            "unknown_alerts": alerts,
        }

    def save(self, path: str | Path) -> dict:
        rep = self.report()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        return rep

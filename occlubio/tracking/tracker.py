"""Lightweight IoU/greedy tracker (SORT-lite, no Kalman, no extra deps).

Enough to assign stable track ids so recognition runs once per track and embeddings are
fused across a track's lifetime (set-based recognition — key for occlusion robustness).

HOOK: for crowded/occluded scenes swap in ByteTrack (FoundationVision/ByteTrack) or
BoT-SORT with ReID; on Jetson use DeepStream `nvtracker` (NvDCF). Keep `update()`'s contract:
take detections, return them with `track_id` set.
"""
from __future__ import annotations

from typing import List

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union


class _Track:
    __slots__ = ("id", "bbox", "age", "hits", "time_since_update")

    def __init__(self, tid: int, bbox: np.ndarray):
        self.id = tid
        self.bbox = bbox
        self.age = 0
        self.hits = 1
        self.time_since_update = 0


class IoUTracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30, min_hits: int = 2):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[_Track] = []
        self._next_id = 0

    def update(self, detections: List) -> List:
        """Assign track_id to each FaceResult (greedy IoU matching). Mutates and returns it."""
        for t in self.tracks:
            t.time_since_update += 1
            t.age += 1

        unmatched = list(range(len(detections)))
        # greedy match: highest IoU first
        pairs = []
        for di, det in enumerate(detections):
            for ti, trk in enumerate(self.tracks):
                iou = _iou(det.bbox, trk.bbox)
                if iou >= self.iou_thresh:
                    pairs.append((iou, di, ti))
        pairs.sort(reverse=True)

        used_det, used_trk = set(), set()
        for iou, di, ti in pairs:
            if di in used_det or ti in used_trk:
                continue
            trk = self.tracks[ti]
            trk.bbox = detections[di].bbox
            trk.hits += 1
            trk.time_since_update = 0
            detections[di].track_id = trk.id
            used_det.add(di)
            used_trk.add(ti)
            if di in unmatched:
                unmatched.remove(di)

        for di in unmatched:
            trk = _Track(self._next_id, detections[di].bbox)
            self._next_id += 1
            detections[di].track_id = trk.id
            self.tracks.append(trk)

        # cull dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return detections

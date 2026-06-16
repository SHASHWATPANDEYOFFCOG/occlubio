"""Passive RGB anti-spoofing (presentation-attack detection).

Disabled by default. When `antispoof.onnx_path` is set, runs a generic ONNX classifier on
the face crop and returns a liveness score.

HOOK: the de-facto edge default is MiniFASNet from Silent-Face-Anti-Spoofing
(minivision-ai/Silent-Face-Anti-Spoofing). Its real preprocessing crops a *scaled* region
around the face (not a tight align) and uses a specific input size — match it exactly or
accuracy will be poor. For generalization, distill a domain-generalized teacher (CFPL-FAS,
CVPR'24) into this model, and add an IR/depth sensor if the hardware allows (architecture §1.6).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from occlubio.utils import get_logger

log = get_logger(__name__)


class AntiSpoof:
    def __init__(self, cfg):
        a = cfg.antispoof
        self.enabled = bool(a.enabled)
        self.min_score = float(a.min_score)
        self.input_size = tuple(a.input_size)
        self.session = None
        self.input_name = None
        if self.enabled and a.onnx_path:
            try:
                import onnxruntime as ort

                self.session = ort.InferenceSession(
                    a.onnx_path,
                    providers=list(getattr(cfg.device, "providers", ["CPUExecutionProvider"])),
                )
                self.input_name = self.session.get_inputs()[0].name
                log.info("AntiSpoof model loaded: %s", a.onnx_path)
            except Exception as e:  # noqa: BLE001
                log.warning("AntiSpoof disabled (failed to load %s): %s", a.onnx_path, e)
                self.enabled = False

    def check(self, aligned_crop: np.ndarray) -> dict:
        if not self.enabled or self.session is None:
            return {"live": True, "score": 1.0}

        x = cv2.resize(aligned_crop, self.input_size).astype(np.float32)
        x = (x - 127.5) / 128.0
        x = np.transpose(x, (2, 0, 1))[None]  # NCHW
        out = self.session.run(None, {self.input_name: x})[0].ravel()
        # Convention: assume last logit/prob == "live". Adapt to your model's head.
        live_score = float(_softmax(out)[-1]) if out.size > 1 else float(out[0])
        return {"live": live_score >= self.min_score, "score": live_score}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

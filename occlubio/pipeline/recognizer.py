from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from occlubio.utils import get_logger, l2_normalize

log = get_logger(__name__)


class CustomRecognizer:
    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None, image_size: int = 112):
        import onnxruntime as ort

        from occlubio.platform_support import onnx_providers

        self.session = ort.InferenceSession(onnx_path, providers=onnx_providers(providers))
        self.input_name = self.session.get_inputs()[0].name
        self.image_size = image_size
        log.info("CustomRecognizer loaded: %s", onnx_path)

    def _preprocess(self, aligned_crop: np.ndarray) -> np.ndarray:
        if aligned_crop.shape[:2] != (self.image_size, self.image_size):
            aligned_crop = cv2.resize(aligned_crop, (self.image_size, self.image_size))
        x = aligned_crop.astype(np.float32)
        x = (x - 127.5) / 128.0
        x = np.transpose(x, (2, 0, 1))[None]
        return x

    def embed(self, aligned_crop: np.ndarray) -> np.ndarray:
        x = self._preprocess(aligned_crop)
        out = self.session.run(None, {self.input_name: x})[0].ravel()
        return l2_normalize(out)

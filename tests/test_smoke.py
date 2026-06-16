"""Smoke tests — no network, no model downloads. Run: pytest -q

Covers the pure-python/core paths: config, alignment, occlusion augmentation, tracker, and
(if installed) the FAISS gallery and AdaFace head.
"""
import numpy as np
import pytest

from occlubio import load_config
from occlubio.config import config_to_dict
from occlubio.data import OcclusionAugmentor
from occlubio.pipeline.aligner import ARCFACE_DST, norm_crop
from occlubio.pipeline.face_analyzer import FaceResult
from occlubio.tracking import IoUTracker


def test_config_loads_with_expected_keys():
    cfg = load_config()
    assert cfg.gallery.top_k >= 1
    assert cfg.recognition.embedding_dim == 512
    assert isinstance(config_to_dict(cfg), dict)


def test_aligner_outputs_112():
    img = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    kps = ARCFACE_DST + np.array([200, 150], np.float32)  # shift template into the image
    crop = norm_crop(img, kps)
    assert crop.shape == (112, 112, 3)


def test_occlusion_augmentor_preserves_shape():
    img = np.random.randint(0, 255, (112, 112, 3), np.uint8)
    aug = OcclusionAugmentor(occlude_prob=1.0, photometric_prob=1.0, seed=0)
    out = aug(img)
    assert out.shape == (112, 112, 3) and out.dtype == np.uint8


def test_tracker_assigns_stable_ids():
    tracker = IoUTracker(iou_thresh=0.3, max_age=5, min_hits=1)

    def det(x1, y1, x2, y2):
        return FaceResult(bbox=np.array([x1, y1, x2, y2], np.float32),
                          kps=np.zeros((5, 2), np.float32), det_score=0.9)

    f1 = tracker.update([det(10, 10, 50, 50)])[0]
    f2 = tracker.update([det(12, 11, 52, 51)])[0]   # same face, moved slightly
    assert f1.track_id == f2.track_id


def test_faiss_gallery_roundtrip(tmp_path):
    faiss = pytest.importorskip("faiss")  # noqa: F841
    from occlubio.gallery import FaissGallery

    g = FaissGallery(dim=8)
    a = np.random.rand(8).astype(np.float32)
    b = np.random.rand(8).astype(np.float32)
    g.add(a, "alice")
    g.add(b, "bob")
    label, score = g.identify(a, threshold=0.5)
    assert label == "alice" and score > 0.99

    g.save(tmp_path / "gal")
    g2 = FaissGallery.load(tmp_path / "gal")
    assert len(g2) == 2 and g2.identify(b, 0.5)[0] == "bob"


def test_adaface_head_shapes():
    torch = pytest.importorskip("torch")
    from occlubio.training.losses import AdaFace

    head = AdaFace(embedding_size=16, num_classes=10)
    emb = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    norms = torch.norm(torch.randn(4, 16), dim=1)
    labels = torch.tensor([0, 1, 2, 3])
    logits = head(emb, norms, labels)
    assert logits.shape == (4, 10)

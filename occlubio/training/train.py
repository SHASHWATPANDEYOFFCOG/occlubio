"""Train an edge, occlusion-aware face embedder (AdaFace head + timm backbone).

This is a runnable *starting skeleton* — not a full WebFace12M recipe. It gives you the right
structure (embedding net -> norm split -> AdaFace -> CE) with occlusion augmentation baked in,
and exports ONNX that plugs straight into inference via `recognition.custom_onnx`.

For SOTA, swap the backbone for IResNet-100 / MobileFaceNet / EdgeFace and add teacher-student
distillation (architecture §3). Run:

    python -m occlubio.training.train --data /path/to/aligned_faces --epochs 20 --out runs/edge
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from occlubio.data import OcclusionAugmentor
from occlubio.training.dataset import AlignedFaceDataset
from occlubio.training.losses import AdaFace, ArcFace
from occlubio.utils import ensure_dir, get_logger

log = get_logger("train")


class EmbeddingNet(nn.Module):
    """timm backbone -> 512-d embedding head. Returns (normalized_embedding, feature_norm)."""

    def __init__(self, backbone: str = "mobilenetv3_small_100", embedding_dim: int = 512):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        feat = self.backbone.num_features
        self.head = nn.Sequential(nn.Linear(feat, embedding_dim), nn.BatchNorm1d(embedding_dim))

    def forward(self, x):
        feat = self.head(self.backbone(x))
        norm = torch.norm(feat, 2, 1, keepdim=True).clamp(min=1e-6)
        return feat / norm, norm.squeeze(1)


def export_onnx(model: EmbeddingNet, out_path: Path, image_size: int = 112):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)

    class _EmbedOnly(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            emb, _ = self.m(x)
            return emb

    torch.onnx.export(
        _EmbedOnly(model).cpu(), dummy, str(out_path),
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=13,
    )
    log.info("Exported ONNX -> %s (plug into recognition.custom_onnx)", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root/<identity>/<aligned_img>")
    ap.add_argument("--out", default="runs/edge_adaface")
    ap.add_argument("--backbone", default="mobilenetv3_small_100")
    ap.add_argument("--head", choices=["adaface", "arcface"], default="adaface")
    ap.add_argument("--embedding-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--occlude-prob", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = ensure_dir(args.out)
    augmentor = OcclusionAugmentor(occlude_prob=args.occlude_prob, photometric_prob=0.5)
    ds = AlignedFaceDataset(args.data, augmentor=augmentor, train=True)
    log.info("dataset: %d images, %d identities", len(ds), ds.num_classes)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)

    model = EmbeddingNet(args.backbone, args.embedding_dim).to(args.device)
    Head = AdaFace if args.head == "adaface" else ArcFace
    head = Head(args.embedding_dim, ds.num_classes).to(args.device)

    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train(); head.train()
        running, seen = 0.0, 0
        for x, y in dl:
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            emb, norm = model(x)
            logits = head(emb, norm, y)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * x.size(0); seen += x.size(0)
        sched.step()
        log.info("epoch %d/%d  loss=%.4f  lr=%.4f", epoch + 1, args.epochs, running / seen, sched.get_last_lr()[0])
        torch.save({"model": model.state_dict(), "head": head.state_dict(),
                    "classes": ds.classes, "args": vars(args)}, out / "last.pt")

    export_onnx(model, out / "model.onnx", image_size=112)
    log.info("done. set recognition.custom_onnx: %s", out / "model.onnx")


if __name__ == "__main__":
    main()

"""Margin heads for face recognition: AdaFace (default) and ArcFace.

AdaFace (Kim et al., CVPR 2022) adapts the margin by *feature norm* (an image-quality proxy),
de-emphasizing unidentifiable (occluded/blurry) samples instead of forcing them — the right
inductive bias for occlusion-robust recognition. This is a faithful re-implementation of the
official formulation (mk-minchul/AdaFace).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(x, axis=1, eps=1e-10):
    return x / torch.clamp(torch.norm(x, 2, axis, keepdim=True), min=eps)


class AdaFace(nn.Module):
    def __init__(self, embedding_size=512, num_classes=1000, m=0.4, h=0.333, s=64.0, t_alpha=0.01):
        super().__init__()
        self.m, self.h, self.s, self.t_alpha = m, h, s, t_alpha
        self.eps = 1e-3
        self.kernel = nn.Parameter(torch.empty(embedding_size, num_classes))
        nn.init.normal_(self.kernel, std=0.01)
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)

    def forward(self, embeddings, norms, labels):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embeddings, kernel_norm).clamp(-1 + self.eps, 1 - self.eps)

        safe_norms = torch.clip(norms, min=0.001, max=100.0).detach()
        with torch.no_grad():
            mean = safe_norms.mean()
            std = safe_norms.std()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = torch.clip(margin_scaler * self.h, -1.0, 1.0).view(-1, 1)  # (B,1) to broadcast

        # angular margin g_angular = -m * margin_scaler
        m_arc = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        g_angular = -self.m * margin_scaler
        m_arc = m_arc * g_angular
        theta = torch.acos(cosine)
        theta_m = torch.clip(theta + m_arc, self.eps, math.pi - self.eps)
        cosine = torch.cos(theta_m)

        # additive margin g_add = m + m * margin_scaler
        m_cos = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        g_add = self.m + self.m * margin_scaler
        cosine = cosine - m_cos * g_add

        return cosine * self.s


class ArcFace(nn.Module):
    """Classic ArcFace margin (Deng et al., CVPR 2019) — simple baseline."""

    def __init__(self, embedding_size=512, num_classes=1000, m=0.5, s=64.0):
        super().__init__()
        self.s, self.m = s, m
        self.kernel = nn.Parameter(torch.empty(embedding_size, num_classes))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings, norms, labels):  # norms unused; kept for a common API
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embeddings, kernel_norm).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        m_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), self.m)
        return torch.cos(theta + m_hot) * self.s

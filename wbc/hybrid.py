"""Improvement 3: advanced feature extraction -- CNN + Transformer multi-scale features fused with the
paper's pointwise/spatial branches through attention (CBAM), compressed to 128-D per pixel so it is a
drop-in replacement for the paper's 128-D feature set.

    ROI ─┬─ local branch  : paper 1x1 SeparableConv ‖ 2x3x3 SeparableConv (trainable)       -> 64
         ├─ CNN branch    : ConvNeXt-T stages 1-4 (frozen) -> 1x1 proj, upsample            -> 4x32
         └─ ViT branch    : DINOv2-S / Swin-T blocks (frozen, global context) -> 1x1 proj   -> 4x32
    concat(320) -> CBAM (channel + spatial attention) -> 3x3 conv -> 128-D features -> 1x1 head
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import SeparableConv2d, TimmBackbone


class CBAM(nn.Module):
    """Woo et al., ECCV 2018: channel attention (which feature sources matter) then spatial attention
    (where they matter)."""

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(c, c // r, 1), nn.ReLU(inplace=True), nn.Conv2d(c // r, c, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


class HybridAttentionFusion(nn.Module):
    trainable = True

    def __init__(self, cnn: str | None = "convnext_t", vit: str | None = "dinov2_s", use_local: bool = True,
                 attention: bool = True, proj: int = 32, out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.cfg = dict(cnn=cnn, vit=vit, use_local=use_local, attention=attention)
        c_total = 0
        self.local = None
        if use_local:
            self.local = nn.ModuleDict(dict(
                pw=nn.Sequential(SeparableConv2d(3, 32, 1), nn.LeakyReLU(0.3)),
                sp=nn.Sequential(SeparableConv2d(3, 32, 3), nn.LeakyReLU(0.3),
                                 SeparableConv2d(32, 32, 3), nn.LeakyReLU(0.3), nn.Dropout(0.2))))
            c_total += 64
        self.cnn = self.vit = None
        if cnn:
            self.cnn = TimmBackbone(cnn, pretrained=pretrained)
            chans = self.cnn.net.feature_info.channels()
            self.cnn_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(c, proj, 1), nn.GELU()) for c in chans)
            c_total += proj * len(chans)
        if vit:
            self.vit = TimmBackbone(vit, pretrained=pretrained)
            dims = [self.vit.out_dim // 4] * 4 if self.vit.is_vit else self.vit.net.feature_info.channels()
            self.vit_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(c, proj, 1), nn.GELU()) for c in dims)
            c_total += proj * len(dims)
        self.attn = CBAM(c_total) if attention else nn.Identity()
        self.fuse = nn.Sequential(nn.Conv2d(c_total, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.GELU())
        self.head = nn.Conv2d(out_dim, 1, 1)
        self.out_dim = out_dim

    def _branch(self, backbone, projs, x, S):
        with torch.no_grad():
            pyr = backbone.pyramid(x)
        return [F.interpolate(p(f), size=(S, S), mode="bilinear", align_corners=False) for p, f in zip(projs, pyr)]

    def features(self, x):
        S = x.shape[-1]
        parts = []
        if self.local is not None:
            parts += [self.local["pw"](x), self.local["sp"](x)]
        if self.cnn is not None:
            parts += self._branch(self.cnn, self.cnn_proj, x, S)
        if self.vit is not None:
            parts += self._branch(self.vit, self.vit_proj, x, S)
        return self.fuse(self.attn(torch.cat(parts, 1)))

    def forward(self, x):  # logits, used during training and as the end-to-end "FCL" head
        return self.head(self.features(x))

    def train(self, mode: bool = True):
        super().train(mode)
        for b in (self.cnn, self.vit):  # frozen backbones always in eval mode (BN / dropout)
            if b is not None:
                b.eval()
        return self


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# This is our upgraded feature extractor ("Improvement 3"). Instead of the paper's single tiny
# network, it listens to three "experts" at once and lets an attention module decide whom to trust:
#  1. the paper's own small local branches (exact pixel colour and fine edges),
#  2. ConvNeXt, a modern convolutional network that is great at textures at several zoom levels,
#  3. a Vision Transformer (DINOv2 or Swin) that sees the whole cell at once (global shape/context).
# The big pretrained experts are kept frozen; we only train small "adapter" layers that shrink each
# expert's output, the CBAM attention block (which learns which channels and which locations are
# important), and a final layer that produces 128 numbers per pixel - the same size as the paper's
# features, so the same classifiers (FCL, SVM, RF, XGBoost) can be compared on equal terms. The
# switches (cnn / vit / use_local / attention) let us switch parts off for ablation experiments.

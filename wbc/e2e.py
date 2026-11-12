"""Recent end-to-end segmentation models used as comparison baselines (trained on the same ROIs).

* UNet          -- Ronneberger et al. 2015 (from scratch, 4 levels).
* SegFormer     -- Xie et al. NeurIPS 2021, MiT-B0 (ADE20K-pretrained, new 1-class head; HF transformers).
* SAM zero-shot -- the whole-cell mask returned by SamCMYKLocalizer (no training at all).
* Hybrid head   -- our HybridAttentionFusion with its own 1x1 head (end-to-end variant of Improvement 3).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import IMAGENET_MEAN, IMAGENET_STD


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([_block(3, c[0])] + [_block(c[i], c[i + 1]) for i in range(4)])
        self.up = nn.ModuleList(nn.ConvTranspose2d(c[i + 1], c[i], 2, 2) for i in reversed(range(4)))
        self.dec = nn.ModuleList(_block(c[i] * 2, c[i]) for i in reversed(range(4)))
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.out(x)


class SegFormer(nn.Module):
    def __init__(self, variant: str = "b0"):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        # ADE20K-pretrained checkpoint (ships safetensors); decode head re-initialised for 1 class
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/segformer-{variant}-finetuned-ade-512-512", num_labels=1, id2label={0: "wbc"},
            label2id={"wbc": 0}, ignore_mismatched_sizes=True, use_safetensors=True)

    def forward(self, x):
        S = x.shape[-1]
        x = (x - IMAGENET_MEAN.to(x)) / IMAGENET_STD.to(x)
        return F.interpolate(self.net(pixel_values=x).logits, size=(S, S), mode="bilinear", align_corners=False)


def build_e2e(key: str) -> nn.Module:
    return {"unet": lambda: UNet(), "segformer_b0": lambda: SegFormer("b0")}[key]()


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# The paper splits the job into two steps: make features, then let a separate classifier label each
# pixel. Modern approaches instead train one network that goes straight from picture to mask. This
# file holds those modern rivals so we can compare against them: U-Net (the classic medical-image
# segmenter with its encoder-decoder "U" shape) and SegFormer (a recent, lightweight Transformer
# segmenter that starts from weights pretrained on the ADE20K scene dataset). SAM's zero-shot mask (no training at all)
# and our own hybrid model's built-in head are also evaluated as end-to-end models elsewhere.

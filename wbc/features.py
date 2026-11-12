"""Phase III feature extractors. Every extractor maps a batch of ROIs (B,3,S,S float in [0,1]) to a
per-pixel feature map (B,D,S,S) so that every pixel becomes one sample for the classifiers.

* PaperCNNFusion  -- the paper's network (Fig. 5): 1x1 SeparableConv branch || 2x(3x3 SeparableConv)
                     branch, LeakyReLU(0.3), AvgPool(1), Dropout(0.2), concat -> 128 features.
* Handcrafted     -- benchmark feature banks M1..M5 (Gabor, +LBP, +edges, +k-means, +Gaussian).
* TimmBackbone    -- recent pretrained models (ResNet-50, ConvNeXt, Swin, DINOv2-ViT, EfficientNet...)
                     with multi-scale feature maps upsampled to pixel resolution.
* (Improvement 3, the proposed hybrid attention-fusion extractor, lives in hybrid.py.)
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SeparableConv2d(nn.Module):
    """Keras SeparableConv2D: depthwise (per-channel) conv followed by 1x1 pointwise conv."""

    def __init__(self, cin, cout, k):
        super().__init__()
        self.depthwise = nn.Conv2d(cin, cin, k, padding=k // 2, groups=cin)  # padding="same"
        self.pointwise = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class PaperCNNFusion(nn.Module):
    """Exact Fig. 5 topology. The paper uses the network purely as a feature extractor (no training
    is described), so by default weights are the seeded random initialisation."""
    trainable = False

    def __init__(self, filters: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.pointwise = nn.Sequential(SeparableConv2d(3, filters, 1), nn.LeakyReLU(0.3),
                                       nn.AvgPool2d(1))
        self.spatial = nn.Sequential(SeparableConv2d(3, filters, 3), nn.LeakyReLU(0.3),
                                     SeparableConv2d(filters, filters, 3), nn.LeakyReLU(0.3),
                                     nn.AvgPool2d(1), nn.Dropout(0.2))
        self.out_dim = 2 * filters

    def forward(self, x):
        return torch.cat([self.pointwise(x), self.spatial(x)], 1)  # c_j = [h_j ; k_j]


# --------------------------------------------------------------------------- handcrafted M1..M5
def _gabor_bank():
    ks = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        for sigma in (1, 3):
            for lamda in (np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi):
                for gamma in (0.05, 0.5):
                    ks.append(cv2.getGaborKernel((9, 9), sigma, theta, lamda, gamma, 0, cv2.CV_32F))
    return ks  # 64 kernels


class Handcrafted:
    """Pixel-wise feature bank used by the paper's benchmarks (Table 3).

    M1: RGB + grey + 64 Gabor;  M2: M1 + LBP;  M3: M2 + Canny/Roberts/Sobel/Scharr/Prewitt;
    M4: M3 + k-means cluster id;  M5: M4 + Gaussian(s=3,7) + median(3)."""
    trainable = False

    def __init__(self, level: str = "M5"):
        self.level = int(level[1])
        self.gabor = _gabor_bank()
        self.out_dim = 4 + 64 + (self.level >= 2) + 5 * (self.level >= 3) + (self.level >= 4) + 3 * (self.level >= 5)

    def _one(self, rgb: np.ndarray) -> np.ndarray:
        from skimage import filters
        from skimage.feature import local_binary_pattern
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gf = g.astype(np.float32) / 255
        f = [rgb[..., i].astype(np.float32) / 255 for i in range(3)] + [gf]
        f += [cv2.filter2D(gf, cv2.CV_32F, k) for k in self.gabor]
        if self.level >= 2:
            f.append(local_binary_pattern(g, 8, 1, "uniform").astype(np.float32) / 9)
        if self.level >= 3:
            f += [cv2.Canny(g, 100, 200).astype(np.float32) / 255, filters.roberts(gf), filters.sobel(gf),
                  filters.scharr(gf), filters.prewitt(gf)]
        if self.level >= 4:
            z = rgb.reshape(-1, 3).astype(np.float32)
            _, lab, cen = cv2.kmeans(z, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                     2, cv2.KMEANS_PP_CENTERS)
            order = np.argsort(cen.sum(1))  # cluster id sorted by darkness -> consistent across images
            rank = np.empty(3, int); rank[order] = np.arange(3)
            f.append(rank[lab.ravel()].reshape(g.shape).astype(np.float32) / 2)
        if self.level >= 5:
            f += [cv2.GaussianBlur(gf, (0, 0), 3), cv2.GaussianBlur(gf, (0, 0), 7), cv2.medianBlur(g, 3) / 255.0]
        return np.stack([np.asarray(a, np.float32) for a in f], 0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        imgs = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        return torch.from_numpy(np.stack([self._one(i) for i in imgs])).to(x.device)

    def eval(self):
        return self

    def to(self, *_):
        return self


# --------------------------------------------------------------------------- recent backbones
TIMM_MODELS = {
    "resnet50": "resnet50.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "convnext_t": "convnext_tiny.fb_in22k_ft_in1k",
    "swin_t": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "dinov2_s": "vit_small_patch14_dinov2.lvd142m",
    "vit_b": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
}


class TimmBackbone(nn.Module):
    """Frozen ImageNet/self-supervised backbone -> multi-scale maps upsampled to S x S and concatenated.
    CNNs/Swin use all pyramid stages; ViTs use 4 evenly spaced transformer blocks."""
    trainable = False

    def __init__(self, key: str, in_size: int = 224, pretrained: bool = True):
        super().__init__()
        import timm
        self.key, self.in_size = key, in_size
        name = TIMM_MODELS.get(key, key)
        self.is_vit = "vit" in name or "dinov2" in name
        self.nhwc = "swin" in name  # timm Swin feature maps are channels-last
        if self.is_vit:
            self.net = timm.create_model(name, pretrained=pretrained, img_size=in_size, num_classes=0)
            n = len(self.net.blocks)
            self.indices = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
            self.out_dim = self.net.embed_dim * 4
        else:
            self.net = timm.create_model(name, pretrained=pretrained, features_only=True)
            self.out_dim = sum(self.net.feature_info.channels())
        for p in self.net.parameters():
            p.requires_grad_(False)

    def pyramid(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = F.interpolate(x, size=(self.in_size, self.in_size), mode="bilinear", align_corners=False)
        x = (x - IMAGENET_MEAN.to(x)) / IMAGENET_STD.to(x)
        if self.is_vit:
            return self.net.forward_intermediates(x, indices=self.indices, output_fmt="NCHW",
                                                  intermediates_only=True)
        feats = self.net(x)
        return [f.permute(0, 3, 1, 2).contiguous() for f in feats] if self.nhwc else feats

    def forward(self, x):
        S = x.shape[-1]
        return torch.cat([F.interpolate(f, size=(S, S), mode="bilinear", align_corners=False)
                          for f in self.pyramid(x)], 1)


def build_extractor(key: str, **kw):
    if key == "paper":
        return PaperCNNFusion()
    if key in ("M1", "M2", "M3", "M4", "M5"):
        return Handcrafted(key)
    if key == "hybrid":
        from .hybrid import HybridAttentionFusion
        return HybridAttentionFusion(**kw)
    return TimmBackbone(key)


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# A "feature" is a list of numbers describing a pixel and its surroundings (colour, edges, texture
# ...). The classifiers later decide "WBC or not" for every pixel from these numbers. This file
# contains the different ways of producing them, so we can compare them fairly:
#  * PaperCNNFusion: the paper's small network - one branch looks at each pixel alone (1x1), the
#    other looks at 3x3 neighbourhoods twice; the two 64-number results are glued into 128 numbers.
#  * Handcrafted M1..M5: the paper's old-school comparison methods built from classic image filters
#    (Gabor texture filters, LBP, edge detectors, k-means colour groups, blurs), each adding more.
#  * TimmBackbone: famous modern pretrained networks (ResNet-50, EfficientNet, ConvNeXt, Swin
#    Transformer, DINOv2 Vision Transformer). We take their internal "maps" at several zoom levels,
#    stretch them back to the ROI size, and stack them so every pixel gets a rich description.

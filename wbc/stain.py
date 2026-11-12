"""Improvement 2: stain normalisation (Macenko / Reinhard) + CLAHE contrast enhancement, applied before Phase I."""
from __future__ import annotations

import cv2
import numpy as np


class MacenkoNormalizer:
    """Macenko et al., ISBI 2009: estimate H&E-like stain vectors via SVD in optical-density space
    and map source stain concentrations onto a reference image's stain matrix."""

    def __init__(self, Io: float = 240, alpha: float = 1, beta: float = 0.15):
        self.Io, self.alpha, self.beta = Io, alpha, beta
        self.HERef = None
        self.maxCRef = None

    def _stain_matrix(self, img: np.ndarray):
        od = -np.log((img.reshape(-1, 3).astype(np.float64) + 1) / self.Io)
        od_hat = od[~np.any(od < self.beta, axis=1)]
        if len(od_hat) < 10:
            raise ValueError("too few stained pixels")
        _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
        plane = eigvecs[:, 1:3]
        that = od_hat @ plane
        phi = np.arctan2(that[:, 1], that[:, 0])
        min_phi, max_phi = np.percentile(phi, self.alpha), np.percentile(phi, 100 - self.alpha)
        v_min = plane @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v_max = plane @ np.array([np.cos(max_phi), np.sin(max_phi)])
        HE = np.array([v_min, v_max]).T if v_min[0] > v_max[0] else np.array([v_max, v_min]).T
        HE = HE * np.sign(HE.sum(axis=0, keepdims=True))  # keep vectors positive
        C = np.linalg.lstsq(HE, od.T, rcond=None)[0]
        maxC = np.array([np.percentile(C[0], 99), np.percentile(C[1], 99)])
        return HE, C, maxC

    def fit(self, ref: np.ndarray):
        self.HERef, _, self.maxCRef = self._stain_matrix(ref)
        return self

    def transform(self, img: np.ndarray) -> np.ndarray:
        try:
            _, C, maxC = self._stain_matrix(img)
        except (ValueError, np.linalg.LinAlgError):
            return img
        C = C * (self.maxCRef / np.maximum(maxC, 1e-6))[:, None]
        out = self.Io * np.exp(-self.HERef @ C)
        return np.clip(out.T.reshape(img.shape), 0, 255).astype(np.uint8)


class ReinhardNormalizer:
    """Reinhard et al. 2001: match per-channel mean/std in LAB colour space."""

    def fit(self, ref: np.ndarray):
        lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
        self.mu, self.sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
        return self

    def transform(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        mu, sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
        lab = (lab - mu) / sd * self.sd + self.mu
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def clahe(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """CLAHE on the L channel only, so hue (stain colour) is preserved."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class Preprocessor:
    def __init__(self, method: str = "macenko", use_clahe: bool = True):
        self.method, self.use_clahe = method, use_clahe
        self.norm = {"macenko": MacenkoNormalizer, "reinhard": ReinhardNormalizer}.get(method, lambda: None)()

    def fit(self, ref: np.ndarray):
        if self.norm is not None:
            self.norm.fit(ref)
        return self

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.norm is not None:
            img = self.norm.transform(img)
        if self.use_clahe:
            img = clahe(img)
        return img

    def __repr__(self):
        return f"{self.method}{'+clahe' if self.use_clahe else ''}"


def pick_reference(images: list[np.ndarray]) -> np.ndarray:
    """Choose the training image whose mean LAB colour is closest to the median -> a 'typical' stain."""
    feats = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2LAB).reshape(-1, 3).mean(0) for im in images])
    med = np.median(feats, axis=0)
    return images[int(np.argmin(np.linalg.norm(feats - med, axis=1)))]


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# Different labs stain blood slides slightly differently, so the same cell can look more blue in
# one hospital and more pink in another. This file evens that out before anything else happens.
# Macenko works out the two dye colours present in an image and re-paints it using the dye colours
# of one chosen "reference" image; Reinhard is a simpler version that just matches the overall
# colour averages. CLAHE then boosts local contrast (brightness only, not colour) so faint cells
# stand out. `pick_reference` automatically chooses the most "typical-looking" training image as
# the reference, and `Preprocessor` bundles everything into one call: clean = pre(image).

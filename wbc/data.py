"""Dataset loading: real blood-smear folders (+ optional masks) and a synthetic smoke-test set."""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MASK_SUFFIXES = ("", "_mask", "-mask", "_gt", "_GT", "_label", "_seg")


@dataclass
class Sample:
    name: str
    image: np.ndarray                 # H x W x 3, RGB uint8
    mask: np.ndarray | None = None    # H x W uint8 {0,1}  (1 = WBC), None if unlabeled
    label: str = ""                   # morphological class (folder name), e.g. "BAS"
    meta: dict = field(default_factory=dict)


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype != np.uint8:  # 16-bit TIFFs in some datasets
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _find_mask(mask_dir: str, stem: str, image_path: str) -> str | None:
    for suf in MASK_SUFFIXES:
        for ext in IMG_EXT:
            p = os.path.join(mask_dir, stem + suf + ext)
            if os.path.exists(p) and os.path.abspath(p) != os.path.abspath(image_path):
                return p
    return None


def load_folder(img_dir: str, mask_dir: str | None = None, limit: int | None = None,
                max_side: int = 400, per_class_limit: int | None = None, seed: int = 0,
                image_ext: str | None = None) -> list[Sample]:
    """Load images recursively. Sub-folder name is used as class label (AML_Cytomorphology_LMU layout).

    image_ext: only treat files with this extension as images (needed when images and masks share a
    folder, e.g. Zheng et al.: 001.bmp image + 001.png mask -> image_ext=".bmp").
    Masks: any non-zero pixel = WBC (works for 0/255 binary masks and for 0/128/255
    nucleus+cytoplasm masks such as Zheng et al. 2018).
    """
    for d, what in ((img_dir, "image"), (mask_dir, "mask")):
        if d and not os.path.isdir(d):
            raise FileNotFoundError(f"{what} folder does not exist: {os.path.abspath(d)}  "
                                    f"(download a dataset first - see README 'Data')")
    paths = sorted(p for p in glob.glob(os.path.join(img_dir, "**", "*"), recursive=True)
                   if p.lower().endswith((image_ext.lower(),) if image_ext else IMG_EXT))
    if not paths:
        raise RuntimeError(f"no image files ({', '.join(IMG_EXT)}) found under {os.path.abspath(img_dir)}")
    if mask_dir and os.path.abspath(mask_dir) != os.path.abspath(img_dir):
        mask_abs = os.path.abspath(mask_dir)
        paths = [p for p in paths if not os.path.abspath(p).startswith(mask_abs)]
    rng = random.Random(seed)
    rng.shuffle(paths)
    if per_class_limit:
        counts: dict[str, int] = {}
        kept = []
        for p in paths:
            c = os.path.basename(os.path.dirname(p))
            if counts.get(c, 0) < per_class_limit:
                kept.append(p)
                counts[c] = counts.get(c, 0) + 1
        paths = kept
    if limit:
        paths = paths[:limit]

    out = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        img = read_rgb(p)
        mask = None
        if mask_dir:
            mp = _find_mask(mask_dir, stem, p)
            if mp is None:
                continue  # only keep annotated images when masks are requested
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            mask = (m > 0).astype(np.uint8)
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(mask, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        scale = max_side / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = cv2.resize(mask, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        out.append(Sample(stem, img, mask, os.path.basename(os.path.dirname(p))))
    if not out:
        raise RuntimeError(f"found {len(paths)} images in {img_dir} but none had a matching mask in {mask_dir} "
                           f"(masks must share the image's file name, optionally with _mask/_gt suffix)")
    return out


# ---------------------------------------------------------------------------------------------
# Synthetic smear generator -- ONLY for smoke-testing the code path, never for reporting results.
# ---------------------------------------------------------------------------------------------
def make_synthetic(n: int = 40, size: int = 256, seed: int = 0) -> list[Sample]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        bg_tint = rng.uniform(-15, 15, 3)
        img = np.full((size, size, 3), (238, 222, 230), np.float32) + bg_tint
        # red blood cells: pink discs with paler centre
        for _ in range(rng.integers(8, 16)):
            c = rng.integers(0, size, 2)
            r = int(rng.integers(16, 26))
            cv2.circle(img, tuple(int(v) for v in c), r, (222, 160, 175), -1)
            cv2.circle(img, tuple(int(v) for v in c), r // 2, (232, 190, 200), -1)
        mask = np.zeros((size, size), np.uint8)
        cx, cy = rng.integers(size // 3, 2 * size // 3, 2)
        rc = int(rng.integers(34, 50))
        axes = (rc, int(rc * rng.uniform(0.8, 1.0)))
        ang = float(rng.uniform(0, 180))
        cyto = tuple(float(v) for v in rng.uniform([170, 140, 200], [215, 180, 235]))
        cv2.ellipse(img, (int(cx), int(cy)), axes, ang, 0, 360, cyto, -1)
        cv2.ellipse(mask, (int(cx), int(cy)), axes, ang, 0, 360, 1, -1)
        nuc = tuple(float(v) for v in rng.uniform([70, 30, 120], [110, 60, 160]))
        for _ in range(rng.integers(1, 5)):  # lobed nucleus
            off = rng.integers(-rc // 3, rc // 3 + 1, 2)
            cv2.circle(img, (int(cx + off[0]), int(cy + off[1])), int(rc * rng.uniform(0.3, 0.5)), nuc, -1)
        img = cv2.GaussianBlur(img, (5, 5), 1.2) + rng.normal(0, 4, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        out.append(Sample(f"syn_{i:03d}", img, mask, label="SYN"))
    return out


def split(samples: list[Sample], test_frac: float = 0.2, seed: int = 0):
    """Image-level split (paper split pixels 80/20, which leaks neighbouring pixels between sets)."""
    idx = list(range(len(samples)))
    random.Random(seed).shuffle(idx)
    n_test = max(1, int(round(len(idx) * test_frac)))
    test = [samples[i] for i in idx[:n_test]]
    train = [samples[i] for i in idx[n_test:]]
    return train, test


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# This file gets the pictures into the program. `load_folder` walks through a folder of blood
# smear images (sub-folder names like "BAS" or "MON" become the cell type), shrinks very large
# images, and, if you also give it a folder of masks, pairs every image with its black-and-white
# "answer sheet" that marks which pixels belong to the white blood cell. `make_synthetic` draws
# fake smear pictures (pink red cells plus one purple white cell) purely so we can check the whole
# program runs before real data is downloaded. `split` divides the images into a training pile
# (80%) and a testing pile (20%) by whole image, so the test cells were never seen in training.

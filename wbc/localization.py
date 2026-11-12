"""Phase I: ROI localisation.

* ``OtsuCMYKLocalizer`` -- faithful re-implementation of the paper's CMYK-moment method
  (RGB->CMYK, C channel, Otsu, 5x5 opening, maximum connected region, image moments, circle, square).
* ``SamCMYKLocalizer``  -- our replacement of the Otsu step with the Segment Anything Model (SAM):
  the C channel only provides a *prompt point*; SAM segments the nucleus, image moments give the
  centroid (as in the paper), and a second SAM call with a box prompt segments the whole cell, whose
  extent defines the ROI square (instead of a heuristic nucleus-diameter margin).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class Localization:
    nucleus: np.ndarray            # H x W {0,1}
    centroid: tuple[float, float]  # (cx, cy) from image moments
    radius: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (square, inside image)
    cell: np.ndarray | None = None  # whole-cell mask (SAM only)
    seconds: float = 0.0


# --------------------------------------------------------------------------------- helpers
def cyan_channel(img: np.ndarray) -> np.ndarray:
    """RGB uint8 -> C of CMYK in [0,1]:  K = 1-max(R',G',B'),  C = (1-R'-K)/(1-K)."""
    rgb = img.astype(np.float32) / 255.0
    k = 1.0 - rgb.max(axis=2)
    return np.where(k < 1, (1 - rgb[..., 0] - k) / np.maximum(1 - k, 1e-6), 0).astype(np.float32)


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(np.uint8)
    return (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def moment_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Eq. (1)-(3): M_ij = sum x^i y^j I(x,y); Cx = M10/M00, Cy = M01/M00."""
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        h, w = mask.shape
        return w / 2, h / 2
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def max_radius(mask: np.ndarray, c: tuple[float, float]) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return min(mask.shape) / 4
    return float(np.sqrt((xs - c[0]) ** 2 + (ys - c[1]) ** 2).max())


def square_box(c, r, shape) -> tuple[int, int, int, int]:
    """Square of side 2r centred on c, shifted (not truncated) to stay inside the image."""
    h, w = shape[:2]
    side = int(min(max(2 * r, 16), h, w))
    x0 = int(round(c[0] - side / 2)); y0 = int(round(c[1] - side / 2))
    x0 = min(max(x0, 0), w - side); y0 = min(max(y0, 0), h - side)
    return x0, y0, x0 + side, y0 + side


def crop_resize(arr: np.ndarray, box, size: int, nearest: bool = False) -> np.ndarray:
    x0, y0, x1, y1 = box
    return cv2.resize(arr[y0:y1, x0:x1], (size, size),
                      interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


# --------------------------------------------------------------------------------- paper (Otsu)
class OtsuCMYKLocalizer:
    name = "otsu"

    def __init__(self, cell_margin: float = 1.35):
        self.cell_margin = cell_margin  # nucleus radius -> cell radius (paper draws circle "around the nucleus")

    def nucleus_mask(self, img: np.ndarray) -> np.ndarray:
        c = (cyan_channel(img) * 255).astype(np.uint8)
        _, bw = cv2.threshold(c, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # S(5,5) opening
        return largest_component(bw)                                           # MCR

    def __call__(self, img: np.ndarray) -> Localization:
        t = time.perf_counter()
        nuc = self.nucleus_mask(img)
        cxy = moment_centroid(nuc)
        r = max_radius(nuc, cxy) * self.cell_margin
        return Localization(nuc, cxy, r, square_box(cxy, r, img.shape), None, time.perf_counter() - t)


# --------------------------------------------------------------------------------- SAM
class SamCMYKLocalizer:
    """CMYK-moment localisation with SAM instead of Otsu.

    model_id: "facebook/sam-vit-base" (94M), "facebook/sam-vit-large", "facebook/sam-vit-huge",
    or any SAM-compatible HF checkpoint (e.g. "flaviagiammarino/medsam-vit-base" for MedSAM).
    """
    name = "sam"

    def __init__(self, model_id: str = "facebook/sam-vit-base", device: str | None = None,
                 margin: float = 1.12):
        from transformers import SamModel, SamProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device).eval()
        self.margin = margin
        self.model_id = model_id

    # -- seed point from the C channel (what Otsu thresholded in the paper)
    @staticmethod
    def seed_point(img: np.ndarray) -> tuple[float, float]:
        c = cv2.GaussianBlur(cyan_channel(img), (0, 0), 3)
        hi = (c >= np.percentile(c, 97)).astype(np.uint8)
        return moment_centroid(largest_component(hi))

    @torch.no_grad()
    def _candidates(self, img, emb, inputs_kw):
        """Return SAM's 3 multimask candidates (3,H,W) and their predicted IoU scores."""
        inputs = self.processor(images=img, return_tensors="pt", **inputs_kw)
        prompt = {k: v.to(self.device) for k, v in inputs.items()
                  if k in ("input_points", "input_labels", "input_boxes")}
        out = self.model(image_embeddings=emb, multimask_output=True, **prompt)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"],
                                                  inputs["reshaped_input_sizes"])[0][0].numpy()
        return masks.astype(np.uint8), out.iou_scores.cpu().numpy().reshape(-1)

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> Localization:
        t = time.perf_counter()
        pix = self.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)
        emb = self.model.get_image_embeddings(pix)
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        px, py = self.seed_point(img)
        h, w = img.shape[:2]
        # 1) nucleus: positive point prompt on the most cyan region
        masks, sc = self._candidates(img, emb, dict(input_points=[[[[px, py]]]], input_labels=[[[1]]]))
        area = masks.reshape(3, -1).mean(1)
        sc = np.where((area > 0.002) & (area < 0.35), sc, -1)
        nuc = largest_component(masks[int(np.argmax(sc))])
        cxy = moment_centroid(nuc)
        r_nuc = max_radius(nuc, cxy)
        # 2) whole cell: box prompt (paper's circle -> square) + positive point on the nucleus centroid
        bx = square_box(cxy, 2 * r_nuc * 1.6, img.shape)
        # negatives at the box corners: a square drawn around the cell circle has its corners outside
        # the cell, which stops SAM from returning "everything in the box" on tightly cropped images
        x0, y0, x1, y1 = bx
        corners = [[x0 + 2, y0 + 2], [x1 - 3, y0 + 2], [x0 + 2, y1 - 3], [x1 - 3, y1 - 3]]
        pts = [list(map(float, cxy))] + [list(map(float, c)) for c in corners]
        masks, sc = self._candidates(img, emb, dict(input_boxes=[[list(map(float, bx))]],
                                                    input_points=[[pts]], input_labels=[[[1, 0, 0, 0, 0]]]))
        inbox = np.zeros((h, w), np.uint8); inbox[bx[1]:bx[3], bx[0]:bx[2]] = 1
        ring = inbox - cv2.erode(inbox, np.ones((3, 3), np.uint8))  # 1-px border of the prompt box
        box_area = max(int(inbox.sum()), 1)
        best, best_s = None, -np.inf
        for m, s_ in zip(masks, sc):
            m = largest_component(np.maximum(m, nuc))
            leak = 1 - (m * inbox).sum() / max(m.sum(), 1)       # fraction outside the prompt box
            ratio = m.sum() / max(nuc.sum(), 1)                    # cell / nucleus area
            border = (m * ring).sum() / max(ring.sum(), 1)         # a cell sits inside the box; background hugs it
            fill = (m * inbox).sum() / box_area
            ok = 1.0 <= ratio <= 8.0 and leak < 0.15 and border < 0.35 and fill < 0.85
            if ok and s_ - leak > best_s:
                best, best_s = m * inbox, s_ - leak
        cell = nuc if best is None else np.maximum(largest_component(best), nuc)
        if cell.sum() < nuc.sum() * 1.05:  # no plausible cell mask -> nucleus + paper margin
            r = r_nuc * 1.6
        else:
            r = max(max_radius(cell, cxy) * self.margin, r_nuc * 1.6)
        return Localization(nuc, cxy, r, square_box(cxy, r, img.shape), cell, time.perf_counter() - t)


def build_localizer(name: str, **kw):
    if name == "otsu":
        return OtsuCMYKLocalizer()
    if name == "sam":
        return SamCMYKLocalizer(**kw)
    raise ValueError(name)


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# The goal here is to find the white blood cell in the picture and cut out a square around it
# (the "region of interest", ROI), so later steps only look at the cell and not at the clutter.
# White-cell nuclei soak up the purple stain, so in the "Cyan" channel of CMYK colour they are the
# brightest thing in the image.
#  * The paper's way (OtsuCMYKLocalizer): pick a brightness cut-off automatically (Otsu), keep the
#    biggest blob, find its centre with "image moments" (a weighted average of pixel positions),
#    draw a circle around it and a square around the circle.
#  * Our way (SamCMYKLocalizer): use the cyan channel only to point at the nucleus, then ask the
#    Segment Anything Model (SAM), a large pretrained AI segmenter, to outline the nucleus exactly.
#    We still take the centre with image moments like the paper, then ask SAM a second time with a
#    box to outline the whole cell, and size the square to fit that outline. This copes better with
#    faint cells, odd colours and touching cells than a single brightness cut-off.

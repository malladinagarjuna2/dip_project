"""Shared trainer for every trainable pixel-wise model (hybrid extractor head, U-Net, SegFormer)."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F


def dice_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def augment(x: torch.Tensor, y: torch.Tensor):
    """Random flips / 90-degree rotations + mild brightness/saturation jitter (staining robustness)."""
    if np.random.rand() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if np.random.rand() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    k = np.random.randint(4)
    x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
    b = x.shape[0]
    gain = 1 + 0.1 * (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)
    grey = x.mean(1, keepdim=True)
    sat = 1 + 0.2 * (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)
    x = ((grey + (x - grey) * sat) * gain).clamp(0, 1)
    return x, y


def fit_segmenter(model, X: torch.Tensor, Y: torch.Tensor, epochs: int = 15, bs: int = 16,
                  lr: float = 1e-3, device: str = "cuda", log=print) -> float:
    """X: (N,3,S,S) float [0,1] on CPU;  Y: (N,1,S,S) {0,1}.  model(x) -> logits (B,1,S,S)."""
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, int(np.ceil(len(X) / bs)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    t = time.perf_counter()
    for ep in range(epochs):
        perm, tot = torch.randperm(len(X)), 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            x, y = augment(X[idx].to(device), Y[idx].to(device).float())
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x)
            loss = dice_bce(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(idx)
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            log(f"    epoch {ep + 1}/{epochs}  loss {tot / len(X):.4f}")
    model.eval()
    return time.perf_counter() - t


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# Some of our models have to learn from examples. `fit_segmenter` is the teaching loop: it shows
# the model batches of cell crops together with the correct WBC masks, measures how wrong the
# prediction is (`dice_bce` mixes pixel-by-pixel error with an overlap score), and nudges the model
# weights to do better, repeating for several passes ("epochs"). `augment` randomly flips, rotates
# and slightly re-colours each crop every time, so the model learns that a cell is still a cell
# when turned around or stained a bit differently.

"""Model complexity: parameters (total / trainable), GFLOPs per ROI, and latency per ROI."""
from __future__ import annotations

import time

import torch
import torch.nn as nn


def count_params(m) -> tuple[int, int]:
    if not isinstance(m, nn.Module):
        return 0, 0
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return tot, tr


@torch.no_grad()
def gflops(fn, x: torch.Tensor) -> float:
    """Counts FLOPs with torch's built-in FlopCounterMode (matmul/conv; 1 MAC = 2 FLOPs)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fc = FlopCounterMode(display=False)
        with fc:
            fn(x)
        return fc.get_total_flops() / 1e9 / x.shape[0]
    except Exception:
        return float("nan")


@torch.no_grad()
def latency_ms(fn, x: torch.Tensor, reps: int = 10) -> float:
    for _ in range(2):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps / x.shape[0] * 1000


def profile(model, fn, size: int, device: str, batch: int = 4) -> dict:
    x = torch.rand(batch, 3, size, size, device=device)
    tot, tr = count_params(model)
    return {"params_M": tot / 1e6, "trainable_params_M": tr / 1e6,
            "gflops_per_roi": gflops(fn, x[:1]) if isinstance(model, nn.Module) else 0.0,
            "latency_ms_per_roi": latency_ms(fn, x)}


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# Accuracy is not everything: a hospital also cares how big and how slow a model is. For each model
# this file measures (1) the number of parameters - the learned numbers inside it, in millions, and
# how many of those we actually trained; (2) GFLOPs - billions of arithmetic operations needed to
# process one cell crop; and (3) latency - milliseconds to process one crop on this computer.

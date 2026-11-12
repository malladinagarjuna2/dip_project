"""Evaluation metrics, Eqs. (8)-(14) of the paper, plus Dice and a true ROC-AUC."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def pixel_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> dict:
    y = y_true.astype(bool).ravel()
    p = prob.ravel()
    yhat = p >= thr
    tp = np.sum(yhat & y); tn = np.sum(~yhat & ~y); fp = np.sum(yhat & ~y); fn = np.sum(~yhat & y)
    eps = 1e-12
    sens = tp / (tp + fn + eps); spec = tn / (tn + fp + eps); prec = tp / (tp + fp + eps)
    try:
        auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),       # (8)
        "precision": prec,                                       # (9)
        "sensitivity": sens,                                     # (10)
        "specificity": spec,                                     # (11)
        "f1": 2 * prec * sens / (prec + sens + eps),             # (12)
        "auc": auc,                                              # true ROC-AUC over thresholds
        "auc_paper": (sens + prec) / 2,                          # (13) exactly as printed in the paper*
        "balanced_acc": (sens + spec) / 2,
        "jaccard": tp / (tp + fp + fn + eps),                    # (14) IoU
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
    }
# * The paper's Eq. (13) is (sensitivity + precision)/2 and its Table 1 numbers match it (e.g. FCL:
#   (0.9584+0.9941)/2 = 0.976). That is not a real ROC-AUC, so we report both: `auc_paper` for a
#   like-for-like comparison with the paper, and `auc` (proper ROC-AUC) for scientific correctness.


def per_image_dice(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """y_true, pred: (N,S,S) binary -> Dice per image (mean +- std is reported)."""
    yt = y_true.reshape(len(y_true), -1).astype(bool); pr = pred.reshape(len(pred), -1).astype(bool)
    inter = (yt & pr).sum(1)
    return 2 * inter / np.maximum(yt.sum(1) + pr.sum(1), 1)


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# This file scores the predictions against the true mask, pixel by pixel. TP = WBC pixels found,
# TN = background correctly ignored, FP = background wrongly called WBC, FN = WBC pixels missed.
# From these we compute the same scores as the paper: accuracy (how many pixels are right),
# sensitivity (how much of the cell we found), specificity (how much background we correctly left
# out), precision (how trustworthy a "WBC" answer is), F1 (balance of the last two), Jaccard/IoU and
# Dice (how well the predicted shape overlaps the true shape), and AUC (quality across all
# confidence cut-offs). We also report the paper's own simplified AUC formula for comparison.

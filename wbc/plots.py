"""Figures + markdown tables written next to results.csv."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# validated categorical palette (dataviz reference), fixed order: classifier -> slot
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "D", "^"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CLF_ORDER = ["FCL", "SVM", "RF", "XGBoost", "-"]


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def dot_plot(res: pd.DataFrame, metric: str, dataset: str, path: str):
    d = res[res.dataset == dataset]
    if d.empty:
        return
    order = d.groupby("display")[metric].max().sort_values().index.tolist()
    y = {m: i for i, m in enumerate(order)}
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(order) + 1.4), facecolor=SURF)
    _style(ax)
    clfs = [c for c in CLF_ORDER if c in d.classifier.unique()]
    for j, c in enumerate(clfs):
        dc = d[d.classifier == c]
        label = "end-to-end model" if c == "-" else c
        color = INK2 if c == "-" else SERIES[j % 4]
        ax.scatter(dc[metric] * 100, [y[m] for m in dc.display], s=42, marker=MARKERS[j % 4] if c != "-" else "*",
                   color=color, edgecolor=SURF, linewidth=1.5, zorder=3, label=label)
    for m in order:  # range line per method
        v = d[d.display == m][metric] * 100
        ax.plot([v.min(), v.max()], [y[m]] * 2, color=GRID, lw=2, zorder=2)
    ax.set_yticks(range(len(order)), order, fontsize=8, color=INK)
    for lbl in ax.get_yticklabels():
        if "ours" in lbl.get_text():
            lbl.set_fontweight("bold")
    ax.set_xlabel(f"{metric.replace('_', ' ')} (%) - {dataset} test set", color=INK2, fontsize=9)
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=1)
    ax.set_title(f"Pixel-wise WBC {metric} by feature extractor and classifier", loc="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def efficiency_plot(res: pd.DataFrame, cp: pd.DataFrame, path: str, metric: str = "f1"):
    d = res[res.dataset == "primary"].groupby("method")[metric].max().rename("score").reset_index()
    d = d.merge(cp[["method", "display", "params_M", "latency_ms_per_roi"]], on="method")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURF)
    _style(ax); ax.grid(axis="y", color=GRID, lw=0.6)
    groups = [("Proposed (ours)", d.method.str.startswith("hybrid"), SERIES[0]),
              ("Paper 2022", d.method == "paper", SERIES[1]),
              ("Other methods", ~d.method.str.startswith("hybrid") & (d.method != "paper"), "#a3a29c")]
    for name, sel, col in groups:
        dd = d[sel]
        ax.scatter(dd.params_M.clip(lower=1e-3), dd.score * 100, s=60, color=col, edgecolor=SURF, lw=2,
                   label=name, zorder=3)
        for _, r in dd.iterrows():
            ax.annotate(r.display.replace(" (ours)", "").replace(" (e2e)", ""), (max(r.params_M, 1e-3), r.score * 100),
                        textcoords="offset points", xytext=(5, 3), fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions, log scale; handcrafted = 0.001)", color=INK2, fontsize=9)
    ax.set_ylabel(f"best {metric.upper()} (%)", color=INK2, fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Model size vs. segmentation quality", loc="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


PRIORITY = ["paper", "M5", "resnet50", "dinov2_s", "hybrid", "unet", "segformer_b0", "sam", "hybrid_e2e"]


def qualitative_grid(vis: dict, path: str, clf: str = "SVM", max_rows: int = 9):
    preds = {k: v for k, v in vis["pred"].items() if k[1] in (clf, "-")}
    if not preds:
        return
    # paper vs. best handcrafted vs. recent vs. ours vs. end-to-end - so our rows are never cut off
    keys = sorted(preds, key=lambda k: PRIORITY.index(k[0]) if k[0] in PRIORITY else 99)[:max_rows]
    n = len(vis["image"])
    rows = 2 + len(keys)
    fig, axes = plt.subplots(rows, n, figsize=(1.25 * n, 1.25 * rows + 0.3), facecolor="white")
    axes = np.atleast_2d(axes)
    from run import DISPLAY
    names = ["Original ROI", "Ground truth"] + [DISPLAY.get(k[0], k[0]).replace(" (ours)", "*") for k in keys]
    for j in range(n):
        img = vis["image"][j]
        mats = [img, None] + [preds[k][j] for k in keys]
        for i, m in enumerate(mats):
            ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.imshow(img)
            else:
                mask = vis["gt"][j] if i == 1 else m
                ax.imshow(np.where(mask[..., None] > 0, img, 1.0))  # paper style: WBC on white
            if j == 0:
                ax.set_ylabel(names[i], fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle(f"Segmented WBCs ({clf} for feature-based methods; * = ours)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def tables(res: pd.DataFrame, cp: pd.DataFrame, out: str):
    cols = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc", "auc_paper", "jaccard", "dice"]
    with open(os.path.join(out, "tables.md"), "w", encoding="utf-8") as f:
        for ds in res.dataset.unique():
            d = res[res.dataset == ds][["display", "classifier", "n_features"] + cols].copy()
            d[cols] = d[cols].round(4)
            f.write(f"## Table - {ds} dataset (cf. paper Tables 1-3)\n\n{d.to_markdown(index=False)}\n\n")
        c = cp.copy()
        for k in ("params_M", "trainable_params_M", "gflops_per_roi", "latency_ms_per_roi", "train_s"):
            c[k] = c[k].astype(float).round(3)
        f.write("## Parameter / complexity comparison\n\n" + c.drop(columns=["method"]).to_markdown(index=False) + "\n")


def make_all(out: str):
    res = pd.read_csv(os.path.join(out, "results.csv"))
    cp = pd.read_csv(os.path.join(out, "complexity.csv"))
    for ds in res.dataset.unique():
        dot_plot(res, "accuracy", ds, os.path.join(out, f"fig_accuracy_{ds}.png"))
        dot_plot(res, "jaccard", ds, os.path.join(out, f"fig_jaccard_{ds}.png"))
    efficiency_plot(res, cp, os.path.join(out, "fig_params_vs_f1.png"))
    vp = os.path.join(out, "vis.pt")
    if os.path.exists(vp):
        qualitative_grid(torch.load(vp, weights_only=False), os.path.join(out, "fig_qualitative.png"))
    try:
        tables(res, cp, out)
    except ImportError:  # DataFrame.to_markdown needs 'tabulate'
        pass


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# This file draws the result pictures and writes the result tables after a run:
#  * fig_accuracy_*.png / fig_jaccard_*.png - one row per method, one coloured dot per classifier,
#    sorted best-to-worst, so you can see at a glance which feature extractor wins (like Fig. 10/11).
#  * fig_params_vs_f1.png - how big each model is (x) against how well it segments (y); top-left is
#    best (small and accurate). Our models are blue, the paper's is orange, the rest grey.
#  * fig_qualitative.png - the actual cut-out cells for several test images and every method,
#    with the background whitened like the paper's Fig. 8/12.
#  * tables.md - the paper-style metric tables plus the parameters/GFLOPs/speed comparison.

"""Ablation study: isolates the contribution of each change w.r.t. the paper.

    Phase I  : Otsu (paper)  vs  SAM (ours)
    Improv. 2: no normalisation  vs  Reinhard+CLAHE  vs  Macenko+CLAHE
    Improv. 3: paper CNN  vs  hybrid variants (no attention / CNN-only / ViT-only / no paper branches / Swin)

Usage:
    python ablation.py --images ... --masks ... [--ext-images ... --ext-masks ...] --out results/ablation
    python ablation.py --synthetic 40 --out results/ablation_smoke --epochs 10
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

import run as R


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", default="results/ablation")
    a.add_argument("--part", default="pipeline,extractor", help="pipeline = localiser x stain grid; extractor = hybrid variants")
    known, rest = a.parse_known_args()
    rows = []

    if "pipeline" in known.part:
        for loc in ("otsu", "sam"):
            for stain, clahe in (("none", False), ("reinhard", True), ("macenko", True)):
                tag = f"{loc}_{stain}{'_clahe' if clahe else ''}"
                sys.argv = ["run.py", *rest, "--localizer", loc, "--stain", stain,
                            "--extractors", "paper,hybrid", "--classifiers", "svm,xgb", "--e2e", "",
                            "--out", os.path.join(known.out, tag)] + ([] if clahe else ["--no-clahe"])
                res, _ = R.main(R.parse())
                locdf = pd.read_csv(os.path.join(known.out, tag, "localization.csv"))
                cov = locdf[locdf.set != "train"].coverage.mean()
                rows.append(res.assign(localizer=loc, preprocessing=f"{stain}{'+CLAHE' if clahe else ''}",
                                       roi_wbc_coverage=cov))

    if "extractor" in known.part:
        sys.argv = ["run.py", *rest, "--extractors",
                    "paper,hybrid,hybrid_noattn,hybrid_cnnonly,hybrid_vitonly,hybrid_nolocal,hybrid_swin",
                    "--classifiers", "fcl,svm,rf,xgb", "--e2e", "", "--out", os.path.join(known.out, "extractor_variants")]
        res, _ = R.main(R.parse())
        rows.append(res.assign(localizer="(run default)", preprocessing="(run default)", roi_wbc_coverage=float("nan")))

    df = pd.concat(rows)
    df.to_csv(os.path.join(known.out, "ablation.csv"), index=False)
    cols = ["localizer", "preprocessing", "display", "classifier", "dataset"]
    summary = df.groupby(cols)[["accuracy", "f1", "jaccard", "auc", "roi_wbc_coverage"]].mean().round(4)
    with open(os.path.join(known.out, "ablation.md"), "w", encoding="utf-8") as f:
        f.write(summary.reset_index().to_markdown(index=False))
    print(summary.to_string())


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# An ablation study switches improvements on and off one at a time to prove which one actually
# helps. Part "pipeline" runs every combination of {Otsu, SAM} localisation with {no colour
# normalisation, Reinhard, Macenko} (+CLAHE), each time comparing the paper's CNN features with our
# hybrid features, and also records how much of the true cell each ROI captured. Part "extractor"
# removes pieces of our hybrid model (attention, CNN expert, Transformer expert, paper branches, or
# swaps DINOv2 for Swin) to show what each piece contributes. All results are merged into
# ablation.csv and a readable ablation.md table.

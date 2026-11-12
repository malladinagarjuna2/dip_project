# WBC feature extraction: CMYK-Moment + SAM, stain normalisation and hybrid CNN-Transformer features

Re-implementation of **Elhassan et al., "Feature Extraction of White Blood Cells Using CMYK-Moment
Localization and Deep Learning in AML Blood Smear Microscopic Images", IEEE Access 10 (2022)**, plus:

| Change | What we do | Where |
|---|---|---|
| **Otsu → SAM** | Phase I keeps CMYK and image moments, but the nucleus is segmented by **SAM**, prompted with a point taken from the C channel. A second SAM call (box prompt) segments the whole cell, and its extent sizes the ROI. | `wbc/localization.py` |
| **Improvement 2: stain normalisation** | Macenko (or Reinhard) normalisation to a reference image picked automatically from the training set, then **CLAHE** on the L channel, applied before Phase I | `wbc/stain.py` |
| **Improvement 3: advanced features** | The paper's 1×1 and 3×3 branches + **ConvNeXt-T** multi-scale maps + **DINOv2 ViT-S** (or **Swin-T**) global features, fused with **CBAM** attention into 128-D per pixel (the same size as the paper's features) | `wbc/hybrid.py` |

## What gets compared

* **Feature extractors × classifiers (FCL / SVM / RF / XGBoost)**, as in Tables 1–3 of the paper:
  * paper CNN fusion
  * handcrafted M1–M5
  * recent backbones: ResNet-50, ConvNeXt-T, Swin-T, DINOv2 ViT-S, EfficientNet-B0, ViT-B
  * our hybrid model
* **Recent end-to-end segmenters:** U-Net, SegFormer-B0, SAM ViT-B zero-shot, and our hybrid with its own head.
* **Parameters:** total and trainable parameters, GFLOPs per ROI, latency per ROI, and training time (`complexity.csv`).
* **Metrics:** accuracy, sensitivity, specificity, precision, F1, Jaccard/IoU, Dice and true ROC-AUC. We also report
  `auc_paper` = (sens+prec)/2, the formula printed as Eq. 13, so results can be compared with the paper's tables.
* **Localisation:** fraction of the ground-truth WBC captured by the ROI, and whether the centroid falls inside the WBC (Otsu vs SAM).
* **Ablations (`ablation.py`):**
  * {Otsu, SAM} × {none, Reinhard+CLAHE, Macenko+CLAHE}
  * hybrid without attention, CNN-only, ViT-only, without the paper's branches, and Swin instead of DINOv2

## Data

Pixel-wise metrics need **ground-truth WBC masks**. The paper's annotations were not released.

| Dataset | Use | Masks? |
|---|---|---|
| AML_Cytomorphology_LMU (TCIA, doi:10.7937/tcia.2019.36f5o9ld), 18,365 images, 15 classes | primary (paper) | **no**: annotate a subset, or use `--pseudo-gt` |
| PBC dataset, Acevedo et al. 2020 (Mendeley Data), 17,092 images | secondary/external (paper) | no |
| Zheng et al. 2018 WBC segmentation sets (GitHub `zxaoyou/segmentation_WBC`), Raabin-WBC GT subset | external sets with real masks | **yes** |

Folder layout: `images/<CLASS>/*.tiff|png|jpg` (class sub-folders are optional). Masks go in a parallel folder,
one file per image with the same stem (`_mask`, `_gt` and similar suffixes are accepted). Any non-zero mask pixel counts as WBC.

`--pseudo-gt` uses SAM's whole-cell masks as labels when no masks exist. Metrics then measure
**agreement with SAM**, not true accuracy. Use it only for exploration, and report it as such.

## Run

```bash
pip install -r requirements.txt

# 1) smoke test on synthetic images (a few minutes; numbers are NOT results)
python run.py --synthetic 40 --extractors paper,M5,convnext_t,hybrid --e2e unet,sam,hybrid_e2e --epochs 10 --out results/smoke

# 2a) real data that works out of the box: Zheng et al. WBC sets (images + GT masks, GPL-3)
git clone --depth 1 https://github.com/zxaoyou/segmentation_WBC data/zheng_wbc
python run.py --images "data/zheng_wbc/Dataset 1" --masks "data/zheng_wbc/Dataset 1" --image-ext .bmp \
              --ext-images "data/zheng_wbc/Dataset 2" --ext-masks "data/zheng_wbc/Dataset 2" --ext-image-ext .bmp \
              --epochs 30 --out results/zheng
#     (images are 001.bmp, masks 001.png in the same folder -> --image-ext .bmp tells them apart)

# 2b) your own data (replace the paths with real folders - they must exist)
python run.py --images data/aml/images --masks data/aml/masks \
              --ext-images data/pbc/images --ext-masks data/pbc/masks \
              --localizer sam --stain macenko --epochs 30 --out results/full

# 3) ablations
python ablation.py --images data/aml/images --masks data/aml/masks --epochs 30 --out results/ablation
```

Useful flags: `--sam-model flaviagiammarino/medsam-vit-base` (MedSAM), `--per-class 50` (balanced subset),
`--roi-size 128`, `--pca 128` (puts backbone features at the paper's 128-D; `0` turns it off).

Outputs per run: `results.csv`, `complexity.csv`, `localization.csv`, `tables.md`, `log.txt`,
`fig_accuracy_*.png`, `fig_jaccard_*.png`, `fig_params_vs_f1.png`, `fig_qualitative.png`.

## Deliberate deviations from the paper (state these in your report)

1. **Train/test split by image**, not by pixel. A pixel-level split puts neighbouring pixels of the same cell in both sets.
2. **Evaluation pixels are sampled** (fixed indices, the same for every method) to keep memory bounded. The sample is uniform, so the metrics are unbiased.
3. **SVM is an RBF approximation** (Nystroem + LinearSVC). An exact RBF-SVC is infeasible on millions of pixels.
4. **Compare localisers by ROI coverage, not by pixel accuracy inside the ROI.** A tight ROI that cuts off
   part of the cell (as Otsu often does) contains almost only WBC pixels, which inflates within-ROI accuracy.
   `localization.csv` / `roi_wbc_coverage` report the fraction of the true cell each ROI captured.
5. The paper never trains its CNN, so we use it as published: a seeded random-init feature extractor. Our hybrid trains only its adapters, attention and fusion layers (about 0.5 M parameters); the backbones stay frozen.

## Code map

```
run.py            experiment driver (Phases I-IV + comparisons + complexity + figures)
ablation.py       ablation grid
wbc/data.py       loading, synthetic generator, split
wbc/stain.py      Improvement 2 (Macenko, Reinhard, CLAHE)
wbc/localization.py  Phase I: Otsu (paper) and SAM (ours) CMYK-moment localisers
wbc/features.py   paper CNN, handcrafted M1-M5, timm backbones
wbc/hybrid.py     Improvement 3: ConvNeXt + DINOv2/Swin + paper branches + CBAM
wbc/e2e.py        U-Net, SegFormer
wbc/train.py      shared Dice+BCE trainer with augmentation
wbc/classifiers.py FCL, SVM, RF, XGBoost
wbc/metrics.py    Eq. 8-14 + Dice + ROC-AUC
wbc/complexity.py params / GFLOPs / latency
wbc/plots.py      figures and markdown tables
```

---

### Plain-language explanation

This project re-builds the 2022 paper's method for finding white blood cells in microscope images,
then upgrades it in three ways:
1. It uses the Segment Anything Model instead of a simple brightness threshold to outline the cell.
2. It evens out stain colours between labs before anything else happens.
3. It uses modern pretrained AI models, combined through an attention module, to describe every pixel.

The scripts train and test the original method, our upgraded one, and several well-known modern
models on the same data. They then produce tables and charts comparing accuracy, model size and
speed, plus an ablation study showing how much each upgrade helps on its own.

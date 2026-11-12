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

## Results (Zheng et al. WBC datasets)

All numbers come from real runs; the logs and CSVs are in `results/`.
- **Data:** Dataset 1 (300 images, split 240 train / 60 test by image) is the primary set. Dataset 2 (100 images from a different imaging system) is used for testing only.
- **Settings:** one seed (0), RTX A2000 8 GB, 30 epochs for the trainable models.
- **Scores:** for each method, the best of the four classifiers (FCL / SVM / RF / XGBoost) is shown.
- **Full details:** [RESULTS.md](RESULTS.md) and `results/zheng/tables.md`.

### Main comparison (`results/zheng/`, SAM + Macenko+CLAHE)

| Method | F1 (test) | IoU (test) | F1 (other microscope) | IoU (other microscope) | Params (M) |
|---|---|---|---|---|---|
| **Hybrid ConvNeXt+DINOv2+CBAM (ours)** | **0.9747** | **0.9507** | **0.8956** | **0.8110** | 49.9 (0.49 trained) |
| SegFormer-B0 (end-to-end) | 0.9721 | 0.9456 | 0.7950 | 0.6597 | 3.7 |
| ResNet-50 features | 0.9676 | 0.9371 | 0.7654 | 0.6200 | 23.5 |
| U-Net (end-to-end) | 0.9651 | 0.9326 | 0.5230 | 0.3541 | 7.8 |
| DINOv2 ViT-S features (frozen) | 0.9536 | 0.9114 | 0.8526 | 0.7431 | 21.6 |
| M5 handcrafted | 0.9395 | 0.8859 | 0.5717 | 0.4003 | 0 |
| **Paper CNN fusion (2022)** | 0.9225 | 0.8561 | 0.5556 | 0.3847 | 0.005 |
| SAM ViT-B zero-shot (end-to-end) | 0.6553 | 0.4873 | 0.5274 | 0.3581 | 93.7 |

## Ablation study (`results/ablation/`)

Command: `python ablation.py --images "data/zheng_wbc/Dataset 1" --masks "data/zheng_wbc/Dataset 1" --image-ext .bmp --epochs 30 --out results/ablation`.
This run used the **primary test set only** (no external set).

### A. Localizer × stain normalization (paper CNN vs our hybrid)

*Coverage* is the fraction of the true WBC that falls inside the ROI.

| Localizer | Preprocessing | WBC coverage | Paper CNN acc / F1 | Hybrid acc / F1 |
|---|---|---|---|---|
| Otsu (paper) | none | 0.655 | 0.9556 / 0.9696 | 0.9782 / 0.9850 |
| Otsu (paper) | Reinhard+CLAHE | **0.469** | 0.9551 / 0.9721 | 0.9688 / 0.9806 |
| Otsu (paper) | Macenko+CLAHE | 0.918 | 0.9366 / 0.9385 | 0.9793 / 0.9797 |
| SAM (ours) | none | 0.923 | 0.9590 / 0.9450 | 0.9803 / 0.9735 |
| SAM (ours) | Reinhard+CLAHE | 0.918 | 0.9544 / 0.9430 | 0.9801 / 0.9751 |
| **SAM (ours)** | **Macenko+CLAHE** | **0.957** | 0.9391 / 0.9225 | 0.9798 / 0.9738 |

**What this shows**
- **SAM + Macenko+CLAHE captures the most of each cell (95.7%).** Otsu alone loses about a third of the cell, and Otsu + Reinhard loses more than half.
- **Read across rows with care: accuracy is measured inside each localizer's own ROI.** A crop that cuts the cell off is mostly WBC pixels, which makes it an easier task. For example, Otsu + none scores F1 0.985 while covering only 65.5% of the cell. Compare localizers by coverage, and compare feature extractors within a row.
- **Within every row, the hybrid beats the paper's CNN**, by 1.4 to 4.3 accuracy points.
- **Stain normalization did not raise segmentation accuracy on this single-lab dataset.** The hybrid stays at about 0.980. The paper CNN drops with Macenko+CLAHE (0.959 → 0.939 under SAM), probably because its untrained random filters depend on raw colour.
- **Where normalization did help is localization.** It raised Otsu's coverage from 0.655 to 0.918, and SAM's centroid-inside-WBC rate from 0.75 to 0.98 on the test set (`results/ablation/*/log.txt`).
- **The cross-lab benefit is untested here**, because this ablation had no external set.

### B. Which parts of the hybrid matter (SAM + Macenko+CLAHE, `results/ablation/extractor_variants/`)

| Variant | Acc | F1 | IoU | Total / trained params (M) | Latency (ms/ROI) | Train time (s) |
|---|---|---|---|---|---|---|
| **ViT-only (DINOv2 + adapters)** | **0.9823** | **0.9770** | **0.9551** | 21.9 / 0.28 | 15.0 | 106 |
| Full hybrid (ConvNeXt + DINOv2 + CBAM) | 0.9806 | 0.9748 | 0.9508 | 49.9 / 0.49 | 25.8 | 202 |
| Without the paper's branches | 0.9804 | 0.9746 | 0.9504 | 49.9 / 0.41 | 24.3 | 167 |
| Swin-T instead of DINOv2 | 0.9803 | 0.9745 | 0.9503 | 55.8 / 0.49 | 28.0 | 212 |
| CNN-only (ConvNeXt + adapters) | 0.9792 | 0.9730 | 0.9474 | 28.1 / 0.28 | 13.1 | 315 |
| Without CBAM attention | 0.9789 | 0.9727 | 0.9468 | 49.9 / 0.47 | 21.2 | 808* |
| Paper CNN fusion (2022) | 0.9391 | 0.9225 | 0.8561 | 0.005 / 0 | 1.3 | 0 |

\* The 808 s is likely a timing outlier (GPU contention), not a property of the model.

**What this shows**
- **Every variant of our extractor beats the paper's CNN by about 4 accuracy points** (F1 +5 points). This is the one large, robust result.
- **The ViT-only variant scored best and is the cheapest to train.** Adding ConvNeXt, CBAM attention or the paper's branches did not improve results on this dataset. Attention adds about 0.2 points, and the paper's branches add nothing measurable.
- **The differences between our variants are small (at most 0.3 accuracy points)** on 60 test images with a single seed. Treat them as roughly equal rather than as a ranking.
- **The choice of classifier barely matters.** FCL, SVM, RF and XGBoost agree within about 0.1 points for every extractor, so the features drive performance.
- **Open question:** the full hybrid's large cross-microscope win (external F1 0.896 vs 0.556) has not yet been tested for the ViT-only variant. To check it, rerun with
  `python ablation.py --part extractor --images "data/zheng_wbc/Dataset 1" --masks "data/zheng_wbc/Dataset 1" --image-ext .bmp --ext-images "data/zheng_wbc/Dataset 2" --ext-masks "data/zheng_wbc/Dataset 2" --ext-image-ext .bmp --epochs 30 --out results/ablation_ext`.

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

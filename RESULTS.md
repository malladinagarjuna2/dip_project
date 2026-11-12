# Results: WBC segmentation (Zheng et al. WBC datasets)

**Setup**
- **Primary set:** Zheng Dataset 1. 300 images (112–120 px), real ground-truth masks, split 240 train / 60 test by image.
- **External set:** Zheng Dataset 2. 100 images (300 × 300 px) from a different imaging system, used for testing only. It plays the role of the paper's secondary dataset.
- **Pipeline:** Macenko + CLAHE, then SAM-based CMYK-moment ROI (128 × 128), then the features or model, then FCL / SVM / RF / XGBoost.
- **Hardware:** RTX A2000 8 GB, 30 epochs for the trainable models.
- **Reproduce:** `python run.py --images "data/zheng_wbc/Dataset 1" ... --out results/zheng` (see README). Full per-classifier tables are in `results/zheng/tables.md`.

> These are **not** the AML_Cytomorphology_LMU images used in the paper, because that dataset has no public masks.
> The comparison is between methods on identical data; the numbers are not directly comparable to the paper's tables.

## 1. Phase I: Otsu (paper) vs SAM (ours), all 400 images

`coverage` is the fraction of the true WBC inside the ROI. `full cell` is the share of images whose ROI contains at least 99% of the WBC.

| Dataset | Preprocessing | Localiser | Coverage | Full cell | Centroid inside WBC |
|---|---|---|---|---|---|
| D1 | none | Otsu | 0.650 | 26% | 0.983 |
| D1 | none | **SAM** | **0.922** | **85%** | 0.800 |
| D1 | Macenko+CLAHE | Otsu | 0.791 | 45% | 0.997 |
| D1 | Macenko+CLAHE | **SAM** | **0.939** | **85%** | 0.990 |
| D2 | none | Otsu | 0.839 | 37% | 1.000 |
| D2 | none | **SAM** | **0.983** | **92%** | 0.990 |
| D2 | Macenko+CLAHE | Otsu | 0.847 | 54% | 0.900 |
| D2 | Macenko+CLAHE | **SAM** | **0.958** | **89%** | 0.890 |

- **SAM vs Otsu:** SAM captures the whole cell 2–3× more often than Otsu. Otsu only finds the nucleus, and the paper's fixed margin cuts off the cytoplasm of large cells.
- **Improvement 2 helps both localisers.** Otsu's coverage on D1 rises from 0.65 to 0.79. SAM's centroid lands inside the WBC 99% of the time instead of 80%.

## 2. Feature extractors and recent models: best over the four classifiers

| Method | Acc (primary) | F1 (primary) | IoU (primary) | Acc (external) | F1 (external) | IoU (external) |
|---|---|---|---|---|---|---|
| **Hybrid ConvNeXt+DINOv2+CBAM (ours)** | **0.9805** | **0.9747** | **0.9507** | **0.9298** | **0.8956** | **0.8110** |
| Hybrid head, end-to-end (ours) | 0.9799 | 0.9740 | 0.9493 | 0.9280 | 0.8937 | 0.8078 |
| SegFormer-B0 (e2e) | 0.9783 | 0.9721 | 0.9456 | 0.8808 | 0.7950 | 0.6597 |
| ResNet-50 features | 0.9750 | 0.9676 | 0.9371 | 0.8240 | 0.7654 | 0.6200 |
| Swin-T features | 0.9740 | 0.9664 | 0.9349 | 0.7831 | 0.7211 | 0.5639 |
| U-Net (e2e) | 0.9728 | 0.9651 | 0.9326 | 0.4085 | 0.5230 | 0.3541 |
| ConvNeXt-T features | 0.9654 | 0.9548 | 0.9135 | 0.9018 | 0.8478 | 0.7358 |
| DINOv2 ViT-S features | 0.9646 | 0.9536 | 0.9114 | 0.9076 | 0.8526 | 0.7431 |
| M5 handcrafted (+Gaussian) | 0.9526 | 0.9395 | 0.8859 | 0.5190 | 0.5717 | 0.4003 |
| M4 handcrafted (+K-means) | 0.9470 | 0.9323 | 0.8732 | 0.5860 | 0.6032 | 0.4318 |
| M1 handcrafted (Gabor) | 0.9424 | 0.9267 | 0.8634 | 0.4225 | 0.5277 | 0.3585 |
| **Paper CNN fusion (2022)** | 0.9391 | 0.9225 | 0.8561 | 0.4920 | 0.5556 | 0.3847 |
| SAM ViT-B zero-shot (e2e) | 0.7270 | 0.6553 | 0.4873 | 0.6552 | 0.5274 | 0.3581 |

**Key findings**
1. **On the primary test set**, Improvement 3 beats the paper's features: F1 0.923 → 0.975 and IoU 0.856 → 0.951. It also beats every *frozen* single backbone. However, the ablation (README, Ablation B) shows that a ViT-only variant with the same trained adapters does as well (F1 0.977 vs 0.975). The gain comes mainly from the pretrained DINOv2 features plus trained adapters, not from fusing CNN and Transformer.
2. **The biggest gain is generalisation.** On the unseen external scanner, the paper's method collapses (F1 0.556), and so does U-Net, trained from scratch (F1 0.523). The hybrid holds F1 0.896 and IoU 0.811, the best by a wide margin. The paper claimed its method generalises; on a real domain shift it does not, while the pretrained-backbone hybrid does.
3. **The paper's ordering holds:** handcrafted M1 < M5, and each added filter helps a little. Our measured paper CNN lands slightly *below* M5, unlike the paper's Table 3. That's expected, since its weights are random and never trained, as the paper describes.
4. **SAM zero-shot whole-cell masks are weak** (F1 0.66). SAM's nucleus masks were always fully inside the cell, but the faint cytoplasm is either missed or leaks into touching red cells. So SAM is used here for **localisation**, where it clearly wins (Section 1), not as the final segmenter. Fine-tuned medical SAM variants (MedSAM) are the natural next step.

## 3. Parameters and complexity (per 128 × 128 ROI)

| Method | Params (M) | Trained params (M) | GFLOPs | Latency (ms) | Training time (s) |
|---|---|---|---|---|---|
| Paper CNN fusion | 0.005 | 0 | 0.17 | 1.2 | 0 |
| Handcrafted M1–M5 | 0 | 0 | ~0 | 9–27 (CPU) | 0 |
| ResNet-50 | 23.5 | 0 | 8.2 | 10.7 | 0 |
| ConvNeXt-T | 27.8 | 0 | 8.9 | 12.6 | 0 |
| Swin-T | 27.5 | 0 | 9.0 | 11.9 | 0 |
| DINOv2 ViT-S | 21.6 | 0 | 12.2 | 11.7 | 0 |
| **Hybrid (ours)** | 49.9 | **0.49** | 33.4 | 27.4 | 194 |
| U-Net | 7.8 | 7.8 | 6.0 | 2.9 | 123 |
| SegFormer-B0 | 3.7 | 3.7 | 0.85 | 1.7 | 26 |
| SAM ViT-B (1024² input) | 93.7 | 0 | 972 | 970 | 0 |

**Trade-off**
- **Hybrid:** the most accurate and most robust model, while training only 0.49 M parameters (1% of the network). The cost is about 27 ms per cell, roughly 16× slower than SegFormer-B0.
- **SegFormer-B0:** if speed matters more than robustness to new scanners, it is the best value.
- **SAM:** by far the most expensive step (about 1 s per image). It runs once per image for localisation only.

## Not yet run
- `ablation.py` on real data (hybrid without attention / CNN-only / ViT-only / without the paper's branches / Swin instead of DINOv2; Reinhard vs Macenko). It runs about seven full experiments, around 2–3 h on this GPU.
- Results on AML_Cytomorphology_LMU, which needs masks first.
- Repeated runs with different seeds for error bars. All numbers above come from one seed (0).

---

### Plain-language explanation
We tested every method on 300 real blood-cell pictures that come with correct outlines, then on 100
pictures from a different microscope that no model had seen.
- **Localisation:** using SAM to find the cell kept the whole cell inside the crop in most cases. The paper's method often cut part of the cell off.
- **Accuracy:** our upgraded feature extractor was the most accurate.
- **New microscope:** the paper's method and a plain U-Net failed badly on the unseen pictures, while ours stayed accurate.
- **Cost:** ours is bigger and slower than the smallest modern model (SegFormer), but we only had to train about 1% of it.

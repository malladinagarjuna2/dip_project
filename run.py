"""End-to-end experiment driver.

Example (real data, full comparison):
    python run.py --images data/train/images --masks data/train/masks \
                  --ext-images data/external/images --ext-masks data/external/masks \
                  --localizer sam --stain macenko --out results/full

Smoke test (synthetic images, a few minutes):
    python run.py --synthetic 60 --extractors paper,M5,convnext_t,hybrid --e2e unet,sam --out results/smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from wbc import data as D
from wbc.classifiers import CLASSIFIER_NAMES, build_classifier
from wbc.complexity import profile
from wbc.e2e import build_e2e
from wbc.features import build_extractor
from wbc.localization import build_localizer, crop_resize
from wbc.metrics import per_image_dice, pixel_metrics
from wbc.stain import Preprocessor, pick_reference
from wbc.train import fit_segmenter

DISPLAY = {
    "paper": "Paper CNN fusion (2022)", "M1": "M1 Gabor", "M2": "M2 +LBP", "M3": "M3 +Edges",
    "M4": "M4 +K-means", "M5": "M5 +Gaussian", "resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0",
    "convnext_t": "ConvNeXt-T", "swin_t": "Swin-T", "dinov2_s": "DINOv2 ViT-S/14", "vit_b": "ViT-B/16",
    "hybrid": "Hybrid ConvNeXt+DINOv2+CBAM (ours)", "hybrid_swin": "Hybrid ConvNeXt+Swin+CBAM (ours)",
    "hybrid_noattn": "Hybrid w/o attention", "hybrid_cnnonly": "Hybrid CNN-only", "hybrid_vitonly": "Hybrid ViT-only",
    "hybrid_nolocal": "Hybrid w/o paper branches",
    "unet": "U-Net (e2e)", "segformer_b0": "SegFormer-B0 (e2e)", "segformer_b2": "SegFormer-B2 (e2e)",
    "sam": "SAM ViT-B zero-shot (e2e)", "hybrid_e2e": "Hybrid head (e2e, ours)",
}
E2E_KEYS = {"unet", "segformer_b0", "segformer_b2", "sam", "hybrid_e2e"}


def family(m: str) -> str:
    if m == "paper":
        return "paper"
    if m[0] == "M" and m[1:].isdigit():
        return "handcrafted"
    if m.startswith("hybrid"):
        return "ours"
    return "e2e" if m in E2E_KEYS else "recent backbone"

HYBRID_VARIANTS = {
    "hybrid": dict(cnn="convnext_t", vit="dinov2_s"),
    "hybrid_swin": dict(cnn="convnext_t", vit="swin_t"),
    "hybrid_noattn": dict(cnn="convnext_t", vit="dinov2_s", attention=False),
    "hybrid_cnnonly": dict(cnn="convnext_t", vit=None),
    "hybrid_vitonly": dict(cnn=None, vit="dinov2_s"),
    "hybrid_nolocal": dict(cnn="convnext_t", vit="dinov2_s", use_local=False),
}


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--images"); a.add_argument("--masks")
    a.add_argument("--ext-images"); a.add_argument("--ext-masks")
    a.add_argument("--image-ext", help="only files with this extension are images (when masks share the folder)")
    a.add_argument("--ext-image-ext", help="same, for the external set")
    a.add_argument("--synthetic", type=int, default=0, help="use N synthetic images (code smoke test only)")
    a.add_argument("--pseudo-gt", action="store_true",
                   help="no masks available: use SAM whole-cell masks as labels (metrics then = agreement with SAM)")
    a.add_argument("--limit", type=int); a.add_argument("--per-class", type=int)
    a.add_argument("--max-side", type=int, default=400)
    a.add_argument("--roi-size", type=int, default=128)
    a.add_argument("--localizer", default="sam", choices=["sam", "otsu"])
    a.add_argument("--sam-model", default="facebook/sam-vit-base")
    a.add_argument("--stain", default="macenko", choices=["macenko", "reinhard", "none"])
    a.add_argument("--no-clahe", action="store_true")
    a.add_argument("--extractors", default="paper,M1,M2,M3,M4,M5,resnet50,convnext_t,swin_t,dinov2_s,hybrid")
    a.add_argument("--classifiers", default="fcl,svm,rf,xgb")
    a.add_argument("--e2e", default="unet,segformer_b0,sam,hybrid_e2e")
    a.add_argument("--pca", type=int, default=128, help="reduce >D-dim backbone features to D (0 = off)")
    a.add_argument("--train-px", type=int, default=512, help="sampled pixels per training ROI")
    a.add_argument("--test-px", type=int, default=2048, help="sampled pixels per test ROI (fixed for all methods)")
    a.add_argument("--epochs", type=int, default=20)
    a.add_argument("--n-vis", type=int, default=8)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="results/run")
    return a.parse_args()


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *msg):
        s = " ".join(str(m) for m in msg)
        print(s, flush=True); self.f.write(s + "\n"); self.f.flush()


# ------------------------------------------------------------------------------------------- phase I
def localize_all(samples, pre, loc, S, pseudo_gt, log, tag):
    X, Y, CELL, rows = [], [], [], []
    for s in samples:
        img = pre(s.image)
        L = loc(img)
        cell = L.cell
        mask = s.mask
        if mask is None:
            if not pseudo_gt or cell is None:
                raise RuntimeError(f"{s.name}: no ground-truth mask (pass --masks or --pseudo-gt with --localizer sam)")
            mask = cell
        x0, y0, x1, y1 = L.box
        cov = mask[y0:y1, x0:x1].sum() / max(mask.sum(), 1)
        cx, cy = int(L.centroid[0]), int(L.centroid[1])
        hit = bool(mask[min(cy, mask.shape[0] - 1), min(cx, mask.shape[1] - 1)])
        rows.append(dict(set=tag, name=s.name, cls=s.label, coverage=cov, centroid_in_wbc=hit,
                         roi_side=x1 - x0, seconds=L.seconds))
        X.append(crop_resize(img, L.box, S))
        Y.append(crop_resize(mask, L.box, S, nearest=True))
        CELL.append(crop_resize(cell, L.box, S, nearest=True) if cell is not None else np.zeros((S, S), np.uint8))
    df = pd.DataFrame(rows)
    log(f"  [{tag}] {len(df)} ROIs | WBC coverage {df.coverage.mean():.4f} | centroid-in-WBC "
        f"{df.centroid_in_wbc.mean():.3f} | {df.seconds.mean() * 1000:.1f} ms/img")
    to_t = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).float() / 255
    return to_t(X), torch.from_numpy(np.stack(Y))[:, None].float(), np.stack(CELL), df


# ------------------------------------------------------------------------------------------- features
@torch.no_grad()
def extract(fn, X, idx, device, bs=8):
    """fn: (B,3,S,S)->(B,D,S,S). idx: (N,k) flat pixel indices to keep per ROI. -> (N*k, D) float32."""
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            f = fn(x)
        f = f.float().flatten(2)                                       # B,D,S*S
        ii = torch.from_numpy(idx[i:i + bs]).to(f.device)              # B,k
        f = torch.gather(f, 2, ii[:, None, :].expand(-1, f.shape[1], -1))  # B,D,k
        out.append(f.permute(0, 2, 1).reshape(-1, f.shape[1]).cpu().numpy())
    return np.concatenate(out)


def sample_idx(n, S, k, rng):
    return np.stack([rng.choice(S * S, size=min(k, S * S), replace=False) for _ in range(n)])


def labels_at(Y, idx):
    flat = Y.flatten(1).numpy()
    return np.take_along_axis(flat, idx, 1).reshape(-1).astype(np.uint8)


# ------------------------------------------------------------------------------------------- main
def main(args=None):
    args = args or parse()
    os.makedirs(args.out, exist_ok=True)
    log = Log(os.path.join(args.out, "log.txt"))
    log(f"\n==== run {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n{json.dumps(vars(args), indent=1)}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = args.roi_size

    # ---------- data
    if args.synthetic:
        log("!! SYNTHETIC DATA - for code verification only, numbers are meaningless for the paper")
        samples = D.make_synthetic(args.synthetic, seed=args.seed)
        ext = D.make_synthetic(max(10, args.synthetic // 4), seed=args.seed + 99)
    else:
        samples = D.load_folder(args.images, args.masks, args.limit, args.max_side, args.per_class, args.seed,
                                args.image_ext)
        ext = D.load_folder(args.ext_images, args.ext_masks, args.limit, args.max_side, seed=args.seed,
                            image_ext=args.ext_image_ext) \
            if args.ext_images else []
    train, test = D.split(samples, 0.2, args.seed)
    log(f"data: train {len(train)}  test {len(test)}  external {len(ext)}")

    # ---------- improvement 2: stain normalisation + CLAHE (reference picked from TRAIN only)
    pre = Preprocessor(args.stain, not args.no_clahe).fit(pick_reference([s.image for s in train]))
    log(f"preprocessing: {pre}")

    # ---------- phase I: localisation (Otsu = paper, SAM = ours)
    loc = build_localizer(args.localizer, **({"model_id": args.sam_model} if args.localizer == "sam" else {}))
    log(f"phase I localiser: {args.localizer}")
    Xtr, Ytr, _, ltr = localize_all(train, pre, loc, S, args.pseudo_gt, log, "train")
    Xte, Yte, CELLte, lte = localize_all(test, pre, loc, S, args.pseudo_gt, log, "test")
    sets = {"primary": (Xte, Yte, CELLte)}
    loc_dfs = [ltr, lte]
    if ext:
        Xex, Yex, CELLex, lex = localize_all(ext, pre, loc, S, args.pseudo_gt, log, "external")
        sets["external"] = (Xex, Yex, CELLex)
        loc_dfs.append(lex)
    pd.concat(loc_dfs).assign(localizer=args.localizer, stain=str(pre)).to_csv(
        os.path.join(args.out, "localization.csv"), index=False)
    sam_profile = None
    if args.localizer == "sam":
        tot = sum(p.numel() for p in loc.model.parameters())
        from wbc.complexity import gflops
        sam_gf = gflops(lambda x: loc.model.get_image_embeddings(x), torch.rand(1, 3, 1024, 1024, device=loc.device))
        sam_profile = dict(params_M=tot / 1e6, trainable_params_M=0.0, gflops_per_roi=sam_gf,
                           latency_ms_per_roi=lte.seconds.mean() * 1000)
        loc.model.to("cpu"); torch.cuda.empty_cache()

    idx_tr = sample_idx(len(Xtr), S, args.train_px, rng)
    ytr = labels_at(Ytr, idx_tr)
    idx_eval = {k: sample_idx(len(v[0]), S, args.test_px, rng) for k, v in sets.items()}
    y_eval = {k: labels_at(sets[k][1], idx_eval[k]) for k in sets}
    log(f"train pixels {len(ytr)} (WBC {ytr.mean():.1%}) | eval pixels " +
        ", ".join(f"{k} {len(v)}" for k, v in y_eval.items()))

    results, cplx, vis = [], [], {"image": Xte[:args.n_vis].permute(0, 2, 3, 1).numpy(),
                                  "gt": Yte[:args.n_vis, 0].numpy(), "pred": {}}

    def record(method, clf, split, prob, y, extra):
        m = pixel_metrics(y, prob)
        results.append(dict(method=method, display=DISPLAY.get(method, method),
                            family=family(method),
                            classifier=clf, dataset=split, **m, **extra))
        log(f"    {DISPLAY.get(method, method):40s} {clf:8s} {split:9s} acc {m['accuracy']:.4f}  "
            f"F1 {m['f1']:.4f}  IoU {m['jaccard']:.4f}  AUC {m['auc']:.4f}")

    # ---------- phase III + IV: feature extractors x classifiers
    hybrid_e2e_model = None
    for key in [k for k in args.extractors.split(",") if k]:
        log(f"\n[extractor] {DISPLAY.get(key, key)}")
        t0 = time.perf_counter()
        model = build_extractor("hybrid", **HYBRID_VARIANTS[key]) if key in HYBRID_VARIANTS else build_extractor(key)
        model = model.to(dev).eval()
        train_s = 0.0
        if getattr(model, "trainable", False):
            train_s = fit_segmenter(model, Xtr, Ytr, epochs=args.epochs, device=dev, log=log)
            if key == "hybrid":
                hybrid_e2e_model = model
        fn = model.features if hasattr(model, "features") else model
        Ftr = extract(fn, Xtr, idx_tr, dev)
        Fev = {k: extract(fn, sets[k][0], idx_eval[k], dev) for k in sets}
        raw_dim = Ftr.shape[1]
        reducer = None
        if args.pca and raw_dim > args.pca:
            reducer = PCA(args.pca, random_state=args.seed).fit(Ftr[:200000])
            Ftr = reducer.transform(Ftr); Fev = {k: reducer.transform(v) for k, v in Fev.items()}
        feat_s = time.perf_counter() - t0
        log(f"  features: raw {raw_dim}-D -> {Ftr.shape[1]}-D | extraction+training {feat_s:.1f}s")
        prof = profile(model, fn, S, dev)
        if not getattr(model, "trainable", False):
            prof["trainable_params_M"] = 0.0  # frozen / untrained feature extractor
        cplx.append(dict(method=key, display=DISPLAY.get(key, key), raw_dim=raw_dim, feat_dim=Ftr.shape[1],
                         train_s=train_s, **prof))

        vis_feats = None
        if args.n_vis:
            full_idx = np.tile(np.arange(S * S), (min(args.n_vis, len(Xte)), 1))
            vis_feats = extract(fn, Xte[:args.n_vis], full_idx, dev)
            if reducer is not None:
                vis_feats = reducer.transform(vis_feats)
        for ck in [c for c in args.classifiers.split(",") if c]:
            t = time.perf_counter()
            clf = build_classifier(ck, args.seed).fit(Ftr, ytr)
            fit_s = time.perf_counter() - t
            for split, F in Fev.items():
                t = time.perf_counter()
                prob = clf.predict_proba(F)[:, 1]
                record(key, CLASSIFIER_NAMES[ck], split, prob, y_eval[split],
                       dict(n_features=Ftr.shape[1], clf_fit_s=fit_s, predict_s=time.perf_counter() - t,
                            feature_s=feat_s))
            if vis_feats is not None:
                vis["pred"][(key, CLASSIFIER_NAMES[ck])] = (clf.predict_proba(vis_feats)[:, 1] >= 0.5).reshape(-1, S, S)
        if model is not hybrid_e2e_model:
            del model
        gc.collect(); torch.cuda.empty_cache()

    # ---------- recent end-to-end segmentation models
    for key in [k for k in args.e2e.split(",") if k]:
        log(f"\n[end-to-end] {DISPLAY.get(key, key)}")
        if key == "sam":
            if args.localizer != "sam":
                log("  skipped (needs --localizer sam)"); continue
            if args.pseudo_gt:
                log("  skipped (SAM masks are the pseudo ground truth)"); continue
            for split, (_, Y, CELL) in sets.items():
                prob = np.take_along_axis(CELL.reshape(len(CELL), -1), idx_eval[split], 1).reshape(-1).astype(float)
                record("sam", "-", split, prob, y_eval[split],
                       dict(dice_img_mean=per_image_dice(Y[:, 0].numpy(), CELL).mean()))
            vis["pred"][("sam", "-")] = CELLte[:args.n_vis].astype(bool)
            cplx.append(dict(method="sam", display=DISPLAY["sam"], raw_dim=np.nan, feat_dim=np.nan, train_s=0.0,
                             **sam_profile))
            continue
        if key == "hybrid_e2e":
            if hybrid_e2e_model is None:
                log("  skipped (run 'hybrid' extractor first)"); continue
            model, train_s = hybrid_e2e_model, 0.0
        else:
            model = build_e2e(key).to(dev)
            train_s = fit_segmenter(model, Xtr, Ytr, epochs=args.epochs, device=dev, log=log)
        model.eval()
        fn = lambda x, m=model: torch.sigmoid(m(x))
        for split, (X, Y, _) in sets.items():
            t = time.perf_counter()
            prob = extract(fn, X, idx_eval[split], dev)[:, 0]
            full = extract(fn, X, np.tile(np.arange(S * S), (len(X), 1)), dev)[:, 0].reshape(-1, S, S)
            record(key, "-", split, prob, y_eval[split],
                   dict(dice_img_mean=per_image_dice(Y[:, 0].numpy(), full >= 0.5).mean(),
                        predict_s=time.perf_counter() - t))
        vis["pred"][(key, "-")] = extract(fn, Xte[:args.n_vis], np.tile(np.arange(S * S), (min(args.n_vis, len(Xte)), 1)),
                                          dev)[:, 0].reshape(-1, S, S) >= 0.5
        cplx.append(dict(method=key, display=DISPLAY.get(key, key), raw_dim=np.nan, feat_dim=np.nan, train_s=train_s,
                         **profile(model, fn, S, dev)))
        del model; gc.collect(); torch.cuda.empty_cache()

    # ---------- save
    res = pd.DataFrame(results); res.to_csv(os.path.join(args.out, "results.csv"), index=False)
    cp = pd.DataFrame(cplx); cp.to_csv(os.path.join(args.out, "complexity.csv"), index=False)
    torch.save(vis, os.path.join(args.out, "vis.pt"))
    from wbc.plots import make_all
    make_all(args.out)
    summary = res[res.dataset == "primary"].pivot_table(index="display", columns="classifier", values="accuracy")
    log("\nPrimary-set overall accuracy:\n" + summary.round(4).to_string())
    log(f"\nsaved to {args.out}")
    return res, cp


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# This is the "start button" that runs the whole study in the same order as the paper's figure:
#  1. Load the blood-smear images (and their true WBC masks) and split them 80% train / 20% test.
#  2. Improvement 2: even out stain colours (Macenko) and boost contrast (CLAHE).
#  3. Phase I: find each white cell and cut a square ROI around it - with SAM (ours) or Otsu (paper).
#     It also logs how much of the real cell each ROI captured, so the two localisers can be compared.
#  4. Phase III: turn every ROI pixel into features using each extractor in turn: the paper's CNN,
#     the handcrafted M1-M5 banks, modern pretrained backbones, and our hybrid attention model
#     (Improvement 3, which is trained first). Huge feature lists are shrunk to 128 with PCA so all
#     methods are compared at the same size as the paper.
#  5. Phase IV: train FCL, SVM, RF and XGBoost on those features and score them on the test set and,
#     if given, on an external dataset from another lab (generalisation test).
#  6. Also train/run modern end-to-end segmenters (U-Net, SegFormer, SAM zero-shot, our hybrid head).
#  7. Record size/speed of every model and save CSV tables, a log, and comparison figures.

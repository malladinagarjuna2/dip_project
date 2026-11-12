"""Phase IV classifiers used by the paper: FCL (fully connected layers), SVM, RF, XGBoost."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class FCL:
    """Keras-style dense head: D -> 128 -> 64 -> 1 with dropout, trained with Adam on GPU."""

    def __init__(self, epochs: int = 20, bs: int = 4096, lr: float = 1e-3, device: str | None = None):
        self.epochs, self.bs, self.lr = epochs, bs, lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        X = torch.tensor(self.scaler.transform(X), dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        d = X.shape[1]
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 1)).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), device=self.device)
            for i in range(0, len(X), self.bs):
                idx = perm[i:i + self.bs]
                loss = lossf(self.net(X[idx]).squeeze(1), y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        X = torch.tensor(self.scaler.transform(X), dtype=torch.float32, device=self.device)
        out = [torch.sigmoid(self.net(X[i:i + 65536])).squeeze(1).cpu() for i in range(0, len(X), 65536)]
        p = torch.cat(out).numpy()
        return np.stack([1 - p, p], 1)


class SVM:
    """RBF-kernel SVM approximated with a Nystroem feature map + linear SVM (exact RBF-SVC is O(n^2)
    and infeasible on millions of pixels). Scores are squashed to [0,1] for AUC."""

    def __init__(self, n_components: int = 500, max_train: int = 60000, seed: int = 0):
        self.max_train, self.seed = max_train, seed
        self.model = make_pipeline(StandardScaler(), Nystroem(gamma=None, n_components=n_components, random_state=seed),
                                   LinearSVC(C=1.0, dual=False, max_iter=2000))

    def fit(self, X, y):
        if len(X) > self.max_train:
            idx = np.random.default_rng(self.seed).choice(len(X), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        s = self.model.decision_function(X)
        p = 1 / (1 + np.exp(-2 * s))
        return np.stack([1 - p, p], 1)


def build_classifier(key: str, seed: int = 0):
    if key == "fcl":
        return FCL()
    if key == "svm":
        return SVM(seed=seed)
    if key == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if key == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist",
                             device="cuda" if torch.cuda.is_available() else "cpu", random_state=seed)
    raise ValueError(key)


CLASSIFIER_NAMES = {"fcl": "FCL", "svm": "SVM", "rf": "RF", "xgb": "XGBoost"}


# ---------------------------------------------------------------------------------------------
# Plain-language explanation
# ---------------------------------------------------------------------------------------------
# After every pixel has been described by a list of numbers (its features), a classifier decides
# "white blood cell" or "background" for it. The paper tries four classic decision-makers and so do
# we, so the comparison is fair:
#  * FCL - a small neural network made of fully connected layers (runs on the GPU).
#  * SVM - a support vector machine; we use a fast approximation of the usual curved (RBF) version
#    because the exact one cannot handle millions of pixels.
#  * RF  - a random forest: many decision trees voting together.
#  * XGBoost - boosted decision trees, each tree fixing the previous ones' mistakes.
# Each one outputs a probability that a pixel is part of the WBC.

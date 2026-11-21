"""P5: the MABAC border region G with a split-conformal coverage guarantee.

Prediction sets  Γ(x) = {c : 1 - P(c|x) <= q̂},  q̂ = ⌈(n+1)(1-α)⌉/n empirical quantile of calibration scores.
Guarantee (exchangeability): P(y ∈ Γ(x)) >= 1 - α  (marginal, finite-sample).

Binary reading in MABAC terms:
    Γ = {1}     -> upper approximation area G+   (decide positive)
    Γ = {0}     -> lower approximation area G-   (decide negative)
    Γ = {0, 1}  -> border approximation area G   (defer to an expert)
    Γ = ∅       -> outside both areas            (atypical input; defer)
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split


class ConformalBorder:
    def __init__(self, estimator, alpha=0.1, cal_size=0.25, random_state=0):
        self.estimator, self.alpha, self.cal_size, self.random_state = estimator, alpha, cal_size, random_state

    def fit(self, X, y):
        Xtr, Xc, ytr, yc = train_test_split(X, y, test_size=self.cal_size, stratify=y, random_state=self.random_state)
        self.model_ = clone(self.estimator).fit(Xtr, ytr)
        self.classes_ = np.asarray(self.model_.classes_)
        P = self.model_.predict_proba(Xc)
        scores = 1 - P[np.arange(len(yc)), np.searchsorted(self.classes_, yc)]
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat_ = np.quantile(scores, level, method="higher")
        return self

    def predict_set(self, X):
        P = self.model_.predict_proba(X)
        return (1 - P) <= self.qhat_  # boolean (m, C)

    def predict_region(self, X):
        """Returns array of 'G+', 'G-', 'G' or 'out' (binary) / class label or 'G' / 'out' (multi-class)."""
        S = self.predict_set(X)
        k = S.sum(1)
        out = np.empty(len(X), dtype=object)
        out[k == 0] = "out"
        out[k > 1] = "G"
        single = k == 1
        lab = self.classes_[np.argmax(S, 1)]
        if len(self.classes_) == 2:
            out[single] = np.where(lab[single] == self.classes_[1], "G+", "G-")
        else:
            out[single] = lab[single]
        return out

    def evaluate(self, X, y):
        S = self.predict_set(X)
        yi = np.searchsorted(self.classes_, y)
        k = S.sum(1)
        decided = k == 1
        pred = np.argmax(S, 1)
        return dict(alpha=self.alpha, coverage=float(S[np.arange(len(y)), yi].mean()),
                    decided_rate=float(decided.mean()), border_rate=float((k > 1).mean()),
                    out_rate=float((k == 0).mean()),
                    acc_decided=float((pred[decided] == yi[decided]).mean()) if decided.any() else np.nan,
                    mean_set_size=float(k.mean()))

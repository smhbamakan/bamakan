"""P3 real-data benchmark of the BAMAKAN global branch.

Protocol: repeated stratified K-fold (default 5 folds x 2 repeats), identical folds for all models,
default hyper-parameters for every model (no per-model tuning in this pilot), standard scaling where needed.
Usage: python benchmark.py <dataset> [<dataset> ...]
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, os, sys, time, warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss, mean_absolute_error,
                             roc_auc_score, brier_score_loss)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pytabkit import RealMLP_TD_Classifier

sys.path.insert(0, f"{ROOT}")
sys.path.insert(0, f"{ROOT}/experiments")
from bamakan import BAMAKANClassifier
from bamakan.gmlvq import GMLVQ
from datasets import DATASETS

warnings.filterwarnings("ignore")
OUT = f"{ROOT}/results/bench"
ONLY = [m for m in os.environ.get("BENCH_MODELS", "").split(",") if m]
SUFFIX = os.environ.get("BENCH_SUFFIX", "")


class NC:
    def fit(self, X, y):
        self.p = make_pipeline(StandardScaler(), NearestCentroid()).fit(X, y)
        self.classes_ = self.p.classes_
        return self

    def predict(self, X):
        return self.p.predict(X)

    def predict_proba(self, X):
        nc = self.p[-1]
        Z = self.p[0].transform(X)
        D = ((Z[:, None, :] - nc.centroids_[None]) ** 2).sum(-1)
        e = np.exp(-(D - D.min(1, keepdims=True)) / 2)
        return e / e.sum(1, keepdims=True)


class MonoXGB:
    """XGBoost with monotone constraints. Directions = sign of the training Spearman correlation of each feature
    with the (ordered) target. Binary: one constrained classifier. Ordinal: Frank & Hall decomposition into K-1
    constrained binary models for P(y > k), made non-increasing in k before taking class probabilities.
    Nominal targets have no order, so the model is not defined there (raises)."""

    def __init__(self, task, seed):
        self.task, self.seed = task, seed

    def _xgb(self):
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=self.seed, verbosity=0,
                             n_jobs=1, monotone_constraints="(" + ",".join(map(str, self.dir_)) + ")")

    def fit(self, X, y):
        from scipy.stats import spearmanr
        if self.task == "nominal":
            raise ValueError("monotone constraints undefined for nominal targets")
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        r = [spearmanr(X[:, j], yi).correlation if np.ptp(X[:, j]) > 0 else 0.0 for j in range(X.shape[1])]
        self.dir_ = [1 if (np.isfinite(c) and c >= 0) else -1 for c in r]
        self.ms_ = []
        for k in range(len(self.classes_) - 1):
            t = (yi > k).astype(int)
            self.ms_.append(self._xgb().fit(X, t) if 0 < t.sum() < len(t) else float(t.mean()))
        return self

    def predict_proba(self, X):
        G = np.column_stack([m.predict_proba(X)[:, 1] if not isinstance(m, float) else np.full(len(X), m)
                             for m in self.ms_])  # P(y > k)
        G = np.minimum.accumulate(G, axis=1)
        cum = np.column_stack([np.ones(len(X)), G, np.zeros(len(X))])
        return np.clip(cum[:, :-1] - cum[:, 1:], 0, 1)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


class _FrankHall:
    """Ordinal wrapper: K-1 binary models for P(y > k), made non-increasing in k."""

    def _fit_binary(self, X, t):
        raise NotImplementedError

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        self.ms_ = []
        for k in range(len(self.classes_) - 1):
            t = (yi > k).astype(int)
            self.ms_.append(self._fit_binary(X, t) if 0 < t.sum() < len(t) else float(t.mean()))
        return self

    def _p1(self, m, X):
        return np.full(len(X), m) if isinstance(m, float) else m.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        G = np.minimum.accumulate(np.column_stack([self._p1(m, X) for m in self.ms_]), axis=1)
        cum = np.column_stack([np.ones(len(X)), G, np.zeros(len(X))])
        return np.clip(cum[:, :-1] - cum[:, 1:], 0, 1)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


class MonoGAM(_FrankHall):
    """Monotone logistic GAM (pyGAM): one penalised B-spline term per feature (10 basis functions) with a monotone
    increasing/decreasing constraint given by the sign of the training Spearman correlation; binary features enter
    linearly. Ordinal targets via Frank-Hall. Undefined for nominal targets."""

    def __init__(self, task):
        self.task = task

    def _fit_binary(self, X, t):
        from pygam import LogisticGAM, s as S_, l as L_
        terms = None
        for j in range(X.shape[1]):
            nu = len(np.unique(X[:, j]))
            if nu <= 2:
                tj = L_(j)
            else:
                tj = S_(j, n_splines=min(10, max(4, nu)), constraints="monotonic_inc" if self.dir_[j] > 0 else "monotonic_dec")
            terms = tj if terms is None else terms + tj
        return _GAMWrap(LogisticGAM(terms, max_iter=200).fit(self.sc_.transform(X), t), self.sc_)

    def fit(self, X, y):
        from scipy.stats import spearmanr
        from sklearn.preprocessing import StandardScaler as SS
        if self.task == "nominal":
            raise ValueError("monotone GAM undefined for nominal targets")
        yi = np.searchsorted(np.unique(y), y)
        r = [spearmanr(X[:, j], yi).correlation if np.ptp(X[:, j]) > 0 else 0.0 for j in range(X.shape[1])]
        self.dir_ = [1 if (np.isfinite(c) and c >= 0) else -1 for c in r]
        self.sc_ = SS().fit(X)
        return super().fit(X, y)


class _GAMWrap:
    def __init__(self, g, sc):
        self.g, self.sc = g, sc

    def predict_proba(self, X):
        p = self.g.predict_proba(self.sc.transform(X))
        return np.column_stack([1 - p, p])


class POLR:
    """Proportional-odds (cumulative logit) regression on standardised inputs (mord.LogisticAT, alpha=1e-3).
    Ordinal targets only."""

    def __init__(self, task):
        self.task = task

    def fit(self, X, y):
        import mord
        if self.task != "ordinal":
            raise ValueError("proportional odds requires an ordinal target")
        self.classes_ = np.unique(y)
        self.m_ = make_pipeline(StandardScaler(), mord.LogisticAT(alpha=1e-3, max_iter=5000))
        self.m_.fit(X, np.searchsorted(self.classes_, y))
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


class FrozenBorder:
    """Ablation: unsupervised border-approximation score with a frozen training reference (entropy weights,
    Spearman directions, one geometric-mean border per criterion, kappa -> 0); only the output layer is learned:
    logistic (binary), proportional odds on the score (ordinal), multinomial logistic on the C one-vs-rest scores
    (nominal)."""

    def __init__(self, task):
        self.task = task

    def fit(self, X, y):
        from bamakan import MABACClassifier
        self.classes_ = np.unique(y)
        self.b_ = MABACClassifier().fit(X, y)
        S = self.b_.decision_function(X)
        if self.task == "ordinal":
            import mord
            self.o_ = mord.LogisticAT(alpha=1e-6, max_iter=5000).fit(S[:, :1] if S.shape[1] == 1 else S.mean(1, keepdims=True), np.searchsorted(self.classes_, y))
        else:
            self.o_ = LogisticRegression(max_iter=5000, C=1e6).fit(S, y)
        return self

    def _S(self, X):
        S = self.b_.decision_function(X)
        return (S[:, :1] if S.shape[1] == 1 else S.mean(1, keepdims=True)) if self.task == "ordinal" else S

    def predict_proba(self, X):
        return self.o_.predict_proba(self._S(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


# Configuration reported in the paper (single fixed setting, no per-dataset tuning):
# hard criterion directions = sign of the training-fold Spearman correlation (ties/zero -> benefit);
# B = 4 ordered power-mean borders initialised at training quantiles, kappa per criterion and border (init 3, bounded to [1e-4, 200]) with the normalised
# soft comparison tanh(kz)/tanh(k), and hybrid Adam -> L-BFGS training with early stopping.
BAMAKAN_CONFIG = dict(direction="corr", n_borders=4, kappa_mode="local", psi="normed", kappa0=3.0, p_init="quantile",
                      lr_schedule="cosine", optimizer="hybrid", lbfgs_block=20, epochs=600, patience=60)


def make_models(task, seed):
    from interpret.glassbox import ExplainableBoostingClassifier
    btask = {"binary": "binary", "nominal": "nominal", "ordinal": "ordinal"}[task]
    return {
        "BAMAKAN": lambda: BAMAKANClassifier(task=btask, **BAMAKAN_CONFIG, random_state=seed),
        "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "NearestCentroid": lambda: NC(),
        "GMLVQ": lambda: GMLVQ(seed),
        "SVM": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                                                         random_state=seed)),
        "kNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "EBM": lambda: ExplainableBoostingClassifier(random_state=seed, n_jobs=1),
        "EBM-main": lambda: ExplainableBoostingClassifier(interactions=0, random_state=seed, n_jobs=1),
        "GAM-mono": lambda: MonoGAM(task),
        "POLR": lambda: POLR(task),
        "Border-frozen": lambda: FrozenBorder(task),
        "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 64), early_stopping=True, max_iter=500,
                                                                      random_state=seed)),
        "RF": lambda: RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=seed,
                                         verbosity=0, n_jobs=1),
        "XGB-mono": lambda: MonoXGB(task, seed),
        "CatBoost": lambda: CatBoostClassifier(random_seed=seed, thread_count=1, verbose=0),
        "RealMLP": lambda: RealMLP_TD_Classifier(random_state=seed, n_threads=1, device="cpu", verbosity=0),
        "LightGBM": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=seed,
                                           verbose=-1, n_jobs=1),
    }


def metrics(task, y, P, classes):
    P = np.clip(P, 1e-7, 1)
    P = P / P.sum(1, keepdims=True)
    pred = classes[np.argmax(P, 1)]
    out = dict(acc=accuracy_score(y, pred), bacc=balanced_accuracy_score(y, pred),
               f1_macro=f1_score(y, pred, average="macro"), logloss=log_loss(y, P, labels=classes))
    if task == "binary":
        out["auc"] = roc_auc_score(y, P[:, 1])
        out["brier"] = brier_score_loss(y, P[:, 1])
        # expected calibration error, 10 bins
        conf = P[:, 1]; bins = np.clip((conf * 10).astype(int), 0, 9)
        out["ece"] = sum(abs(conf[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
                         for b in range(10) if (bins == b).any())
    if task == "binary":
        from sklearn.metrics import average_precision_score
        out["pr_auc"] = average_precision_score(y == classes[1], P[:, 1])
    if task == "ordinal":
        from sklearn.metrics import cohen_kappa_score
        out["mae"] = mean_absolute_error(y, pred)  # in class-index units
        out["qwk"] = cohen_kappa_score(y, pred, weights="quadratic")
        Y = (np.searchsorted(classes, y)[:, None] <= np.arange(len(classes))[None]).astype(float)
        out["rps"] = float(np.mean(np.sum((np.cumsum(P, 1) - Y)[:, :-1] ** 2, 1) / max(len(classes) - 1, 1)))
    return out


def run(name, n_splits=5, n_repeats=1):
    X, y, task, fn = DATASETS[name]()
    classes = np.unique(y)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    rows = []
    for f, (tr, te) in enumerate(rskf.split(X, y)):
        for mname, mk in make_models(task, f).items():
            if ONLY and mname not in ONLY:
                continue
            t0 = time.time()
            try:
                m = mk().fit(X[tr], y[tr])
                fit_t = time.time() - t0
                t1 = time.time()
                P = m.predict_proba(X[te])
                inf_t = (time.time() - t1) / len(te) * 1e3
                # align columns to all classes
                Pf = np.zeros((len(te), len(classes)))
                Pf[:, np.searchsorted(classes, np.asarray(m.classes_))] = P
                r = metrics(task, y[te], Pf, classes)
                r.update(fit_s=fit_t, infer_ms_per_1k=inf_t * 1e3 / 1e3)
            except Exception as e:  # record failures explicitly
                r = dict(error=repr(e)[:200])
            r.update(dataset=name, task=task, model=mname, fold=f)
            rows.append(r)
        print(name, "fold", f, "done", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(f"{OUT}/{name}{SUFFIX}.json", "w"))
    return rows


if __name__ == "__main__":
    for n in sys.argv[1:]:
        t = time.time()
        run(n)
        print(n, "finished in", round(time.time() - t, 1), "s", flush=True)

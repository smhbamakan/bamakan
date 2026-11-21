"""P2 synthetic experiments for the BAMAKAN global branch.

S1  parameter recovery (directions, weights, border locations in original units)
S2  rank/region reversal of classical (transductive) MABAC vs. BAMAKAN (inductive)
S3  identifiability of kappa: distance-type vs region-count-type ground truth
S4  accuracy across data regimes vs. baselines (incl. a non-monotone XOR regime = expected failure)
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, sys, time, warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, f"{ROOT}")
from bamakan import BAMAKANClassifier, MABACClassifier, Normaliser
sys.path.insert(0, f"{ROOT}/experiments")
from benchmark import BAMAKAN_CONFIG
FAST = {k: BAMAKAN_CONFIG[k] for k in ("optimizer", "lbfgs_block", "epochs", "patience")}  # training procedure only (S1-S3 use one border)

warnings.filterwarnings("ignore")
OUT = f"{ROOT}/results"
sig = lambda z: 1 / (1 + np.exp(-z))


def gen_threshold(n, seed, d_inf=5, d_noise=3):
    """Monotone threshold model with known weights, directions and borders (original units)."""
    r = np.random.RandomState(seed)
    X = np.column_stack([r.lognormal(0, 0.5, n) * (j + 1) * 10 for j in range(d_inf + d_noise)])
    w = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
    s = np.array([1, 1, 0, 1, 0])
    t = np.array([np.quantile(X[:, j], q) for j, q in enumerate([0.3, 0.5, 0.6, 0.4, 0.7])])
    contrib = np.column_stack([w[j] * np.tanh(4 * ((X[:, j] - t[j]) / X[:, j].std()) * (1 if s[j] else -1))
                               for j in range(d_inf)])
    lat = contrib.sum(1)
    y = (r.rand(n) < sig(8 * (lat - np.median(lat)))).astype(int)
    return X, y, w, s, t


def s1_recovery(seeds=10, n=4000):
    rows = []
    for sd in range(seeds):
        X, y, w, s, t = gen_threshold(n, sd)
        m = BAMAKANClassifier(n_borders=1, direction="corr", random_state=sd, **FAST).fit(X, y)
        b = m.borders_in_original_units()
        lw = np.array([r["weight"] for r in b])
        ls = np.array([r["benefit"] > 0.5 for r in b]).astype(int)
        lv = np.array([r["value"] for r in b])
        rel_err = np.abs(lv[:5] - t) / X[:, :5].std(0)
        rows.append(dict(seed=sd, dir_acc=float((ls[:5] == s).mean()),
                         w_spearman=float(spearmanr(lw[:5], w).statistic),
                         w_noise_share=float(lw[5:].sum()),
                         border_err_sd=float(np.mean(rel_err)),
                         kappa=float(m.net_.kappa().mean().item())))
    return rows


def s2_reversal(seeds=20, n=200, add=40):
    """Fraction of original points whose G+/G- region changes when `add` new points are scored."""
    out = []
    for sd in range(seeds):
        r = np.random.RandomState(sd)
        X = r.lognormal(0, 0.6, (n, 6))
        y = (X[:, 0] + X[:, 1] - X[:, 2] + 0.3 * r.randn(n) > 1.0).astype(int)
        Xnew = r.lognormal(0.8, 0.6, (add, 6))  # shifted new alternatives

        def cls_mabac(Xs):
            N = Normaliser().fit(Xs).transform(Xs)
            w = np.full(6, 1 / 6)
            V = w * (N + 1)
            return (V - np.prod(V, 0) ** (1 / len(V))).sum(1)

        a = cls_mabac(X)
        b = cls_mabac(np.vstack([X, Xnew]))[:n]
        ra, rb = np.argsort(np.argsort(-a)), np.argsort(np.argsort(-b))
        m = BAMAKANClassifier(n_borders=1, random_state=sd, **FAST).fit(X, y)
        c = m.decision_function(X)[:, 0]
        d = m.decision_function(np.vstack([X, Xnew]))[:n, 0]
        out.append(dict(seed=sd, mabac_region_flip=float(np.mean(np.sign(a) != np.sign(b))),
                        mabac_rank_changes=float(np.mean(ra != rb)),
                        bamakan_region_flip=float(np.mean(np.sign(c) != np.sign(d))),
                        bamakan_rank_changes=float(np.mean(np.argsort(np.argsort(-c)) != np.argsort(np.argsort(-d))))))
    return out


def s3_kappa(seeds=8, n=4000, d=6):
    rows = []
    for sd in range(seeds):
        r = np.random.RandomState(sd)
        X = r.rand(n, d)
        w = np.full(d, 1 / d)
        lin = (w * X).sum(1)
        cnt = (w * np.sign(X - 0.5)).sum(1)
        for kind, lat in [("distance", lin), ("region_count", cnt)]:
            y = (r.rand(n) < sig(25 * (lat - np.median(lat)))).astype(int)
            m = BAMAKANClassifier(n_borders=1, random_state=sd, kappa0=1.0, **FAST).fit(X, y)
            rows.append(dict(seed=sd, truth=kind, kappa=float(m.net_.kappa().mean().item())))
    return rows


def regimes(n, seed):
    r = np.random.RandomState(seed)
    X = r.lognormal(0, 0.5, (n, 8))
    Z = (X - X.mean(0)) / X.std(0)
    out = {}
    out["linear"] = Z[:, 0] + 0.7 * Z[:, 1] - 0.5 * Z[:, 2] + 0.3 * Z[:, 3]
    out["monotone_threshold"] = np.tanh(3 * Z[:, 0]) + 0.8 * (Z[:, 1] > 0.5) - 0.8 * (Z[:, 2] > -0.3) + 0.5 * np.tanh(2 * Z[:, 3])
    out["region_count"] = np.sign(Z[:, :6] - 0.2).sum(1) / 6
    out["xor_nonmonotone"] = np.sign(Z[:, 0]) * np.sign(Z[:, 1]) + 0.3 * Z[:, 2]
    res = {}
    for k, lat in out.items():
        y = (r.rand(n) < sig(4 * (lat - np.median(lat)) / lat.std())).astype(int)
        res[k] = (X, y)
    return res


def models(seed):
    return {
        "BAMAKAN": lambda: BAMAKANClassifier(**BAMAKAN_CONFIG, random_state=seed),
        "MABAC (classical)": lambda: MABACClassifier(),
        "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, verbosity=0, random_state=seed),
    }


def s4_regimes(seeds=5, n=3000):
    from interpret.glassbox import ExplainableBoostingClassifier
    rows = []
    for sd in range(seeds):
        for reg, (X, y) in regimes(n, sd).items():
            tr = np.arange(n) < int(0.7 * n)
            ms = models(sd)
            ms["EBM"] = lambda: ExplainableBoostingClassifier(random_state=sd, interactions=0)
            for name, mk in ms.items():
                m = mk().fit(X[tr], y[tr])
                p = m.predict_proba(X[~tr])[:, 1]
                rows.append(dict(seed=sd, regime=reg, model=name, auc=float(roc_auc_score(y[~tr], p)),
                                 acc=float(accuracy_score(y[~tr], m.predict(X[~tr])))))
    return rows


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res = {}
    res["S1"] = s1_recovery(); print("S1", time.time() - t0, flush=True)
    res["S2"] = s2_reversal(); print("S2", time.time() - t0, flush=True)
    res["S3"] = s3_kappa(); print("S3", time.time() - t0, flush=True)
    res["S4"] = s4_regimes(); print("S4", time.time() - t0, flush=True)
    json.dump(res, open(f"{OUT}/synthetic.json", "w"), indent=1)
    import pandas as pd
    for k in ["S1", "S2", "S3"]:
        df = pd.DataFrame(res[k])
        print(k); print(df.drop(columns="seed").groupby("truth").agg(["mean", "std"]) if k == "S3" else df.drop(columns="seed").agg(["mean", "std"]).T)
    df = pd.DataFrame(res["S4"])
    print(df.pivot_table(index="regime", columns="model", values="auc", aggfunc="mean").round(3))
    print(df.pivot_table(index="regime", columns="model", values="auc", aggfunc="std").round(3))

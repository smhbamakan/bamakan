"""Clinical analyses on Heart Failure, HCV, Cirrhosis and Maternal Health Risk.
A CIs (bootstrap over patients, out-of-fold predictions of repeat 0)   B per-class recall and confusion matrices
C calibration + decision curve                                          D bootstrap stability of borders and directions
E representative patients                                               F class-conditional (Mondrian) cross-conformal
G if-then scorecards: fidelity in CV and the final card"""
import os, sys, json, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, f"{ROOT}/experiments"]
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, cohen_kappa_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from bamakan import BAMAKANClassifier
from benchmark import BAMAKAN_CONFIG
from datasets import DATASETS
torch.set_num_threads(1)
OUT = f"{ROOT}/results/clinical"
DS = ["heartfail", "hcv", "cirrhosis", "maternal"]
CMP = ["BAMAKAN", "EBM-main", "EBM", "LogReg", "CatBoost", "GAM-mono", "XGB-mono"]
rng = np.random.RandomState(0)
res = {}


def oof(name, rep=0):
    X, y, task, f = DATASETS[name]()
    P = {}
    for k in range(5):
        z = np.load(f"{ROOT}/results/cv/parts/{name}__r{rep}_f{k}.npz")
        for m in CMP:
            if m in z:
                P.setdefault(m, np.zeros((len(y), len(np.unique(y)))))[z["idx"]] = z[m]
    return X, y, task, f, P


def stats(task, y, P):
    yh = P.argmax(1)
    if task == "binary":
        return dict(auc=roc_auc_score(y, P[:, 1]), bacc=balanced_accuracy_score(y, yh))
    e = P @ np.arange(P.shape[1])
    return dict(mae=np.abs(np.round(e) - y).mean(), mae_argmax=np.abs(yh - y).mean(), qwk=cohen_kappa_score(y, yh, weights="quadratic"),
                bacc=balanced_accuracy_score(y, yh))


for name in DS:
    X, y, task, f, P = oof(name)
    r = res[name] = {"task": task, "n": len(y), "classes": np.bincount(y).tolist()}
    # A: bootstrap CIs
    A = {}
    for m, Pm in P.items():
        point = stats(task, y, Pm); boots = []
        for _ in range(1000):
            i = rng.randint(0, len(y), len(y))
            if len(np.unique(y[i])) < len(np.unique(y)): continue
            boots.append(stats(task, y[i], Pm[i]))
        B = pd.DataFrame(boots)
        A[m] = {k: [point[k], float(B[k].quantile(.025)), float(B[k].quantile(.975))] for k in point}
    r["A_ci"] = A
    # B: per-class recall and confusion (BAMAKAN and EBM-main)
    r["B_recall"] = {m: recall_score(y, P[m].argmax(1), average=None).tolist() for m in ["BAMAKAN", "EBM-main", "CatBoost"]}
    r["B_confusion"] = {m: confusion_matrix(y, P[m].argmax(1)).tolist() for m in ["BAMAKAN", "EBM-main"]}
    # C: calibration of the top-risk event (binary: y=1; ordinal: y >= 1, i.e. any disease / not low risk)
    C = {}
    for m in ["BAMAKAN", "EBM-main", "LogReg", "CatBoost"]:
        p = P[m][:, 1:].sum(1); ev = (y >= 1).astype(int)
        q = np.quantile(p, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
        C[m] = dict(pred=[float(p[b == k].mean()) for k in range(10) if (b == k).any()],
                    obs=[float(ev[b == k].mean()) for k in range(10) if (b == k).any()],
                    ece=float(sum(abs(p[b == k].mean() - ev[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any())))
        if task == "binary":
            nb = []
            for t in np.linspace(0.05, 0.6, 12):
                tp = ((p >= t) & (ev == 1)).mean(); fp = ((p >= t) & (ev == 0)).mean()
                nb.append([float(t), float(tp - fp * t / (1 - t))])
            C[m]["net_benefit"] = nb
    if task == "binary":
        C["treat_all"] = [[float(t), float(ev.mean() - (1 - ev.mean()) * t / (1 - t))] for t in np.linspace(0.05, 0.6, 12)]
    r["C_calibration"] = C
    # F: Mondrian cross-conformal at alpha = 0.1 (thresholds from the other folds' out-of-fold scores, per true class)
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42).split(X, y))
    F = {}
    for m in ["BAMAKAN", "EBM-main", "CatBoost", "LogReg"]:
        sc = 1 - P[m][np.arange(len(y)), y]; cov = np.zeros(len(y)); size = np.zeros(len(y))
        for tr, te in folds:
            qk = []
            for k in range(P[m].shape[1]):
                s = np.sort(sc[tr][y[tr] == k]); n = len(s)
                qk.append(s[min(n - 1, int(np.ceil((n + 1) * 0.9)) - 1)] if n else 1.0)
            S = (1 - P[m][te]) <= np.array(qk)[None, :]
            cov[te] = S[np.arange(len(te)), y[te]]; size[te] = S.sum(1)
        F[m] = dict(coverage_per_class=[float(cov[y == k].mean()) for k in range(P[m].shape[1])], mean_set_size=float(size.mean()))
    r["F_conformal"] = F
    # D/E/G need fitted models
    cfg = {**BAMAKAN_CONFIG, "task": task}
    # G: scorecard fidelity in CV (repeat-0 folds)
    G = {"full": [], "card": [], "card_refit": [], "agree": [], "agree_refit": [], "n_rules": []}
    for k, (tr, te) in enumerate(folds):
        m = BAMAKANClassifier(**{**cfg, "random_state": k}).fit(X[tr], y[tr])
        c0 = m.scorecard(f); c1 = m.scorecard(f, X=X[tr], y=y[tr])
        Pf, P0, P1 = m.predict_proba(X[te]), m.predict_proba_scorecard(X[te], c0), m.predict_proba_scorecard(X[te], c1)
        key = "auc" if task == "binary" else "mae"
        G["full"].append(stats(task, y[te], Pf)[key]); G["card"].append(stats(task, y[te], P0)[key])
        G["card_refit"].append(stats(task, y[te], P1)[key])
        G["agree"].append(float((Pf.argmax(1) == P0.argmax(1)).mean())); G["agree_refit"].append(float((Pf.argmax(1) == P1.argmax(1)).mean()))
        G["n_rules"].append(len(c1["rules"]))
    r["G_fidelity"] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in G.items()}
    full = BAMAKANClassifier(**{**cfg, "random_state": 0}).fit(X, y)
    card = full.scorecard(f, X=X, y=y)
    r["G_card"] = dict(rules=[{k: x[k] for k in ("feature", "op", "value", "points")} for x in card["rules"]],
                       unit_logit_per_point=card["unit"], intercept=card["intercept"], cutpoints=np.asarray(card["thresholds"]).tolist())
    # E: representative patients (lowest, median and highest predicted risk) with per-criterion contributions
    risk = full.predict_proba(X)[:, 1:].sum(1); order = np.argsort(risk)
    phi = full.explain(X) * float(torch.exp(full.net_.log_tau))
    r["E_patients"] = [dict(row=int(i), risk=float(risk[i]), true=int(y[i]), values=dict(zip(f, map(float, X[i]))),
                            contrib=dict(zip(f, map(float, phi[i])))) for i in (order[0], order[len(order) // 2], order[-1])]
    # D: bootstrap stability (50 resamples): direction agreement and border locations in original units
    dirs, bord = [], []
    for b in range(50):
        i = np.random.RandomState(b).randint(0, len(y), len(y))
        mb = BAMAKANClassifier(**{**cfg, "random_state": b}).fit(X[i], y[i])
        dirs.append(mb.net_.s_full().detach().numpy()[0, :, 0])
        rows = mb.borders_in_original_units(f)
        bord.append({(x["feature"], x["border"]): x["value"] for x in rows})
    s_full = full.net_.s_full().detach().numpy()[0, :, 0]
    D = {}
    for j, fn in enumerate(f):
        vals = np.array([[bb[(fn, k)] for k in range(cfg["n_borders"])] for bb in bord])
        vals = np.sort(vals, 1)
        lo, hi = full.norm_.inverse(0, j), full.norm_.inverse(1, j)
        D[fn] = dict(direction_agreement=float((np.array(dirs)[:, j] == s_full[j]).mean()),
                     border_median=np.median(vals, 0).tolist(), border_iqr=(np.percentile(vals, 75, 0) - np.percentile(vals, 25, 0)).tolist(),
                     range=[float(lo), float(hi)])
    r["D_stability"] = D
    json.dump(res, open(f"{OUT}/clinical.json", "w"), indent=1, default=float)
    print(name, "done", flush=True)
print("PHASE5_DONE")

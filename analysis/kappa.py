"""Does the learned sharpness kappa carry interpretable information?
(1) Per criterion: kappa vs fidelity of the sharp-limit (step) approximation of its contribution.
(2) Per criterion: kappa vs an empirical, model-free step-likeness index of the data
    S_j = R2(best single split of y on x_j) / R2(isotonic fit of y on x_j)  (1 = a single threshold explains all that
    any monotone function of x_j explains; small = gradual relation).
(3) Datasets on which multiple borders / adaptive kappa matter (ablation losses vs n and number of classes)."""
import os, sys, json, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, f"{ROOT}/experiments"]
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor
from bamakan import BAMAKANClassifier
from benchmark import BAMAKAN_CONFIG
from run_cv import load
from ablation import DATASETS_ALL
torch.set_num_threads(2)
rows = []
for name in DATASETS_ALL:
    if name == "maternal": continue
    X, y, task, _ = load(name)
    if task == "nominal": continue
    m = BAMAKANClassifier(**{**BAMAKAN_CONFIG, "task": task, "random_state": 0}).fit(X, y)
    net = m.net_; N = m._N(X)
    with torch.no_grad():
        U = net.u(N)[:, 0]; M = net.borders()[0]; k = net.kappa()[0]; beta = torch.softmax(net.gamma, -1)[0]; w = net.weights()[0]
        phi = net.contributions(N)[:, 0].numpy(); tau = float(torch.exp(net.log_tau))
        step = (w[None, :] * (beta[None] * torch.sign(U - M[None])).sum(-1)).numpy()
        s = net.s_full()[0, :, 0].numpy()
    yi = np.searchsorted(np.unique(y), y).astype(float)
    imp = tau * np.abs(phi - phi.mean(0)).mean(0)
    for j in range(X.shape[1]):
        if np.var(phi[:, j]) < 1e-12 or len(np.unique(X[:, j])) < 5: continue  # skip binary / constant criteria
        a = np.polyfit(step[:, j], phi[:, j], 1) if np.var(step[:, j]) > 0 else None
        r2_step = 1 - np.var(phi[:, j] - np.polyval(a, step[:, j])) / np.var(phi[:, j]) if a is not None else 0.0
        xj = X[:, j] if s[j] == 1 else -X[:, j]
        iso = IsotonicRegression(out_of_bounds="clip").fit(xj, yi).predict(xj)
        st = DecisionTreeRegressor(max_depth=1).fit(xj[:, None], yi).predict(xj[:, None])
        r2i = 1 - np.var(yi - iso) / np.var(yi); r2s = 1 - np.var(yi - st) / np.var(yi)
        rows.append(dict(dataset=name, task=task, j=j, log_kappa=float((beta[j] * torch.log(k[j])).sum()), r2_step=r2_step,
                         step_index=r2s / r2i if r2i > 0.01 else np.nan, r2_iso=r2i, importance=imp[j], rel_imp=imp[j] / imp.max()))
    print(name, flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{ROOT}/results/kappa/kappa.csv", index=False)
I = D[D.rel_imp >= 0.1]
out = {"n_criteria": int(len(I)), "n_datasets": int(I.dataset.nunique()),
       "rho_kappa_fidelity": spearmanr(I.log_kappa, I.r2_step), "rho_kappa_stepindex": spearmanr(I.dropna().log_kappa, I.dropna().step_index)}
# within-dataset (removes dataset-level confounding): mean of per-dataset Spearman correlations
per = [spearmanr(g.log_kappa, g.step_index).correlation for _, g in I.dropna().groupby("dataset") if len(g) >= 4]
out["within_dataset_rho_stepindex_mean"] = float(np.nanmean(per)); out["within_n"] = len(per)
q = pd.qcut(I.log_kappa, 3, labels=["low", "mid", "high"])
out["fidelity_by_kappa_tercile"] = I.groupby(q).r2_step.median().round(3).to_dict()
out["stepindex_by_kappa_tercile"] = I.groupby(q).step_index.median().round(3).to_dict()
# (3) ablation losses vs dataset size and number of classes
A = pd.DataFrame(json.load(open(f"{ROOT}/results/ablation/all.json"))); A = A[A.dataset != "maternal"]
Am = A.groupby(["dataset", "task", "variant"]).mean(numeric_only=True).reset_index()
prim = {"binary": ("auc", 1), "nominal": ("acc", 1), "ordinal": ("mae", -1)}
sz = {n: (lambda t: (len(t[1]), len(np.unique(t[1]))))(load(n)) for n in Am.dataset.unique()}
for v in ["B1", "kappa_fixed"]:
    L = []
    for t, (met, sg) in prim.items():
        P = Am[Am.task == t].pivot(index="dataset", columns="variant", values=met)
        for d_, x in (sg * (P[v] - P["default"])).items(): L.append((d_, x, *sz[d_]))
    L = pd.DataFrame(L, columns=["dataset", "gain", "n", "K"])
    out[f"{v}_rho_gain_n"] = spearmanr(L.gain, L.n); out[f"{v}_rho_gain_K"] = spearmanr(L.gain, L.K)
    big = L[L.n >= 2000]; small = L[L.n < 2000]
    out[f"{v}_mean_gain_n>=2000"] = (float(big.gain.mean()), len(big)); out[f"{v}_mean_gain_n<2000"] = (float(small.gain.mean()), len(small))
print(json.dumps(out, indent=1, default=lambda o: [round(float(v), 4) for v in o] if hasattr(o, "__iter__") else str(o)))
json.dump(out, open(f"{ROOT}/results/kappa/kappa.json", "w"), indent=1, default=lambda o: [float(v) for v in o] if hasattr(o, "__iter__") else str(o))

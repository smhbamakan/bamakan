"""All figures of manuscript v0.7, drawn at the printed text width (6.7 in) so that fonts appear at their nominal size.
Run from the repository root after analysis/make_tables.py:  python analysis/make_figures.py"""
import os, sys, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, f"{ROOT}/experiments"]
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.gridspec import GridSpec

W = 6.7  # text width in inches
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
O = f"{ROOT}/figures"
RED, G, BLUE, GREEN, PURPLE = "#c0392b", "#7f8c8d", "#2e86c1", "#27ae60", "#8e44ad"
os.chdir(ROOT)

# ---------------------------------------------------------------- Fig 1: framework (two rows)
fig, ax = plt.subplots(figsize=(W, 4.8)); ax.axis("off"); ax.set_xlim(0, 3); ax.set_ylim(-0.15, 3.0)
boxes = [("Criteria", "$x_1,\\dots,x_d$", "raw values"),
         ("Normalisation", "$n_j(x)$", "bounds frozen on\nthe training data"),
         ("Direction", "$s_{j}\\in\\{0,1\\}$", "benefit or cost, fixed\nbefore fitting"),
         ("Border levels", "$m_{jk}=M_{p_{jk}}(u_j)$", "B ordered power-mean\nborders per criterion"),
         ("Comparison", "$\\psi_{\\kappa_{jk}}(u_j-m_{jk})$", "learned sharpness:\nsmooth $\\leftrightarrow$ step"),
         ("Output layer", "$P(y\\mid x)$", "threshold / cumulative\nlink / multi-head")]
pos = [(0.05, 1.85), (1.05, 1.85), (2.05, 1.85), (0.05, 0.5), (1.05, 0.5), (2.05, 0.5)]
for i, ((t, f, s), (x, y)) in enumerate(zip(boxes, pos)):
    ax.add_patch(FancyBboxPatch((x, y), 0.9, 1.05, boxstyle="round,pad=0.02", fc="#fbeee6" if i in (3, 4) else "#f4f6f7",
                                ec="#34495e", lw=1.2))
    ax.text(x + 0.45, y + 0.86, t, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + 0.45, y + 0.60, f, ha="center", va="center", fontsize=11)
    ax.text(x + 0.45, y + 0.25, s, ha="center", va="center", fontsize=9)
arr = dict(arrowstyle="-|>", color="#34495e", lw=1.3)
for a, b in [(0, 1), (1, 2), (3, 4), (4, 5)]:
    ax.annotate("", xy=(pos[b][0] - 0.01, pos[b][1] + 0.52), xytext=(pos[a][0] + 0.91, pos[a][1] + 0.52), arrowprops=arr)
ax.plot([2.5, 2.5, 0.5], [1.83, 1.71, 1.71], color="#34495e", lw=1.3)
ax.annotate("", xy=(0.5, 1.57), xytext=(0.5, 1.72), arrowprops=arr)
ax.text(1.5, 0.2, "contribution $\\varphi_j(x)=w_j\\sum_k\\beta_{jk}\\,\\psi_{\\kappa_{jk}}(u_j-m_{jk})$,   score $S(x)=\\sum_j\\varphi_j(x)$",
        ha="center", fontsize=10)
ax.text(1.5, -0.12, "Each prediction = signed criterion contributions; borders reported in original units", ha="center",
        fontsize=9.5, style="italic", color="#34495e")
plt.savefig(f"{O}/fig1_framework.png"); plt.close()

# ---------------------------------------------------------------- Fig 2: synthetic (a, b on top; c below)
r = json.load(open("results/synthetic.json"))
fig = plt.figure(figsize=(W, 6.2)); gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.05], hspace=0.55, wspace=0.35)
a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
s2 = pd.DataFrame(r["S2"])
vals = [s2.mabac_region_flip.mean(), s2.bamakan_region_flip.mean(), s2.mabac_rank_changes.mean(), s2.bamakan_rank_changes.mean()]
a0.bar([0, 1, 2.6, 3.6], vals, color=[G, RED, G, RED], width=0.7)
a0.set_xticks([0.5, 3.1], ["Region flips", "Rank changes"])
for xx, v in zip([0, 1, 2.6, 3.6], vals): a0.text(xx, v + 0.02, f"{100 * v:.0f}%", ha="center", fontsize=9)
a0.set_ylabel("Share of existing cases"); a0.set_ylim(0, 1.45); a0.set_title("(a) 100 new cases are scored")
a0.legend(handles=[Patch(color=G, label="Set-based rule"), Patch(color=RED, label="BAMAKAN")], frameon=False, loc="upper left", ncol=1)
s3 = pd.DataFrame(r["S3"])
for i, t in enumerate(["distance", "region_count"]):
    v = s3[s3.truth == t].kappa.values
    a1.scatter(np.full(len(v), i) + np.random.RandomState(0).uniform(-.1, .1, len(v)), v, color=[BLUE, RED][i], s=30)
a1.set_yscale("log"); a1.set_xticks([0, 1], ["distance-type\ntruth", "count-type\ntruth"]); a1.set_xlim(-.5, 1.5)
a1.set_ylabel("Learned $\\kappa$ (log scale)"); a1.set_title("(b) Form of aggregation")
s4 = pd.DataFrame(r["S4"]).groupby(["regime", "model"]).auc.mean().unstack()
regs = ["linear", "monotone_threshold", "region_count", "xor_nonmonotone"]; lab = ["Linear", "Threshold", "Count", "XOR (non-monotone)"]
ms = [("BAMAKAN", RED), ("EBM", BLUE), ("XGBoost", GREEN), ("LogReg", PURPLE)]
for k, (m, c) in enumerate(ms):
    a2.bar(np.arange(4) + k * 0.2 - 0.3, s4.loc[regs, m], 0.19, color=c, label="EBM (main effects)" if m == "EBM" else m)
a2.set_xticks(range(4), lab); a2.set_ylim(0.5, 1.0); a2.set_ylabel("Test AUC"); a2.set_title("(c) Accuracy by data-generating regime")
a2.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.savefig(f"{O}/fig2_synthetic.png"); plt.close()

# ---------------------------------------------------------------- shared rank data (from make_v07 outputs)
EXCL = {"maternal"}; CLIN = ["heartfail", "hcv", "cirrhosis"]; MONO_DS = ["era", "esl", "lev", "swd"]
MAIN = ["BAMAKAN", "TabPFN", "EBM-main", "LogReg", "CatBoost", "XGBoost", "LightGBM", "RF", "SVM", "RealMLP", "MLP", "GMLVQ", "kNN", "NearestCentroid"]
PRIM = {"binary": ("auc", 1), "nominal": ("acc", 1), "ordinal": ("mae", -1)}
NAME = {"NearestCentroid": "Nearest centroid"}
rows = []
for f in glob.glob("results/cv/*.json"): rows += json.load(open(f))
rows += [json.load(open(f)) for f in glob.glob("results/cv_tabpfn/*.json")]
df = pd.DataFrame(rows)
df = df[(df.repeat < 2) & df["error"].isna() & ~df.dataset.isin(EXCL)]
df = df[~(df.model.isin(["EBM", "EBM-main"]) & (df.repeat > 0))]
mean = df.groupby(["dataset", "task", "model"]).mean(numeric_only=True).reset_index()
S = pd.concat([sg * mean[mean.task == t].pivot(index="dataset", columns="model", values=m)[MAIN] for t, (m, sg) in PRIM.items()])
R = S.rank(axis=1, ascending=False); avg = R.mean().sort_values()
fr = json.load(open("tables/friedman.json")); CD = fr["cd"]; k = len(avg)

# ---------------------------------------------------------------- Fig 3: critical-difference diagram
fig, ax = plt.subplots(figsize=(W, 3.6)); ax.set_xlim(0.2, k + 0.8); ax.set_ylim(0, 1); ax.axis("off")
ax.hlines(0.86, 1, k, color="k", lw=1)
for t in range(1, k + 1):
    ax.vlines(t, 0.845, 0.875, color="k", lw=1); ax.text(t, 0.9, str(t), ha="center", fontsize=9)
ax.hlines(0.98, 1, 1 + CD, color="k", lw=2.5); ax.text(1 + CD / 2, 1.0, f"CD = {CD:.2f}", ha="center", fontsize=9.5)
half = (k + 1) // 2
for i, (mn, rr) in enumerate(avg.items()):
    left = i < half; j = i if left else k - 1 - i; y = 0.76 - j * 0.105
    xt = 0.75 if left else k + 0.25
    hi = mn == "BAMAKAN"
    ax.plot([rr, rr, xt], [0.86, y, y], color=RED if hi else "0.35", lw=1.4 if hi else 1)
    ax.text(xt - 0.05 if left else xt + 0.05, y, f"{NAME.get(mn, mn)} ({rr:.2f})", ha="right" if left else "left", va="center",
            fontsize=9.5, fontweight="bold" if hi else "normal", color=RED if hi else "k")
plt.savefig(f"{O}/fig3_cd.png"); plt.close()

# ---------------------------------------------------------------- Fig 4: normalised rank per dataset (horizontal bars)
pos = []
for task, (met, sg) in PRIM.items():
    P = mean[mean.task == task].pivot(index="dataset", columns="model", values=met)
    P = P.drop(columns=[c for c in ["Border-frozen", "EBM"] if c in P])
    Rk = (sg * P).rank(axis=1, ascending=False)
    for ds in Rk.index:
        kk = Rk.loc[ds].notna().sum(); pos.append(dict(dataset=ds, task=task, nr=(Rk.loc[ds, "BAMAKAN"] - 1) / (kk - 1)))
pos = pd.DataFrame(pos)
tier = lambda d: "Clinical" if d in CLIN else ("Monotone" if d in MONO_DS else ("Held-out" if d == "german" else "Development"))
pos["tier"] = pos.dataset.map(tier); order = ["Development", "Held-out", "Monotone", "Clinical"]
pos["o"] = pos.tier.map({t: i for i, t in enumerate(order)})
pos = pos.sort_values(["o", "nr"], ascending=[False, False]).reset_index(drop=True)
cols = {"binary": RED, "ordinal": BLUE, "nominal": GREEN}
fig, ax = plt.subplots(figsize=(W, 7.6))
ax.barh(range(len(pos)), np.maximum(pos.nr, 0.008), color=[cols[t] for t in pos.task], height=0.75)
ax.set_yticks(range(len(pos)), pos.dataset, fontsize=9); ax.set_xlim(0, 1)
ax.set_xlabel("Normalised rank of BAMAKAN (0 = best, 1 = worst)"); ax.axvline(0.5, ls=":", color=G, lw=1)
tiers = pos.tier.values
for i in range(1, len(pos)):
    if tiers[i] != tiers[i - 1]: ax.axhline(i - 0.5, color="k", lw=0.7)
for t in order:
    idx = np.where(tiers == t)[0]
    if len(idx): ax.text(1.01, idx.mean(), t, va="center", fontsize=9.5, fontweight="bold", transform=ax.get_yaxis_transform())
ax.set_ylim(-0.6, len(pos) - 0.4)
ax.legend(handles=[Patch(color=c, label=t) for t, c in cols.items()], frameon=False, loc="lower right")
plt.savefig(f"{O}/fig4_ranks.png"); plt.close()

# ---------------------------------------------------------------- Fig 5 (file fig6_pareto): accuracy vs size
Ef = pd.DataFrame(json.load(open("results/efficiency/efficiency.json"))); Ef = Ef[Ef.fit_s.notna()]
sz = Ef.groupby("model").pickle_bytes.median() / 1024
fig, ax = plt.subplots(figsize=(W, 4.6))
off = {"EBM-main": (6, -12), "LogReg": (6, 4), "CatBoost": (6, 4), "SVM": (6, -12), "RealMLP": (6, 4), "RF": (-14, 8), "XGBoost": (6, -12),
       "GMLVQ": (6, -12), "LightGBM": (6, 4), "EBM": (-10, 8)}
for m in avg.index:
    if m not in sz: continue
    hi = m == "BAMAKAN"
    ax.scatter(sz[m], avg[m], s=110 if hi else 45, color=RED if hi else G, zorder=3, edgecolor="k" if hi else "none")
    ax.annotate(NAME.get(m, m), (sz[m], avg[m]), xytext=off.get(m, (6, 4)), textcoords="offset points", fontsize=9.5,
                fontweight="bold" if hi else "normal", color=RED if hi else "k")
ax.set_xscale("log"); ax.set_xlabel("Stored model size (KB, median over 8 datasets, log scale)")
ax.set_ylabel("Average rank over 30 datasets\n(lower is better)"); ax.invert_yaxis(); ax.grid(alpha=0.25)
plt.savefig(f"{O}/fig5_size_vs_rank.png"); plt.close()

# ---------------------------------------------------------------- Fig 6 (file fig8_clinical): calibration + decision curve
C = json.load(open("results/clinical/clinical.json"))
mc = {"BAMAKAN": RED, "EBM-main": BLUE, "LogReg": PURPLE, "CatBoost": GREEN}
fig = plt.figure(figsize=(W, 6.6)); gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)
for a, ds, tt in [(fig.add_subplot(gs[0, 0]), "heartfail", "(a) Heart Failure: death"), (fig.add_subplot(gs[0, 1]), "hcv", "(b) HCV: any liver disease")]:
    cc = C[ds]["C_calibration"]
    for m, c in mc.items():
        a.plot(cc[m]["pred"], cc[m]["obs"], "o-", color=c, ms=4, lw=1.5 if m == "BAMAKAN" else 1.1, label=f"{m} ({cc[m]['ece']:.3f})")
    a.plot([0, 1], [0, 1], ":", color=G); a.set_xlabel("Predicted probability"); a.set_ylabel("Observed frequency")
    a.set_title(tt); a.legend(frameon=False, title="Model (ECE)", title_fontsize=8.5, loc="lower right", fontsize=8)
a = fig.add_subplot(gs[1, :]); cc = C["heartfail"]["C_calibration"]
for m, c in mc.items():
    nb = np.array(cc[m]["net_benefit"]); a.plot(nb[:, 0], nb[:, 1], color=c, lw=2 if m == "BAMAKAN" else 1.4, label=m)
ta = np.array(cc["treat_all"]); a.plot(ta[:, 0], ta[:, 1], color=G, ls="--", lw=1.2, label="Treat all")
a.axhline(0, color="k", lw=0.8, label="Treat none"); a.set_ylim(-0.05, 0.33)
a.set_xlabel("Threshold probability"); a.set_ylabel("Net benefit"); a.set_title("(c) Heart Failure: decision curve")
a.legend(frameon=False, ncol=3)
plt.savefig(f"{O}/fig6_clinical.png"); plt.close()

# ---------------------------------------------------------------- Fig 7: explanations on Heart Failure
import torch; torch.set_num_threads(2)
from datasets import DATASETS; from benchmark import make_models
X, y, task, f = DATASETS["heartfail"](); m = make_models(task, 0)["BAMAKAN"]().fit(X, y)
imp = m.feature_importances(X); cont = [j for j in range(len(f)) if len(np.unique(X[:, j])) > 2]
top = sorted(cont, key=lambda j: -imp[j])[:4]; tau = float(torch.exp(m.net_.log_tau))
phi = m.explain(X) * tau; bo = pd.DataFrame(m.borders_in_original_units(f))
unit = {"age": "years", "ejection_fraction": "%", "serum_creatinine": "mg/dL", "serum_sodium": "mEq/L", "creatinine_phosphokinase": "mcg/L", "platelets": "kiloplatelets/mL"}
fig = plt.figure(figsize=(W, 8.6))
L, Wd, Hh = 0.11, 0.37, 0.17
axs = np.array([fig.add_axes([L + c * (Wd + 0.13), 0.80 - r * 0.27, Wd, Hh]) for r in range(2) for c in range(2)])
for n_, j in enumerate(top):
    a = axs.flat[n_]; o = np.argsort(X[:, j])
    a.plot(X[o, j], phi[o, j], color=RED, lw=2); a.axhline(0, color="k", lw=0.6)
    for v in bo[bo.feature == f[j]].value:
        if X[:, j].min() <= v <= X[:, j].max(): a.axvline(v, ls=":", color=G, lw=1.1)
    a.set_title(f"({'abcd'[n_]}) {f[j].replace('_', ' ')}"); a.set_xlabel(unit.get(f[j], "original units"))
    a.set_ylabel("Contribution (logit)")
    if f[j] == "creatinine_phosphokinase": a.set_xscale("log")
a = fig.add_axes([0.36, 0.04, 0.61, 0.33])
i = int(np.argmax(m.predict_proba(X)[:, 1])); c = phi[i]; oo = np.argsort(np.abs(c))[::-1][:8]
a.barh(range(len(oo)), c[oo], color=[RED if v > 0 else BLUE for v in c[oo]])
a.set_yticks(range(len(oo)), [f"{f[q].replace('_', ' ')} = {X[i, q]:g}" for q in oo], fontsize=9.5); a.invert_yaxis()
a.axvline(0, color="k", lw=0.6); a.set_xlabel("Contribution to the logit of death")
a.set_title(f"(e) Patient with the highest predicted risk, P(death) = {m.predict_proba(X[i:i + 1])[0, 1]:.2f}")
plt.savefig(f"{O}/fig7_explanation.png"); plt.close()
print("figures written")

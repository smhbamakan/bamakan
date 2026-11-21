"""Tables of the manuscript. Run from the repository root: python analysis/make_tables.py, then
python analysis/make_figures.py (final figures)."""
import json, glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

T, F = "tables", "figures"
os.makedirs(T, exist_ok=True)
RED, G, BLUE = "#c0392b", "#7f8c8d", "#2e86c1"
EXCL = {"maternal"}  # excluded on a data-quality criterion fixed before modelling (55% duplicate rows)
CLIN = ["heartfail", "hcv", "cirrhosis"]
MONO_DS = ["era", "esl", "lev", "swd"]
MAIN = ["BAMAKAN", "TabPFN", "EBM-main", "LogReg", "CatBoost", "XGBoost", "LightGBM", "RF", "SVM", "RealMLP", "MLP", "GMLVQ",
        "kNN", "NearestCentroid"]
PRIM = {"binary": ("auc", 1, 0.01), "nominal": ("acc", 1, 0.01), "ordinal": ("mae", -1, 0.02)}
NAME = {"NearestCentroid": "Nearest centroid", "Border-frozen": "Frozen-border rule"}


def md(df, path, floatfmt=3):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    open(path, "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ load repeated-CV results
rows = []
for f in glob.glob("results/cv/*.json"):
    rows += json.load(open(f))
rows += [json.load(open(f)) for f in glob.glob("results/cv_tabpfn/*.json")]
df = pd.DataFrame(rows)
df = df[(df.repeat < 2) & df["error"].isna() & ~df.dataset.isin(EXCL)]
df = df[~(df.model.isin(["EBM", "EBM-main"]) & (df.repeat > 0))]
mean = df.groupby(["dataset", "task", "model"]).mean(numeric_only=True).reset_index()
nds = mean.dataset.nunique()
print("datasets:", nds)


def table(task, models=None):
    met, sgn, _ = PRIM[task]
    P = mean[mean.task == task].pivot(index="dataset", columns="model", values=met)
    return (P if models is None else P[[m for m in models if m in P]]), sgn


# ------------------------------------------------------------------ Table 4: overall ranks (14 models defined on all tasks)
S = pd.concat([table(t, MAIN)[0] * table(t)[1] for t in PRIM]).dropna(axis=1)
R = S.rank(axis=1, ascending=False)
fr = friedmanchisquare(*[S[c] for c in S.columns])
k, n = S.shape[1], S.shape[0]
q05 = {14: 3.354, 15: 3.391, 13: 3.313}[k]
CD = q05 * np.sqrt(k * (k + 1) / (6 * n))
rng = np.random.RandomState(0)
boot = np.array([R.iloc[rng.randint(0, n, n)].mean().values for _ in range(5000)])
tab4 = pd.DataFrame({"Model": [NAME.get(c, c) for c in R.columns], "Average rank": R.mean().values,
                     "95% CI": [f"{a:.2f}–{b:.2f}" for a, b in zip(np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0))],
                     "Top-3 datasets": (R <= 3).sum().values}).sort_values("Average rank")
md(tab4, f"{T}/tab_ranks.md", 2)
print(f"Friedman chi2={fr.statistic:.1f} p={fr.pvalue:.2g} CD={CD:.2f} n={n} k={k}")
json.dump(dict(chi2=fr.statistic, p=fr.pvalue, cd=CD, n=n, k=k), open(f"{T}/friedman.json", "w"), default=float)

# CD diagram
avg = R.mean().sort_values()
fig, ax = plt.subplots(figsize=(9, 3.2)); lo, hi = 1, k
ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(0, 1); ax.axis("off"); ax.hlines(0.8, lo, hi, color="k", lw=1)
for t in range(lo, hi + 1):
    ax.vlines(t, 0.78, 0.82, color="k", lw=1); ax.text(t, 0.86, str(t), ha="center", fontsize=8)
ax.hlines(0.95, lo, lo + CD, color="k", lw=2); ax.text(lo + CD / 2, 0.97, f"CD = {CD:.2f}", ha="center", fontsize=8)
half = (len(avg) + 1) // 2
for i, (mn, r) in enumerate(avg.items()):
    left = i < half; y = 0.7 - (i if left else len(avg) - 1 - i) * 0.6 / half; xt = lo - 0.2 if left else hi + 0.2
    ax.plot([r, r, xt], [0.8, y, y], color=RED if mn == "BAMAKAN" else "0.3", lw=1)
    ax.text(xt, y, f"{NAME.get(mn, mn)} ({r:.2f})", ha="right" if left else "left", va="center", fontsize=8,
            fontweight="bold" if mn == "BAMAKAN" else "normal")
plt.tight_layout(); plt.savefig(f"{F}/fig3_cd.png", dpi=300); plt.close()

# ------------------------------------------------------------------ Table 5: per task family, ranks + Bayesian signed-rank (ROPE)


def bsr(x, rope, n=20000, seed=0):
    r = np.random.RandomState(seed); z = np.concatenate([[0.0], x]); m = len(z)
    Ssum = z[:, None] + z[None, :]; lo_, hi_ = (Ssum < -2 * rope), (Ssum > 2 * rope)
    w = r.dirichlet(np.ones(m), n)
    pl = np.einsum("ni,ij,nj->n", w, lo_, w); pr = np.einsum("ni,ij,nj->n", w, hi_, w); pe = 1 - pl - pr
    win = np.argmax(np.stack([pr, pe, pl], 1), 1)
    return [(win == j).mean() for j in range(3)]


fam_rows, bayes_rows = [], []
for task in PRIM:
    P, sgn = table(task)
    P = P.drop(columns=[c for c in ["Border-frozen", "EBM"] if c in P])
    Rk = (sgn * P).rank(axis=1, ascending=False).mean()
    met, _, rope = PRIM[task]
    for c in P.columns:
        if c == "BAMAKAN": continue
        x = (sgn * (P["BAMAKAN"] - P[c])).dropna().values
        b, e, w = bsr(x, rope)
        p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
        bayes_rows.append(dict(task=task, model=c, n=len(x), mean_diff=x.mean(), p_wilcoxon=p, P_better=b, P_rope=e, P_worse=w,
                               rank_model=Rk[c], rank_bamakan=Rk["BAMAKAN"]))
B = pd.DataFrame(bayes_rows)
B["p_holm"] = np.nan
for task in PRIM:
    i = B.task == task; B.loc[i, "p_holm"] = multipletests(B.loc[i, "p_wilcoxon"], method="holm")[1]
B.to_csv(f"{T}/bayes_all.csv", index=False)
KEY = ["TabPFN", "EBM-main", "GAM-mono", "XGB-mono", "POLR", "CatBoost", "LogReg", "XGBoost", "RealMLP"]
tb = B[B.model.isin(KEY)].copy()
tb["Task (metric, n)"] = tb.apply(lambda r: f"{r.task} ({PRIM[r.task][0].upper()}, {r.n})", axis=1)
tb["Competitor"] = tb.model
tb["Rank B / C"] = tb.apply(lambda r: f"{r.rank_bamakan:.2f} / {r.rank_model:.2f}", axis=1)
tb["Mean diff."] = tb.mean_diff
tb["p (Holm)"] = tb.p_holm
tb["P(B better)"] = tb.P_better; tb["P(equal)"] = tb.P_rope; tb["P(C better)"] = tb.P_worse
tb = tb[["Task (metric, n)", "Competitor", "Rank B / C", "Mean diff.", "p (Holm)", "P(B better)", "P(equal)", "P(C better)"]]
md(tb, f"{T}/tab_bayes.md", 3)
for task in PRIM:
    P, sgn = table(task); P = P.drop(columns=[c for c in ["Border-frozen", "EBM"] if c in P])
    print(task, "ranks:", (sgn * P).rank(axis=1, ascending=False).mean().sort_values().round(2).head(6).to_dict(), len(P.columns))

# Frozen-border ablation (all tasks)
abl = []
for task in PRIM:
    P, sgn = table(task)
    x = sgn * (P["BAMAKAN"] - P["Border-frozen"]); abl.append((task, len(x), x.mean(), (x > 0).sum(), wilcoxon(x).pvalue))
json.dump(abl, open(f"{T}/frozen.json", "w"), default=float)
print("frozen", abl)

# ------------------------------------------------------------------ Figure 4: normalised rank per dataset
pos = []
for task in PRIM:
    P, sgn = table(task); P = P.drop(columns=[c for c in ["Border-frozen", "EBM"] if c in P])
    Rk = (sgn * P).rank(axis=1, ascending=False)
    for ds in Rk.index:
        kk = Rk.loc[ds].notna().sum()
        pos.append(dict(dataset=ds, task=task, nr=(Rk.loc[ds, "BAMAKAN"] - 1) / (kk - 1), k=kk))
pos = pd.DataFrame(pos)
tier = lambda d: "Clinical" if d in CLIN else ("Monotone" if d in MONO_DS else ("Held-out" if d == "german" else "Development"))
pos["tier"] = pos.dataset.map(tier)
order = ["Development", "Held-out", "Monotone", "Clinical"]
pos = pos.sort_values(["tier", "nr"], key=lambda s: s.map({t: i for i, t in enumerate(order)}) if s.name == "tier" else s)
fig, ax = plt.subplots(figsize=(10, 3.2))
cols = {"binary": RED, "ordinal": BLUE, "nominal": "#27ae60"}
ax.bar(range(len(pos)), np.maximum(pos.nr, 0.012), color=[cols[t] for t in pos.task])
tiers = pos.tier.values
for i in range(1, len(pos)):
    if tiers[i] != tiers[i - 1]: ax.axvline(i - 0.5, color="k", lw=0.6)
for t in order:
    idx = np.where(tiers == t)[0]
    if len(idx): ax.text(idx.mean(), 1.02, t, ha="center", fontsize=8, transform=ax.get_xaxis_transform())
ax.set_xticks(range(len(pos)), pos.dataset, rotation=70, fontsize=7); ax.set_ylabel("Normalised rank\n(0 = best, 1 = worst)")
ax.axhline(0.5, ls=":", color=G, lw=0.8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=t) for t, c in cols.items()], fontsize=7, frameon=False, ncol=3, loc="upper left")
ax.set_ylim(0, 1); ax.set_xlim(-0.6, len(pos) - 0.4)
plt.tight_layout(); plt.savefig(f"{F}/fig4_ranks.png", dpi=300); plt.close()

# ------------------------------------------------------------------ Table A1: primary metric per dataset (all models)
for task in PRIM:
    P, sgn = table(task)
    P = P.round(3); P.insert(0, "Dataset", P.index)
    md(P.reset_index(drop=True), f"{T}/tabA_{task}.md", 3)

# ------------------------------------------------------------------ Table 6: ablation
A = pd.DataFrame(json.load(open("results/ablation/all.json")))
A = A[~A.dataset.isin(EXCL)]
Am = A.groupby(["dataset", "task", "variant"]).mean(numeric_only=True).reset_index()
lab = {"B1": "One border (B = 1)", "B2": "Two borders (B = 2)", "B8": "Eight borders (B = 8)", "kappa_fixed": "κ fixed at 3",
       "kappa_global": "One κ shared by all comparisons", "kmax20": "κ_max = 20", "kmax50": "κ_max = 50", "kmax100": "κ_max = 100",
       "kmax500": "κ_max = 500", "dir_soft": "Learned soft directions (v1.0)", "norm_rank": "Rank (ECDF) normalisation",
       "delta0": "No extrapolation margin (δ = 0)", "delta0.25": "δ = 0.25", "delta1": "δ = 1", "free_borders": "Unordered (free) exponents",
       "l1_1e-3": "L1 on weights, 10⁻³", "l1_1e-2": "L1 on weights, 10⁻²"}
ab = []
for v in lab:
    xs = []
    for task, (met, sgn, _) in PRIM.items():
        P = Am[Am.task == task].pivot(index="dataset", columns="variant", values=met)
        xs += list(sgn * (P[v] - P["default"]))
    x = np.array(xs)
    ab.append({"Variant": lab[v], "Mean Δ": x.mean(), "Better / worse": f"{(x > 1e-9).sum()} / {(x < -1e-9).sum()}",
               "p (Wilcoxon)": wilcoxon(x).pvalue if np.any(x != 0) else 1.0,
               "Parameters (median)": Am[Am.variant == v].n_params.median(), "Fit time (s, median)": Am[Am.variant == v].fit_s.median()})
ab = pd.DataFrame(ab)
ab["p (Holm)"] = multipletests(ab["p (Wilcoxon)"], method="holm")[1]
ab = ab[["Variant", "Mean Δ", "Better / worse", "p (Wilcoxon)", "p (Holm)", "Parameters (median)"]]
md(ab, f"{T}/tab_ablation.md", 4)
print("default params", Am[Am.variant == "default"].n_params.median())

# ------------------------------------------------------------------ Table 7: synthetic stress tests
Sx = pd.DataFrame(json.load(open("results/synthetic_ext.json")))
Sx["Factor"] = Sx.scenario.map({"S5a": "Correlation ρ", "S5b": "Interaction γ", "S5c": "Covariate shift δ"})
Sx["Level"] = [r["rho"] if r["scenario"] == "S5a" else (r["gamma"] if r["scenario"] == "S5b" else r["delta"]) for r in Sx.to_dict("records")]
pv = Sx.pivot_table(index=["Factor", "Level"], columns="model", values="auc").reset_index()
dr = Sx[Sx.model == "BAMAKAN"].groupby(["Factor", "Level"]).dir_recovered.mean().values
pv["Directions recovered"] = dr
pv = pv[["Factor", "Level", "BAMAKAN", "EBM-main", "XGB-mono", "XGBoost", "LogReg", "Directions recovered"]]
md(pv, f"{T}/tab_synth.md", 3)
# partial directions (Section 6.3)
Pd = pd.DataFrame([json.load(open(f)) for f in glob.glob("results/partial_dir/*.json")])
Pd = Pd[~Pd.dataset.isin(EXCL)]
pdres = []
for task in ["binary", "ordinal"]:
    met, sgn, _ = PRIM[task]
    a = Pd[Pd.task == task].groupby("dataset")[met].mean(); b = mean[(mean.task == task) & (mean.model == "BAMAKAN")].set_index("dataset")[met]
    x = sgn * (a - b).dropna(); pdres.append((task, len(x), x.mean(), int((x > 1e-6).sum()), int((x < -1e-6).sum()), wilcoxon(x).pvalue))
json.dump(pdres, open(f"{T}/partial.json", "w")); print("partial", pdres)

# ------------------------------------------------------------------ Table 8: efficiency
E = pd.read_csv("results/efficiency/efficiency_summary.csv")
E = E.rename(columns={"model": "Model", "fit_s": "Fit (s)", "infer_ms": "Inference (ms / 1,000 cases)", "kb": "Stored model (KB)",
                      "rss": "Peak memory increase (MB)", "n": "Datasets"})
E["Model"] = E["Model"].map(lambda m: NAME.get(m, m))
md(E.drop(columns=["Datasets"]), f"{T}/tab_eff.md", 2)
Ef = pd.DataFrame(json.load(open("results/efficiency/efficiency.json")))
Ef = Ef[Ef.fit_s.notna()]
ratio = Ef.pivot(index="dataset", columns="model", values="pickle_bytes")
tr = Ef.pivot(index="dataset", columns="model", values="fit_s")
json.dump({"size_ratio_EBM": float((ratio.EBM / ratio.BAMAKAN).median()), "size_ratio_EBMmain": float((ratio["EBM-main"] / ratio.BAMAKAN).median()),
           "time_ratio_EBM": float((tr.EBM / tr.BAMAKAN).median()), "time_ratio_EBMmain": float((tr["EBM-main"] / tr.BAMAKAN).median())},
          open(f"{T}/eff_ratios.json", "w"))
print(json.load(open(f"{T}/eff_ratios.json")))

# Figure 6: accuracy-size trade-off
sz = Ef.groupby("model").pickle_bytes.median() / 1024
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for m in R.columns:
    if m not in sz: continue
    hi_ = m == "BAMAKAN"
    ax.scatter(sz[m], avg[m], s=80 if hi_ else 35, color=RED if hi_ else G, zorder=3)
    ax.annotate(NAME.get(m, m), (sz[m], avg[m]), xytext=(5, 3), textcoords="offset points", fontsize=7,
                fontweight="bold" if hi_ else "normal")
ax.set_xscale("log"); ax.set_xlabel("Stored model size (KB, median over 8 datasets, log scale)")
ax.set_ylabel(f"Average rank over {n} datasets\n(lower is better)"); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(f"{F}/fig5_size_vs_rank.png", dpi=300); plt.close()

# ------------------------------------------------------------------ Tables 10-11: clinical
C = json.load(open("results/clinical/clinical.json"))
crow = []
for ds in CLIN:
    r = C[ds]; key = "auc" if r["task"] == "binary" else "mae"
    for m in ["BAMAKAN", "EBM-main", "GAM-mono", "XGB-mono", "CatBoost", "LogReg"]:
        a = r["A_ci"][m]
        rec = r["B_recall"].get(m)
        row = {"Dataset": ds, "Model": m, key.upper() + " (95% CI)": f"{a[key][0]:.3f} ({a[key][1]:.3f}–{a[key][2]:.3f})",
               "bAcc (95% CI)": f"{a['bacc'][0]:.3f} ({a['bacc'][1]:.3f}–{a['bacc'][2]:.3f})"}
        if r["task"] == "ordinal":
            row["QWK"] = f"{a['qwk'][0]:.3f}"
        crow.append(row)
CT = pd.DataFrame(crow).fillna("")
CT["Metric (95% CI)"] = CT.get("AUC (95% CI)", "").astype(str) + CT.get("MAE (95% CI)", "").astype(str)
CT = CT[["Dataset", "Model", "Metric (95% CI)", "bAcc (95% CI)", "QWK"]]
md(CT, f"{T}/tab_clin.md")
# recall + conformal
cr = []
for ds in CLIN:
    r = C[ds]
    for m in ["BAMAKAN", "EBM-main", "CatBoost"]:
        f_ = r["F_conformal"][m]
        cr.append({"Dataset": ds, "Model": m, "Recall per class": " / ".join(f"{v:.2f}" for v in r["B_recall"][m]),
                   "Class-conditional coverage (α = 0.1)": " / ".join(f"{v:.2f}" for v in f_["coverage_per_class"]),
                   "Mean set size": f"{f_['mean_set_size']:.2f}", "ECE": f"{r['C_calibration'][m]['ece']:.3f}" if m in r["C_calibration"] else ""})
md(pd.DataFrame(cr), f"{T}/tab_clin2.md")
# scorecard fidelity + card
fd = []
for ds in CLIN + ["maternal"]:
    g = C[ds]["G_fidelity"]; key = "AUC" if C[ds]["task"] == "binary" else "MAE"
    fd.append({"Dataset": ds, "Metric": key, "Full model": f"{g['full'][0]:.3f}", "Card, sharp limit": f"{g['card'][0]:.3f}",
               "Card, refitted points": f"{g['card_refit'][0]:.3f}", "Agreement with full model": f"{g['agree_refit'][0]:.2f}",
               "Rules (mean)": f"{g['n_rules'][0]:.1f}"})
md(pd.DataFrame(fd), f"{T}/tab_fid.md")
card = C["heartfail"]["G_card"]
unit = {"age": "years", "ejection_fraction": "%", "serum_creatinine": "mg/dL", "serum_sodium": "mEq/L", "creatinine_phosphokinase": "mcg/L",
        "platelets": "kiloplatelets/mL"}
cd_rows = [{"Criterion": x["feature"].replace("_", " "), "Rule": f"{x['op']} {x['value']:.3g}" + (f" {unit.get(x['feature'], '')}" if x['op'] != '=' else ""),
            "Points": x["points"]} for x in sorted(card["rules"], key=lambda z: (z["feature"], z["value"]))]
md(pd.DataFrame(cd_rows), f"{T}/tab_card.md")
json.dump({"unit": card["unit_logit_per_point"], "cut": card["cutpoints"]}, open(f"{T}/card_meta.json", "w"), default=float)
# stability summary
st = {ds: {k: v["direction_agreement"] for k, v in C[ds]["D_stability"].items()} for ds in CLIN}
json.dump(st, open(f"{T}/stability.json", "w"), indent=1)

# Figure 7: calibration (3 datasets) + decision curve (Heart Failure)
CLIN_CAL = ["heartfail", "hcv"]
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
mc = {"BAMAKAN": RED, "EBM-main": BLUE, "LogReg": "#8e44ad", "CatBoost": "#27ae60"}
for a, ds in zip(ax[:2], CLIN_CAL):
    cc = C[ds]["C_calibration"]
    for m, c in mc.items():
        a.plot(cc[m]["pred"], cc[m]["obs"], "o-", color=c, ms=3, lw=1.2, label=f"{m} (ECE {cc[m]['ece']:.3f})")
    a.plot([0, 1], [0, 1], ":", color=G); a.set_xlabel("Predicted probability", fontsize=8); a.set_title(
        {"heartfail": "Heart Failure: death", "hcv": "HCV: any liver disease", "cirrhosis": "Cirrhosis: stage ≥ 2"}[ds], fontsize=9)
    a.legend(fontsize=6, frameon=False)
ax[0].set_ylabel("Observed frequency", fontsize=8)
cc = C["heartfail"]["C_calibration"]
for m, c in mc.items():
    nb = np.array(cc[m]["net_benefit"]); ax[2].plot(nb[:, 0], nb[:, 1], color=c, lw=1.4, label=m)
ta = np.array(cc["treat_all"]); ax[2].plot(ta[:, 0], ta[:, 1], color=G, ls="--", lw=1, label="Treat all")
ax[2].axhline(0, color="k", lw=0.6, label="Treat none"); ax[2].set_ylim(-0.05, 0.35)
ax[2].set_xlabel("Threshold probability", fontsize=8); ax[2].set_ylabel("Net benefit", fontsize=8)
ax[2].set_title("Heart Failure: decision curve", fontsize=9); ax[2].legend(fontsize=6, frameon=False)
plt.tight_layout(); plt.savefig(f"{F}/fig6_clinical.png", dpi=300); plt.close()

# ------------------------------------------------------------------ dataset table (update tiers)
DT = open("analysis/datasets_table.md").read().replace("| Fixed test | german |", "| Held-out | german |").replace(
    "| Fixed test | nslkdd", "| Fixed test set | nslkdd")
open(f"{T}/tab_datasets.md", "w").write(DT)
# ESL fold counts
print("done")

# ------------------------------------------------------------------ Table A5: EBM with interactions vs main effects
ea = []
for task, (met, sgn, _) in PRIM.items():
    P, _s = table(task)
    for ds in P.index:
        ea.append({"Dataset": ds, "Task": task, "Metric": met.upper(), "EBM": P.loc[ds, "EBM"], "EBM-main": P.loc[ds, "EBM-main"],
                   "BAMAKAN": P.loc[ds, "BAMAKAN"]})
md(pd.DataFrame(ea), f"{T}/tabA_ebm.md", 3)

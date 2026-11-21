"""Bayesian signed-rank tests (with ROPE), bootstrap CIs of average ranks, full result tables."""
import json, glob, numpy as np, pandas as pd
rows = []
for f in glob.glob("results/cv/*.json"):
    rows += json.load(open(f))
df = pd.DataFrame(rows)
df = df[(df.repeat < 2) & df["error"].isna()]
df = df[~(df.model.isin(["EBM", "EBM-main"]) & (df.repeat > 0))]
prim = {"binary": ("auc", 1, 0.01), "nominal": ("acc", 1, 0.01), "ordinal": ("mae", -1, 0.02)}
mean = df.groupby(["dataset", "task", "model"]).mean(numeric_only=True).reset_index()
REF = "BAMAKAN"


def bayes_signed_rank(x, rope, n=20000, seed=0):
    """Benavoli et al. (2017, JMLR) Bayesian signed-rank test with a Dirichlet process prior (pseudo-observation at 0).
    x: per-dataset differences (REF - other, sign-adjusted so > 0 favours REF). Returns P(REF better), P(rope), P(other better)."""
    rng = np.random.RandomState(seed)
    z = np.concatenate([[0.0], x]); m = len(z)
    S = z[:, None] + z[None, :]
    lo, hi = (S < -2 * rope), (S > 2 * rope)
    w = rng.dirichlet(np.ones(m), n)
    pl = np.einsum("ni,ij,nj->n", w, lo, w); pr = np.einsum("ni,ij,nj->n", w, hi, w); pe = 1 - pl - pr
    win = np.argmax(np.stack([pr, pe, pl], 1), 1)
    return [(win == k).mean() for k in range(3)]


out = []
for task, (met, sgn, rope) in prim.items():
    P = mean[mean.task == task].pivot(index="dataset", columns="model", values=met)
    for c in P.columns.drop(REF):
        x = (sgn * (P[REF] - P[c])).dropna().values
        b, e, w = bayes_signed_rank(x, rope)
        out.append(dict(task=task, metric=met, rope=rope, model=c, n=len(x), mean_diff=x.mean(),
                        p_bamakan_better=b, p_rope=e, p_other_better=w))
B = pd.DataFrame(out); B.to_csv("tables/bayes_all_with_ebm.csv", index=False)
print(B.round(3).to_string())

# bootstrap CI of average ranks over datasets (models defined on all tasks)
P = pd.concat([sgn * mean[mean.task == t].pivot(index="dataset", columns="model", values=m)
               for t, (m, sgn, _) in prim.items()]).dropna(axis=1)
R = P.rank(axis=1, ascending=False)
rng = np.random.RandomState(0)
boot = np.array([R.iloc[rng.randint(0, len(R), len(R))].mean().values for _ in range(5000)])
CI = pd.DataFrame({"avg_rank": R.mean(), "ci_lo": np.percentile(boot, 2.5, 0), "ci_hi": np.percentile(boot, 97.5, 0),
                   "p_best": pd.Series((boot == boot.min(1, keepdims=True)).mean(0), index=R.columns)}).sort_values("avg_rank")
CI.to_csv("tables/rank_ci_with_ebm.csv"); print("\n", CI.round(3).to_string())

# full tables: mean (sd over folds) for every dataset, model and main metric
sd = df.groupby(["dataset", "task", "model"]).std(numeric_only=True).reset_index()
for met in ["auc", "acc", "bacc", "f1_macro", "mae", "qwk", "logloss", "brier", "ece"]:
    if met not in mean: continue
    M = mean.pivot(index="dataset", columns="model", values=met); S = sd.pivot(index="dataset", columns="model", values=met)
    T = M.round(3).astype(str) + " (" + S.round(3).astype(str) + ")"
    T = T.where(M.notna(), "")
    T.dropna(how="all").to_csv(f"tables/full_{met}.csv")

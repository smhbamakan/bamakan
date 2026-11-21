"""Simple interface for users who are not machine-learning specialists.

    from bamakan import BAMAKAN, prepare
    X, y, names, labels = prepare(df, target="outcome")
    model = BAMAKAN(task="binary").fit(X, y)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import BAMAKANClassifier

#: Configuration used in the paper (fixed before the experiments, not tuned per dataset).
PAPER_CONFIG = dict(direction="corr", n_borders=4, kappa_mode="local", psi="normed", kappa0=3.0, p_init="quantile",
                    lr_schedule="cosine", optimizer="hybrid", lbfgs_block=20, epochs=600, patience=60)


def BAMAKAN(task="auto", compact=False, random_state=0, **overrides):
    """A BAMAKAN classifier with the configuration of the paper.

    task: "binary", "ordinal" (ordered classes, e.g. low < medium < high), "nominal" (unordered classes) or "auto".
    compact: True uses one border per criterion (about three times fewer parameters); recommended for datasets with
        fewer than about 2,000 cases, where it is as accurate as the default.
    overrides: any other argument of BAMAKANClassifier."""
    cfg = dict(PAPER_CONFIG)
    if compact:
        cfg["n_borders"] = 1
    cfg.update(overrides)
    return BAMAKANClassifier(task=task, random_state=random_state, **cfg)


def prepare(df: pd.DataFrame, target: str, drop=(), class_order=None):
    """Turn a table into model inputs.

    Rows with missing values are removed; text columns with two values become 0/1, other text columns become one-hot
    indicators. class_order lists the classes from lowest to highest for ordinal outcomes (e.g. ["low", "mid",
    "high"]); otherwise classes are sorted. Returns X (float array), y (integer codes 0..K-1), the criterion names and
    the class labels in code order."""
    df = df.drop(columns=[c for c in drop if c in df.columns]).dropna().reset_index(drop=True)
    yraw = df.pop(target)
    for c in list(df.columns):
        if df[c].dtype == object or str(df[c].dtype) in ("category", "bool"):
            vals = sorted(df[c].astype(str).unique())
            if len(vals) == 2:
                df[c] = (df[c].astype(str) == vals[1]).astype(float)
                df = df.rename(columns={c: f"{c}={vals[1]}"})
            else:
                df = pd.get_dummies(df, columns=[c], dtype=float)
    labels = list(class_order) if class_order is not None else sorted(yraw.astype(str).unique(), key=_natural)
    code = {str(v): i for i, v in enumerate(labels)}
    unknown = set(yraw.astype(str)) - set(code)
    if unknown:
        raise ValueError(f"classes not in class_order: {sorted(unknown)}")
    y = yraw.astype(str).map(code).values.astype(int)
    return df.astype(float).values, y, list(df.columns), labels


def _natural(v):
    try:
        return (0, float(v), "")
    except ValueError:
        return (1, 0.0, v)


def report(model, X, y, names, labels=None):
    """Plain-language tables of a fitted model: criterion importance and direction, borders in original units."""
    imp = model.feature_importances(X)
    s = model.net_.s_full().detach().numpy()[0, :, 0]
    top = "higher class" if model.task_ != "nominal" else "the predicted class"
    T = pd.DataFrame({"criterion": names, "importance": np.round(imp, 3),
                      "effect": [f"larger value -> {top}" if v >= 0.5 else f"smaller value -> {top}" for v in s]})
    T = T.sort_values("importance", ascending=False).reset_index(drop=True)
    B = None
    if model.shape_ == "monotone":
        B = pd.DataFrame(model.borders_in_original_units(names))
        if len(B):
            binary = {n for j, n in enumerate(names) if len(np.unique(X[:, j])) <= 2}  # 0/1 criteria: no level
            keep = T.criterion[T.importance > 0.05 * T.importance.max()]
            keep = [c for c in keep if c not in binary]
            B = B[B.feature.isin(keep)][["feature", "value"]].rename(columns={"feature": "criterion", "value": "border level"})
            B["border level"] = B["border level"].round(3)
            B = B.drop_duplicates().sort_values(["criterion", "border level"]).reset_index(drop=True)
    return T, B


def encode_like(df_new: pd.DataFrame, names):
    """Encode new cases with the same columns as the training data returned by prepare(). Missing indicator
    columns are filled with 0; rows with missing values in used columns raise an error."""
    df = df_new.copy()
    out = pd.DataFrame(index=df.index)
    for n in names:
        if n in df.columns:
            out[n] = pd.to_numeric(df[n], errors="coerce")
        elif "=" in n and n.split("=", 1)[0] in df.columns:  # binary text column
            c, v = n.split("=", 1); out[n] = (df[c].astype(str) == v).astype(float)
        else:  # one-hot column "<col>_<value>"
            hit = [c for c in df.columns if n.startswith(f"{c}_")]
            if hit:
                c = max(hit, key=len); out[n] = (df[c].astype(str) == n[len(c) + 1:]).astype(float)
            else:
                raise ValueError(f"column for criterion '{n}' not found in the new data")
    if out.isna().any().any():
        raise ValueError("new data contain missing or non-numeric values in: " + ", ".join(out.columns[out.isna().any()]))
    return out.values.astype(float)

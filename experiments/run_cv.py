"""Repeated stratified 5-fold CV (2 repeats) for all models, with per-fold predictions saved.

Usage: python experiments/run_cv.py <dataset> [...]
Output: results/cv/<dataset>.json (metrics per model, repeat, fold, incl. fit diagnostics for BAMAKAN)
        results/cv/parts/<dataset>__r<rep>_f<fold>.{json,npz} (fold checkpoints: test indices and predicted probabilities)
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, os, sys, time, warnings
import numpy as np
import torch
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/experiments")
from sklearn.model_selection import RepeatedStratifiedKFold
from benchmark import make_models, metrics
from datasets import DATASETS
from datasets import german

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
OUT = f"{ROOT}/results/cv"
MODELS = ["BAMAKAN", "EBM", "EBM-main", "GAM-mono", "XGB-mono", "POLR", "Border-frozen", "LogReg", "CatBoost",
          "XGBoost", "LightGBM", "RF", "SVM", "RealMLP", "MLP", "GMLVQ", "kNN", "NearestCentroid"]
MODELS = [m for m in os.environ.get("CV_MODELS", "").split(",") if m] or MODELS
UNDEFINED = {"nominal": {"GAM-mono", "XGB-mono", "POLR"}, "binary": {"POLR"}, "ordinal": set()}
N_REP = int(os.environ.get("CV_REPEATS", 2))
# EBM variants dominate run time (about 90% of it); they are evaluated on the first repeat (5 folds) only.
SLOW_FIRST_REP_ONLY = {"EBM", "EBM-main"}
FIT_LIMIT = int(os.environ.get("CV_FIT_LIMIT", 1800))  # seconds; a fit beyond this is recorded as an error


def load(name):
    if name == "german":
        X, y, task, f = german()
        return X, y, task, f
    return DATASETS[name]()


def _timeout(*_):
    raise TimeoutError("fit exceeded time limit")


def run(name):
    """Fold-level checkpointing: each (repeat, fold) is written to results/cv/parts/ and skipped on restart."""
    import signal
    X, y, task, _ = load(name)
    classes = np.unique(y)
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REP, random_state=42).split(X, y)
    for i, (tr, te) in enumerate(folds):
        rep, f = divmod(i, 5)
        part = f"{OUT}/parts/{name}__r{rep}_f{f}"
        if os.path.exists(part + ".json"):
            continue
        rows, preds = [], {"idx": te}
        mk = make_models(task, 100 * rep + f)
        for mn in MODELS:
            if mn in UNDEFINED[task] or (rep > 0 and mn in SLOW_FIRST_REP_ONLY):
                continue
            t0 = time.time()
            signal.signal(signal.SIGALRM, _timeout); signal.alarm(FIT_LIMIT)
            try:
                m = mk[mn]().fit(X[tr], y[tr])
                signal.alarm(0)
                fit_s = time.time() - t0
                t1 = time.time()
                P = np.zeros((len(te), len(classes)))
                P[:, np.searchsorted(classes, np.asarray(m.classes_))] = m.predict_proba(X[te])
                inf_ms = (time.time() - t1) / len(te) * 1e6  # ms per 1000 cases
                r = metrics(task, y[te], P, classes)
                r.update(fit_s=fit_s, infer_ms_per_1k=inf_ms)
                if mn == "BAMAKAN":
                    r.update(m.diagnostics())
                preds[mn] = P.astype(np.float32)
            except Exception as e:
                r = dict(error=repr(e)[:200])
            finally:
                signal.alarm(0)
            r.update(dataset=name, task=task, model=mn, repeat=rep, fold=f)
            rows.append(r)
        np.savez_compressed(part + ".npz", **preds)
        json.dump(rows, open(part + ".tmp", "w")); os.replace(part + ".tmp", part + ".json")
        print(name, "repeat", rep, "fold", f, "done", flush=True)
    rows = sum([json.load(open(f"{OUT}/parts/{name}__r{i // 5}_f{i % 5}.json")) for i in range(5 * N_REP)], [])
    json.dump(rows, open(f"{OUT}/{name}.json", "w"))


if __name__ == "__main__":
    for n in sys.argv[1:]:
        if os.path.exists(f"{OUT}/{n}.json") and not os.environ.get("CV_FORCE"):
            print(n, "exists, skipped", flush=True); continue
        t = time.time(); run(n); print(n, "finished in", round(time.time() - t, 1), "s", flush=True)

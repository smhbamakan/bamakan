"""TabPFN on the repeated-CV splits of run_cv.py (2 x 5 stratified folds, random_state = 42), for comparison with results/cv.
Output: results/cv_tabpfn/<dataset>__r<rep>_f<fold>.json (checkpoints; existing files are skipped).
Usage (e.g. on Google Colab with a GPU): python experiments/run_tabpfn.py [dataset ...]"""
import os, sys, json, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, f"{ROOT}/experiments"]
warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.model_selection import RepeatedStratifiedKFold
from tabpfn import TabPFNClassifier
from run_cv import load
from benchmark import metrics
from ablation import DATASETS_ALL
OUT = f"{ROOT}/results/cv_tabpfn"; os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
for name in (sys.argv[1:] or DATASETS_ALL):
    X, y, task, _ = load(name); classes = np.unique(y)
    for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42).split(X, y)):
        rep, f = i // 5, i % 5
        part = f"{OUT}/{name}__r{rep}_f{f}.json"
        if os.path.exists(part): continue
        t0 = time.time()
        try:
            m = TabPFNClassifier(device=dev, random_state=100 * rep + f, ignore_pretraining_limits=True).fit(X[tr], y[tr])
            fit_s = time.time() - t0
            P = np.zeros((len(te), len(classes))); P[:, np.searchsorted(classes, m.classes_)] = m.predict_proba(X[te])
            r = metrics(task, y[te], P, classes); r["fit_s"] = fit_s
        except Exception as e:
            r = dict(error=repr(e)[:300])
        r.update(dataset=name, task=task, model="TabPFN", repeat=rep, fold=f, device=dev)
        json.dump(r, open(part, "w"))
        print(name, rep, f, {k: round(v, 4) for k, v in r.items() if k in ("auc", "acc", "mae", "fit_s")}, r.get("error", ""), flush=True)
print("TABPFN_DONE")

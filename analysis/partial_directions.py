"""BAMAKAN with conditional (partial) directions on the 2x5 CV folds of run_cv.py; compared with direction='corr'."""
import os, sys, json, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, f"{ROOT}/experiments"]
import numpy as np
from multiprocessing import Pool
OUT = f"{ROOT}/results/partial_dir"


def job(a):
    name, rep, f, tr, te = a
    part = f"{OUT}/{name}__r{rep}_f{f}.json"
    if os.path.exists(part):
        return part
    warnings.filterwarnings("ignore")
    import torch; torch.set_num_threads(1)
    from bamakan import BAMAKANClassifier
    from benchmark import BAMAKAN_CONFIG, metrics
    from run_cv import load
    X, y, task, _ = load(name); classes = np.unique(y)
    t0 = time.time()
    m = BAMAKANClassifier(**{**BAMAKAN_CONFIG, "direction": "partial", "random_state": 100 * rep + f}).fit(X[tr], y[tr])
    P = np.zeros((len(te), len(classes))); P[:, np.searchsorted(classes, m.classes_)] = m.predict_proba(X[te])
    r = metrics(task, y[te], P, classes); r.update(fit_s=time.time() - t0, dataset=name, task=task, repeat=rep, fold=f)
    json.dump(r, open(part, "w")); return part


if __name__ == "__main__":
    from sklearn.model_selection import RepeatedStratifiedKFold
    from run_cv import load
    from ablation import DATASETS_ALL
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for n in DATASETS_ALL:
        X, y, task, _ = load(n)
        if task == "nominal":
            continue  # nominal tasks use learned interval directions; unaffected
        for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42).split(X, y)):
            jobs.append((n, i // 5, i % 5, tr, te))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        for k, r in enumerate(p.imap_unordered(job, jobs), 1):
            print(f"[{k}/{len(jobs)}] {os.path.basename(r)}", flush=True)
    print("PARTIAL_DONE", flush=True)

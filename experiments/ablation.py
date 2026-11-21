"""Component ablations of BAMAKAN on all 31 datasets (same 2x5 repeated CV folds as run_cv.py).

Usage: python experiments/ablation.py [n_workers]      (portable: no SIGALRM, works on Windows)
Output: results/ablation/parts/<dataset>__r<rep>_f<fold>.json (checkpoints, skipped on restart)
        results/ablation/all.json (merged)
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, os, sys, time, warnings, glob
import numpy as np
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/experiments")

OUT = f"{ROOT}/results/ablation"
DATASETS_ALL = ["breast_cancer", "benefits", "cirrhosis", "glass", "chile", "hdma", "maternal", "mroz", "ripley", "swd",
                "participation", "lev", "cpus_ord", "birthwt", "iris", "diamonds_cut", "computers_ord", "caravan",
                "default", "boston_ord", "crabs", "hcv", "wine", "oj", "turnout", "pima", "era", "heartfail", "esl",
                "biopsy", "german"]
# variant -> (constructor overrides, module-level switches in bamakan.core)
VARIANTS = {
    "default": ({}, {}),
    "B1": ({"n_borders": 1}, {}), "B2": ({"n_borders": 2}, {}), "B8": ({"n_borders": 8}, {}),
    "kappa_fixed": ({}, {"FIX_KAPPA": True}),
    "kappa_global": ({"kappa_mode": "global"}, {}),
    "kmax20": ({}, {"KAPPA_MAX": 20.0}), "kmax50": ({}, {"KAPPA_MAX": 50.0}),
    "kmax100": ({}, {"KAPPA_MAX": 100.0}), "kmax500": ({}, {"KAPPA_MAX": 500.0}),
    "dir_soft": ({"direction": "learn"}, {}),
    "norm_rank": ({"normalisation": "rank"}, {}),
    "delta0": ({}, {"DELTA": 0.0}), "delta0.25": ({}, {"DELTA": 0.25}), "delta1": ({}, {"DELTA": 1.0}),
    "free_borders": ({}, {"ORDER_BORDERS": False}),
    "l1_1e-3": ({}, {"L1": 1e-3}), "l1_1e-2": ({}, {"L1": 1e-2}),
}
DEFAULT_SWITCHES = dict(DELTA=0.5, KAPPA_MAX=200.0, FIX_KAPPA=False, ORDER_BORDERS=True, L1=0.0)


def job(args):
    name, rep, f, tr, te = args
    part = f"{OUT}/parts/{name}__r{rep}_f{f}.json"
    if os.path.exists(part):
        return name, rep, f, "skipped"
    warnings.filterwarnings("ignore")
    import torch; torch.set_num_threads(1)
    import bamakan.core as core
    from bamakan import BAMAKANClassifier
    from benchmark import BAMAKAN_CONFIG, metrics
    from run_cv import load
    X, y, task, _ = load(name)
    classes = np.unique(y)
    rows = []
    for v, (kw, sw) in VARIANTS.items():
        for k, val in {**DEFAULT_SWITCHES, **sw}.items():
            setattr(core, k, val)
        t0 = time.time()
        try:
            m = BAMAKANClassifier(**{**BAMAKAN_CONFIG, **kw, "random_state": 100 * rep + f}).fit(X[tr], y[tr])
            fit_s = time.time() - t0
            P = np.zeros((len(te), len(classes)))
            P[:, np.searchsorted(classes, np.asarray(m.classes_))] = m.predict_proba(X[te])
            r = metrics(task, y[te], P, classes); r["fit_s"] = fit_s
            r.update(m.diagnostics())
            r["n_params"] = int(sum(p.numel() for p in m.net_.parameters() if p.requires_grad))
        except Exception as e:
            r = dict(error=repr(e)[:200])
        r.update(dataset=name, task=task, variant=v, repeat=rep, fold=f)
        rows.append(r)
    for k, val in DEFAULT_SWITCHES.items():
        setattr(core, k, val)
    json.dump(rows, open(part + ".tmp", "w")); os.replace(part + ".tmp", part)
    return name, rep, f, "done"


def main():
    from multiprocessing import Pool
    from sklearn.model_selection import RepeatedStratifiedKFold
    from run_cv import load
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, os.cpu_count() - 1)
    jobs = []
    for name in DATASETS_ALL:
        X, y, _, _ = load(name)
        for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42).split(X, y)):
            jobs.append((name, i // 5, i % 5, tr, te))
    t = time.time()
    with Pool(n_workers) as pool:
        for k, (name, rep, f, st) in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"[{k}/{len(jobs)}] {name} r{rep} f{f} {st}  ({(time.time() - t) / 60:.1f} min)", flush=True)
    rows = sum([json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/parts/*.json"))], [])
    json.dump(rows, open(f"{OUT}/all.json", "w"))
    print("ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()

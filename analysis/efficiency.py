"""Controlled efficiency measurements. Each (dataset, model) runs in a fresh process, single-threaded, on the
same machine: fit time on the full dataset, inference time per 1000 cases (median of 5), serialized size (pickle bytes),
peak resident memory increase during fit, number of fitted parameters where defined."""
import os, sys, json, time, pickle, subprocess, resource
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ["breast_cancer", "pima", "german", "heartfail", "cirrhosis", "wine", "boston_ord", "default"]
MODELS = ["BAMAKAN", "EBM", "EBM-main", "GAM-mono", "XGB-mono", "POLR", "LogReg", "CatBoost", "XGBoost", "LightGBM", "RF",
          "SVM", "RealMLP", "MLP", "GMLVQ", "kNN", "NearestCentroid"]
OUT = f"{ROOT}/results/efficiency/efficiency.json"


def one(name, mn):
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    sys.path[:0] = [ROOT, f"{ROOT}/experiments"]
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, torch; torch.set_num_threads(1)
    from benchmark import make_models
    from run_cv import load, UNDEFINED
    X, y, task, _ = load(name)
    if mn in UNDEFINED[task]:
        return None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter(); m = make_models(task, 0)[mn]().fit(X, y); fit_s = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Xp = X[np.random.RandomState(0).randint(0, len(X), 1000)]
    ts = []
    for _ in range(5):
        t = time.perf_counter(); m.predict_proba(Xp); ts.append(time.perf_counter() - t)
    try:
        size = len(pickle.dumps(m))
    except Exception:
        size = None
    r = dict(dataset=name, task=task, model=mn, n=len(X), d=X.shape[1], fit_s=fit_s, infer_ms_per_1k=1000 * float(np.median(ts)),
             pickle_bytes=size, peak_rss_mb=(rss1 - rss0) / 1024)
    if mn == "BAMAKAN":
        r["n_params"] = int(sum(p.numel() for p in m.net_.parameters()))
    return r


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(json.dumps(one(sys.argv[1], sys.argv[2]))); sys.exit()
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["dataset"], r["model"]) for r in rows}
    for name in DS:
        for mn in MODELS:
            if (name, mn) in done: continue
            p = subprocess.run([sys.executable, "-W", "ignore", __file__, name, mn], capture_output=True, text=True, timeout=3600)
            line = [l for l in p.stdout.splitlines() if l.startswith("{") or l == "null"]
            r = json.loads(line[-1]) if line else dict(dataset=name, model=mn, error=p.stderr[-300:])
            if r is None: r = dict(dataset=name, model=mn, skipped=True)
            rows.append(r); json.dump(rows, open(OUT, "w"), indent=1)
            print(name, mn, {k: r.get(k) for k in ("fit_s", "pickle_bytes", "peak_rss_mb", "error")}, flush=True)
    print("EFF_DONE", flush=True)

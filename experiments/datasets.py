"""Real tabular datasets available offline (bundled in sklearn / ISLP / pydataset).

Each loader returns (X: float ndarray, y: ndarray, task: 'binary'|'nominal'|'ordinal', feature_names).
Leakage columns are removed where documented.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _pd(name):
    from pydataset import data
    return data(name)


def _prep(df, target, drop=(), ordinal_maps=None):
    df = df.drop(columns=[c for c in drop if c in df.columns]).dropna().reset_index(drop=True)
    y = df.pop(target)
    ordinal_maps = ordinal_maps or {}
    for c, mp in ordinal_maps.items():
        df[c] = df[c].map(mp)
    for c in list(df.columns):
        if df[c].dtype == object or str(df[c].dtype) == "category":
            vals = sorted(df[c].astype(str).unique())
            if len(vals) == 2:
                df[c] = (df[c].astype(str) == vals[1]).astype(float)
            else:
                df = pd.get_dummies(df, columns=[c], drop_first=True, dtype=float)
    return df.astype(float).values, y.values, list(df.columns)


def pima():
    df = pd.concat([_pd("Pima.tr"), _pd("Pima.te")], ignore_index=True)
    X, y, f = _prep(df, "type"); return X, (y == "Yes").astype(int), "binary", f

def biopsy():
    X, y, f = _prep(_pd("biopsy"), "class", drop=["ID"]); return X, (y == "malignant").astype(int), "binary", f

def breast_cancer():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(); return d.data, d.target, "binary", list(d.feature_names)

def hdma():
    X, y, f = _prep(_pd("Hdma"), "deny"); return X, (y == "yes").astype(int), "binary", f

def mroz():
    X, y, f = _prep(_pd("Mroz"), "work", drop=["hoursw", "wagew", "hearnw"])  # leakage: hours/wage given work
    return X, (y == "yes").astype(int), "binary", f

def participation():
    X, y, f = _prep(_pd("Participation"), "lfp"); return X, (y == "yes").astype(int), "binary", f

def birthwt():
    df = _pd("birthwt"); df["race"] = df["race"].astype(str)
    X, y, f = _prep(df, "low", drop=["bwt"]); return X, y.astype(int), "binary", f

def turnout():
    X, y, f = _prep(_pd("turnout"), "vote"); return X, y.astype(int), "binary", f

def benefits():
    X, y, f = _prep(_pd("Benefits"), "ui"); return X, (y == "yes").astype(int), "binary", f

def default():
    from ISLP import load_data
    X, y, f = _prep(load_data("Default"), "default"); return X, (y == "Yes").astype(int), "binary", f

def caravan():
    from ISLP import load_data
    X, y, f = _prep(load_data("Caravan"), "Purchase"); return X, (y == "Yes").astype(int), "binary", f

def oj():
    from ISLP import load_data
    df = load_data("OJ"); df["StoreID"] = df["StoreID"].astype(str)
    X, y, f = _prep(df, "Purchase", drop=["STORE", "Store7"]); return X, (y == "MM").astype(int), "binary", f

def ripley():
    df = pd.concat([_pd("synth.tr"), _pd("synth.te")], ignore_index=True)
    X, y, f = _prep(df, "yc"); return X, y.astype(int), "binary", f

# ---- nominal multiclass
def wine():
    from sklearn.datasets import load_wine
    d = load_wine(); return d.data, d.target, "nominal", list(d.feature_names)

def iris():
    from sklearn.datasets import load_iris
    d = load_iris(); return d.data, d.target, "nominal", list(d.feature_names)

def glass():
    X, y, f = _prep(_pd("fgl"), "type"); return X, pd.factorize(y, sort=True)[0], "nominal", f

def crabs():
    df = _pd("crabs"); df["cls"] = df["sp"].astype(str) + df["sex"].astype(str)
    X, y, f = _prep(df, "cls", drop=["sp", "sex", "index"]); return X, pd.factorize(y, sort=True)[0], "nominal", f

def chile():
    X, y, f = _prep(_pd("Chile"), "vote"); return X, pd.factorize(y, sort=True)[0], "nominal", f

# ---- ordinal
def diamonds(n=6000, seed=0):
    df = _pd("diamonds").sample(n, random_state=seed)
    cut = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
    color = {c: i for i, c in enumerate("JIHGFED")}
    clar = {c: i for i, c in enumerate(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])}
    df["cut"] = df["cut"].map(cut)
    X, y, f = _prep(df, "cut", ordinal_maps={"color": color, "clarity": clar}); return X, y.astype(int), "ordinal", f

def _binned(df, target, k=4, drop=()):
    yv = df[target].values
    y = np.digitize(yv, np.quantile(yv, np.linspace(0, 1, k + 1)[1:-1]))
    df = df.drop(columns=[target]); df["__y"] = y
    X, y, f = _prep(df, "__y", drop=drop); return X, y.astype(int), "ordinal", f

def boston_ord():
    return _binned(_pd("Boston"), "medv")

def computers_ord():
    return _binned(_pd("Computers"), "price")

def cpus_ord():
    return _binned(_pd("cpus"), "perf", drop=["name", "estperf"])


# ---- standard monotone / MCDM-sorting benchmarks (Ben-David; WEKA), ordinal targets
def _mono(name):
    from scipy.io import arff
    df = pd.DataFrame(arff.loadarff(f"{ROOT}/data/external/mono/{name}.arff")[0]).astype(float)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    return X, np.searchsorted(np.unique(y), y).astype(int), "ordinal", list(df.columns[:-1])

def era(): return _mono("ERA")
def esl(): return _mono("ESL")
def lev(): return _mono("LEV")
def swd(): return _mono("SWD")



CLIN = f"{ROOT}/data/external/clinical"


def maternal():  # Ahmed & Kashem (UCI 2023): maternal health risk triage, ordinal low < mid < high
    df = pd.read_csv(f"{CLIN}/maternal.csv", encoding="utf-8-sig")
    df["RiskLevel"] = df.RiskLevel.map({"low risk": 0, "mid risk": 1, "high risk": 2})
    X, y, f = _prep(df, "RiskLevel")
    return X, y.astype(int), "ordinal", f


def heartfail():  # Chicco & Jurman (BMC MIDM 2020): death in follow-up; 'time' dropped (follow-up length leaks label)
    X, y, f = _prep(pd.read_csv(f"{CLIN}/heartfail.csv"), "DEATH_EVENT", drop=("time",))
    return X, y.astype(int), "binary", f


def hcv():  # Hoffmann et al. (UCI 2020): liver disease stage donor < hepatitis < fibrosis < cirrhosis
    df = pd.read_csv(f"{CLIN}/hcv.csv").drop(columns=["0"])
    df = df[df.Category != "0s=suspect Blood Donor"]
    df["Category"] = df.Category.str[0].astype(int)
    X, y, f = _prep(df, "Category")
    return X, y.astype(int), "ordinal", f



def cirrhosis():  # Mayo PBC cohort (UCI 2023 release, Dickson et al.): histologic stage 1 < 2 < 3 < 4 from clinical/lab data;
    # follow-up outcomes (N_Days, Status) and treatment arm (Drug) dropped; complete cases only
    df = pd.read_csv(f"{CLIN}/cirrhosis.csv")
    X, y, f = _prep(df, "Stage", drop=("ID", "N_Days", "Status", "Drug"))
    return X, (y - 1).astype(int), "ordinal", f


DATASETS = {
    # binary
    "pima": pima, "biopsy": biopsy, "breast_cancer": breast_cancer, "hdma": hdma, "mroz": mroz,
    "participation": participation, "birthwt": birthwt, "turnout": turnout, "benefits": benefits,
    "default": default, "caravan": caravan, "oj": oj, "ripley": ripley,
    # nominal
    "wine": wine, "iris": iris, "glass": glass, "crabs": crabs, "chile": chile,
    # ordinal
    "diamonds_cut": diamonds, "boston_ord": boston_ord, "computers_ord": computers_ord, "cpus_ord": cpus_ord,
    "maternal": maternal, "heartfail": heartfail, "hcv": hcv, "cirrhosis": cirrhosis, "era": era, "esl": esl, "lev": lev, "swd": swd,
}

if __name__ == "__main__":
    for k, f in DATASETS.items():
        X, y, t, fn = f()
        print(f"{k:15s} {t:8s} n={X.shape[0]:6d} d={X.shape[1]:3d} classes={len(np.unique(y))} "
              f"minority={np.bincount(y).min() / len(y):.3f} nan={np.isnan(X).sum()}")


def german():
    """German Credit (UCI Statlog, 1000 x 20, binary): positive class = bad credit risk. Not used during development."""
    import pandas as pd
    X, y, f = _prep(pd.read_csv(f"{_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))}/data/external/GermanCredit.csv"), "credit_risk")
    return X, (y == 0).astype(int), "binary", f

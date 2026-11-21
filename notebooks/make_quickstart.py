"""Builds notebooks/BAMAKAN_quickstart.ipynb (run: python notebooks/make_quickstart.py)."""
import json, os

C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s, form=False: C.append({"cell_type": "code", "metadata": ({"cellView": "form"} if form else {}),
                                       "execution_count": None, "outputs": [], "source": s})

md("""# BAMAKAN in five steps

BAMAKAN is an interpretable classifier. For every criterion (column) it learns how important it is, in which direction
it acts, and **reference levels (borders) in the original units** — for example *ejection fraction below 33 %*. Every
prediction is explained criterion by criterion, and the model can be turned into a simple **point score**.

**How to use this notebook:** run the cells from top to bottom (▶ button, or Shift + Enter). Only the grey forms need
your input. No programming is required.

Paper: *BAMAKAN: A Learnable Border-Approximation Classifier with Monotone, Criterion-Level Explanations* ·
Code: https://github.com/smhbamakan/bamakan""")

md("## 1. Install")
code("""#@title Install BAMAKAN (takes about a minute)
!pip -q install git+https://github.com/smhbamakan/bamakan
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from bamakan import BAMAKAN, prepare, report, encode_like
print("Ready.")""", True)

md("""## 2. Load your data
Choose **Upload my CSV file** to use your own table (one row per case, one column per criterion, plus one column with
the outcome), or try the built-in example (Heart Failure: death during follow-up).""")
code("""#@title Load data
data_source = "Example: Heart Failure" #@param ["Example: Heart Failure", "Upload my CSV file"]
if data_source == "Upload my CSV file":
    from google.colab import files
    up = files.upload()
    df = pd.read_csv(next(iter(up)))
else:
    df = pd.read_csv("https://raw.githubusercontent.com/smhbamakan/bamakan/main/data/external/clinical/heartfail.csv")
    df = df.drop(columns=["time"])  # follow-up time depends on the outcome
print(f"{len(df)} rows, {df.shape[1]} columns")
display(df.head())""", True)

md("""## 3. Settings
* **target_column** – the column to predict.
* **task** – *binary* (two classes), *ordinal* (ordered classes, e.g. low < medium < high), *nominal* (unordered
  classes) or *auto*.
* **class_order** – for ordinal outcomes, the classes from lowest to highest separated by commas (leave empty for
  numbers or binary outcomes).
* **drop_columns** – columns to ignore, separated by commas (IDs, dates, anything decided after the outcome).
* **compact** – one border per criterion; recommended for fewer than about 2,000 rows.""")
code("""#@title Settings
target_column = "DEATH_EVENT" #@param {type:"string"}
task = "binary" #@param ["auto", "binary", "ordinal", "nominal"]
class_order = "" #@param {type:"string"}
drop_columns = "" #@param {type:"string"}
compact = True #@param {type:"boolean"}
test_share = 0.2 #@param {type:"slider", min:0.1, max:0.4, step:0.05}

order = [c.strip() for c in class_order.split(",") if c.strip()] or None
drop = [c.strip() for c in drop_columns.split(",") if c.strip()]
X, y, names, labels = prepare(df, target=target_column, drop=drop, class_order=order)
print(f"{X.shape[0]} cases, {X.shape[1]} criteria, classes (low to high): {labels}")
print("class sizes:", dict(zip(labels, np.bincount(y))))""", True)

md("## 4. Train and test")
code("""#@title Train the model and measure its accuracy on held-out cases
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_share, stratify=y, random_state=0)
model = BAMAKAN(task=task, compact=compact).fit(Xtr, ytr)
P = model.predict_proba(Xte); pred = P.argmax(1)
print(f"Accuracy:           {accuracy_score(yte, pred):.3f}")
print(f"Balanced accuracy:  {balanced_accuracy_score(yte, pred):.3f}")
if len(labels) == 2:
    print(f"AUC:                {roc_auc_score(yte, P[:, 1]):.3f}")
if model.task_ == "ordinal":
    print(f"Mean absolute error (classes): {np.abs(np.round(P @ np.arange(len(labels))) - yte).mean():.3f}")
print("\\nConfusion matrix (rows = true class, columns = predicted class):")
display(pd.DataFrame(confusion_matrix(yte, pred, labels=range(len(labels))), index=labels, columns=labels))""", True)

md("""## 5. Understand the model
**Importance** is the average size of a criterion's contribution to the prediction (logit scale). **Borders** are the
learned reference levels in the original units of each criterion.""")
code("""#@title Importance, direction and borders of each criterion
T, B = report(model, Xtr, ytr, names, labels)
display(T)
top = T.head(10)[::-1]
plt.figure(figsize=(7, 0.4 * len(top) + 1)); plt.barh(top.criterion, top.importance, color="#c0392b")
plt.xlabel("Importance"); plt.title("Most important criteria"); plt.tight_layout(); plt.show()
if B is not None:
    print("Learned border levels (original units):"); display(B)""", True)

code("""#@title Contribution curves of the four most important criteria
import torch
phi = model.explain(X) * float(torch.exp(model.net_.log_tau))
cont = [names.index(c) for c in T.criterion if len(np.unique(X[:, names.index(c)])) > 2][:4]
fig, ax = plt.subplots(1, len(cont), figsize=(4 * len(cont), 3.2))
for a, j in zip(np.atleast_1d(ax), cont):
    o = np.argsort(X[:, j]); a.plot(X[o, j], phi[o, j], color="#c0392b", lw=2); a.axhline(0, color="k", lw=0.5)
    a.set_title(names[j]); a.set_xlabel("value"); a.set_ylabel("contribution")
plt.tight_layout(); plt.show()
print("Above zero: the criterion pushes the case towards the higher class; below zero: towards the lower class.")""", True)

md("""## 6. Point score (binary and ordinal outcomes)
The model is turned into a short list of **if-then rules with integer points**. Add the points of the rules a case
fulfils; a higher total means a higher class.""")
code("""#@title Build the point score
if model.shape_ == "monotone":
    card = model.scorecard(names, X=Xtr, y=ytr)
    rules = pd.DataFrame([{"criterion": r["feature"], "rule": f"{r['op']} {r['value']:.4g}", "points": r["points"]}
                          for r in card["rules"]]).sort_values(["criterion", "points"])
    display(rules.reset_index(drop=True))
    Pc = model.predict_proba_scorecard(Xte, card)
    print(f"Point score on held-out cases: accuracy {accuracy_score(yte, Pc.argmax(1)):.3f}"
          + (f", AUC {roc_auc_score(yte, Pc[:, 1]):.3f}" if len(labels) == 2 else ""))
    print("Points of several rules for the same criterion add up.")
else:
    print("Point scores are available for binary and ordinal outcomes only.")""", True)

md("## 7. Explain one case")
code("""#@title Why did the model predict this for one case?
case_number = 0 #@param {type:"integer"}
x = Xte[case_number:case_number + 1]
p = model.predict_proba(x)[0]
c = model.explain(x)[0] * float(torch.exp(model.net_.log_tau))
print("Predicted probabilities:", {l: round(float(v), 3) for l, v in zip(labels, p)}, "| true class:", labels[yte[case_number]])
o = np.argsort(np.abs(c))[::-1][:10][::-1]
plt.figure(figsize=(7, 0.4 * len(o) + 1))
plt.barh([f"{names[j]} = {x[0, j]:g}" for j in o], c[o], color=["#c0392b" if v > 0 else "#2e86c1" for v in c[o]])
plt.axvline(0, color="k", lw=0.6); plt.xlabel("contribution (red: towards higher class, blue: towards lower class)")
plt.tight_layout(); plt.show()""", True)

md("## 8. Predict new cases (optional)")
code("""#@title Upload a CSV with new cases (same columns, outcome column not needed) and download predictions
from google.colab import files
final = BAMAKAN(task=task, compact=compact).fit(X, y)  # refit on all data
up = files.upload(); new = pd.read_csv(next(iter(up)))
Pn = final.predict_proba(encode_like(new, names))
res = new.copy(); res["predicted_class"] = [labels[i] for i in Pn.argmax(1)]
for i, l in enumerate(labels): res[f"probability_{l}"] = Pn[:, i].round(4)
res.to_csv("bamakan_predictions.csv", index=False); display(res.head()); files.download("bamakan_predictions.csv")""", True)

nb = {"nbformat": 4, "nbformat_minor": 5, "cells": C,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "BAMAKAN_quickstart.ipynb"), "w"), indent=1)
print("written")

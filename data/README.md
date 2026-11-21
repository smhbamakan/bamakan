# External datasets

These files are copies of public datasets, included so that the experiments run offline. They remain subject to the
terms of their original sources; please cite the original sources when using them.

| File | Dataset | Source |
|---|---|---|
| `GermanCredit.csv` | Statlog (German Credit Data) | UCI Machine Learning Repository |
| `mono/ERA.arff`, `ESL.arff`, `LEV.arff`, `SWD.arff` | Monotone ordinal datasets (Ben-David, 1992) | WEKA / OpenML collections |
| `clinical/heartfail.csv` | Heart Failure Clinical Records (Chicco & Jurman, 2020) | UCI Machine Learning Repository |
| `clinical/hcv.csv` | HCV data (Hoffmann et al., 2018) | UCI Machine Learning Repository |
| `clinical/cirrhosis.csv` | Mayo Clinic primary biliary cirrhosis cohort (Dickson et al., 1989) | UCI Machine Learning Repository |
| `clinical/maternal.csv` | Maternal Health Risk | UCI Machine Learning Repository (excluded from the rankings: 55% duplicate rows) |

All other datasets are loaded from Python packages (scikit-learn, ISLP, pydataset); see `experiments/datasets.py`.

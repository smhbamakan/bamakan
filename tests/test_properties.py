"""Tests for numerical stability, border ordering, fixed directions, monotonicity and reflection invariance."""
import sys, os
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from bamakan.core import log_power_mean, BAMAKANClassifier  # noqa: E402
from benchmark import BAMAKAN_CONFIG  # noqa: E402

CFG = {**BAMAKAN_CONFIG, "epochs": 120, "patience": 20}


def _data(task, n=400, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 10, size=(n, 4))
    z = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * (X[:, 2] > 5) + rng.normal(0, 1, n)
    if task == "binary":
        y = (z > np.median(z)).astype(int)
    else:
        y = np.digitize(z, np.quantile(z, [0.33, 0.66]))
    return X, y


def test_power_mean_gradient_double_precision():
    """Analytic gradients of log M_p (logsumexp branch and Taylor branch) match finite differences."""
    torch.manual_seed(0)
    logu = torch.log(torch.rand(50, 1, dtype=torch.float64) + 1.0)
    for p0 in (-6.0, -1.0, -2e-3, 0.0, 5e-4, 2e-3, 1.0, 6.0):
        p = torch.tensor([p0], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda q: log_power_mean(logu, q), (p,), eps=1e-6, atol=1e-5)


def test_power_mean_continuous_at_branch_switch():
    """The Taylor branch (|p| < 1e-3) and the exact branch agree at the switch point."""
    logu = torch.log(torch.rand(200, 1, dtype=torch.float64) + 1.0)
    lo = log_power_mean(logu, torch.tensor([1e-3 - 1e-9], dtype=torch.float64))
    hi = log_power_mean(logu, torch.tensor([1e-3 + 1e-9], dtype=torch.float64))
    assert abs(float(lo - hi)) < 1e-7


@pytest.mark.parametrize("task", ["binary", "ordinal"])
def test_borders_are_ordered(task):
    X, y = _data(task)
    m = BAMAKANClassifier(task=task, random_state=0, **CFG).fit(X, y)
    with torch.no_grad():
        M = m.net_.borders().numpy()
    assert np.all(np.diff(M, axis=-1) >= -1e-7)


@pytest.mark.parametrize("task", ["binary", "ordinal"])
def test_hard_directions_and_monotone_probabilities(task):
    """With hard directions, P(y=1|x) (binary) and every cumulative P(y>k|x) (ordinal) are monotone in each criterion,
    in the direction given by s."""
    X, y = _data(task)
    m = BAMAKANClassifier(task=task, random_state=0, **CFG).fit(X, y)
    s = m.net_.s_full().detach().numpy()[0, :, 0]
    assert set(np.unique(s)) <= {0.0, 1.0}
    rng = np.random.RandomState(1)
    for j in range(X.shape[1]):
        base = X[rng.choice(len(X), 30)]
        grid = np.linspace(-2, 12, 40)  # includes extrapolation beyond the training range
        for b in base:
            Z = np.repeat(b[None], len(grid), 0); Z[:, j] = grid
            P = m.predict_proba(Z)
            cum = P[:, 1:] if task == "binary" else np.cumsum(P[:, ::-1], 1)[:, ::-1][:, 1:]  # P(y>k)
            dif = np.diff(cum, axis=0) * (1 if s[j] == 1 else -1)
            assert dif.min() > -1e-6


def test_reflection_invariance_objective_level():
    """Replacing a criterion by C - x_j flips its inferred direction and leaves the oriented inputs u (hence the
    training objective at every parameter value) unchanged. Equality of the fitted parameters is NOT asserted:
    the non-convex optimiser is sensitive to float32 rounding."""
    X, y = _data("binary")
    X2 = X.copy(); X2[:, 1] = 20.0 - X2[:, 1]
    cfg = {**CFG, "epochs": 1, "patience": 1}
    m1 = BAMAKANClassifier(task="binary", random_state=0, **cfg).fit(X, y)
    m2 = BAMAKANClassifier(task="binary", random_state=0, **cfg).fit(X2, y)
    s1 = m1.net_.s_full().detach().numpy()[0, :, 0]; s2 = m2.net_.s_full().detach().numpy()[0, :, 0]
    assert s1[1] == 1 - s2[1] and np.all(np.delete(s1, 1) == np.delete(s2, 1))
    orient = lambda m, Z, s: np.where(s == 1, m.norm_.transform(Z), 1 - m.norm_.transform(Z))
    assert np.max(np.abs(orient(m1, X, s1) - orient(m2, X2, s2))) < 1e-9


def test_borders_in_original_units_round_trip():
    """A border reported in original units maps back to the same oriented value u = m."""
    X, y = _data("binary")
    m = BAMAKANClassifier(task="binary", random_state=0, **CFG).fit(X, y)
    rows = m.borders_in_original_units()
    with torch.no_grad():
        M = m.net_.borders().numpy()[0]; s = m.net_.s_full().numpy()[0]
    for r in rows:
        j, k = r["feature"], r["border"]
        n = float(m.norm_.transform(np.array([[r["value"] if c == j else 0.0 for c in range(X.shape[1])]]))[0, j])
        u = s[j, k] * n + (1 - s[j, k]) * (1 - n) + 1
        assert abs(u - M[j, k]) < 1e-4


def test_diagnostics_reported():
    X, y = _data("binary")
    d = BAMAKANClassifier(task="binary", random_state=0, **CFG).fit(X, y).diagnostics()
    assert d["directions_soft"] == 0.0 and d["n_restores"] >= 0

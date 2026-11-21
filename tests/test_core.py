import numpy as np, torch, pytest
from sklearn.datasets import make_classification
from bamakan import BAMAKANClassifier, MABACClassifier, Normaliser, entropy_weights, corr_directions, log_power_mean, BorderNet

def data(n=400, d=6, seed=0, classes=2):
    X, y = make_classification(n_samples=n, n_features=d, n_informative=4, n_redundant=0, n_classes=classes,
                               n_clusters_per_class=1, random_state=seed)
    return X, y

def test_power_mean_special_cases():
    u = torch.rand(500, 1, 3) + 0.5
    lu = torch.log(u)
    for p, ref in [(0.0, torch.exp(lu.mean(0))), (1.0, u.mean(0)), (-1.0, 1 / (1 / u).mean(0)), (2.0, (u**2).mean(0).sqrt())]:
        got = torch.exp(log_power_mean(lu, torch.full((1, 3, 1), p)))[..., 0]
        assert torch.allclose(got, ref, rtol=1e-4), p
    big = torch.exp(log_power_mean(lu, torch.full((1, 3, 1), 300.0)))[..., 0]
    assert torch.allclose(big, u.max(0).values, rtol=2e-2)

def test_prop1_classical_mabac_is_special_case():
    X, y = data()
    norm = Normaliser("minmax").fit(X); N = norm.transform(X)
    s = corr_directions(N, y.reshape(-1, 1).astype(float)); w = entropy_weights(np.where(s[0] > 0.5, N, 1 - N))
    net = BorderNet(1, X.shape[1], 1, "per_head", s, w[None], "fixed", kappa0=1e-4, p_init=np.zeros(1))
    net.ref = torch.as_tensor(N, dtype=torch.float32)
    S_net = net.score(net.ref).detach().numpy()[:, 0]
    # classical MABAC, written step by step as in Pamucar & Cirovic (2015)
    ntil = np.where(s[0] > 0.5, N, 1 - N)
    V = w * (ntil + 1)
    G = np.prod(V, axis=0) ** (1 / len(V))
    S_cls = (V - G).sum(1)
    assert np.allclose(S_net, S_cls, atol=1e-4)
    assert np.mean(np.sign(S_net) == np.sign(S_cls)) == 1.0

def test_lemma1_equivalence_with_saw():
    X, _ = data()
    N = Normaliser().fit(X).transform(X); w = entropy_weights(N)
    V = w * (N + 1); G = np.prod(V, 0) ** (1 / len(V)); S = (V - G).sum(1)
    saw = (w * N).sum(1)
    assert np.allclose(S - saw, (S - saw)[0])  # constant offset -> identical ranking

@pytest.mark.parametrize("task,classes", [("binary", 2), ("nominal", 3), ("ordinal", 4)])
def test_fit_predict_and_proba(task, classes):
    X, y = data(classes=classes)
    if task == "ordinal":  # ordinal labels from a binned monotone latent score
        lat = X[:, 0] + 0.5 * X[:, 1] - X[:, 2] + 0.3 * np.random.RandomState(0).randn(len(X))
        y = np.digitize(lat, np.quantile(lat, [0.25, 0.5, 0.75]))
    m = BAMAKANClassifier(task=task, epochs=60, random_state=0).fit(X, y)
    P = m.predict_proba(X)
    assert P.shape == (len(X), classes) and np.allclose(P.sum(1), 1) and (P >= 0).all()
    assert (m.predict(X) == y).mean() > 0.6

def test_theorem1_inductive_no_rank_reversal():
    X, y = data()
    m = BAMAKANClassifier(epochs=60).fit(X[:300], y[:300])
    s1 = m.decision_function(X[300:350])
    s2 = m.decision_function(np.vstack([X[300:350], X[350:] * 5 + 3]))[:50]  # add extreme new points
    assert np.array_equal(s1, s2)
    # classical transductive MABAC changes when new alternatives are added
    def mabac(Xs):
        N = Normaliser().fit(Xs).transform(Xs); w = np.full(Xs.shape[1], 1 / Xs.shape[1])
        V = w * (N + 1); return (V - np.prod(V, 0) ** (1 / len(V))).sum(1)
    a = mabac(X[300:350]); b = mabac(np.vstack([X[300:350], X[350:] * 5 + 3]))[:50]
    assert not np.allclose(a, b)

@pytest.mark.parametrize("norm", ["minmax", "rank"])
def test_theorem2_ivs_affine(norm):
    X, y = data()
    a = np.array([3.0, 0.01, 7, 1, 100, 0.5]); b = np.array([-5, 2, 0, 10, -100, 1])
    m1 = BAMAKANClassifier(normalisation=norm, epochs=50, random_state=1).fit(X, y)
    m2 = BAMAKANClassifier(normalisation=norm, epochs=50, random_state=1).fit(X * a + b, y)
    assert np.allclose(m1.predict_proba(X), m2.predict_proba(X * a + b), atol=1e-4)

def test_theorem3a_ifc_linear_cost_reformulation():
    X, y = data()
    X2 = X.copy(); X2[:, 0] = 10 - X2[:, 0]; X2[:, 3] = 6240 - X2[:, 3]
    for direction in ["corr", "learn"]:
        m1 = BAMAKANClassifier(direction=direction, epochs=50, random_state=2).fit(X, y)
        m2 = BAMAKANClassifier(direction=direction, epochs=50, random_state=2).fit(X2, y)
        assert np.allclose(m1.predict_proba(X), m2.predict_proba(X2), atol=1e-4)

def test_theorem3b_reciprocal_rank_invariant_minmax_not():
    X, y = data(); X = X - X.min(0) + 1.0
    X2 = X.copy(); X2[:, 1] = 60 / X2[:, 1]; X2[:, 2] = 60 / X2[:, 2]
    r1 = BAMAKANClassifier(normalisation="rank", epochs=50, random_state=3).fit(X, y)
    r2 = BAMAKANClassifier(normalisation="rank", epochs=50, random_state=3).fit(X2, y)
    assert np.allclose(r1.predict_proba(X), r2.predict_proba(X2), atol=1e-4)
    N1 = Normaliser().fit(X).transform(X)[:, 1]; N2 = 1 - Normaliser().fit(X2).transform(X2)[:, 1]
    assert not np.allclose(N1, N2)  # min-max is not invariant to the reciprocal formulation

def test_theorem4_monotonicity():
    X, y = data()
    m = BAMAKANClassifier(direction="corr", epochs=60).fit(X, y)
    s = m.net_.s_full()[0, :, 0].numpy()
    grid = np.linspace(X.min() - 1, X.max() + 1, 200)
    base = np.tile(X[:1], (200, 1))
    for j in range(X.shape[1]):
        Z = base.copy(); Z[:, j] = grid
        p = m.predict_proba(Z)[:, 1]
        dp = np.diff(p)
        assert (dp >= -1e-7).all() if s[j] > 0.5 else (dp <= 1e-7).all()

def test_classical_mabac_baseline_runs():
    for c in (2, 3):
        X, y = data(classes=c)
        m = MABACClassifier().fit(X, y)
        assert (m.predict(X) == y).mean() > 1.0 / c

def test_explanations_shapes():
    X, y = data()
    m = BAMAKANClassifier(epochs=30).fit(X, y)
    phi = m.explain(X[:5]); assert phi.shape == (5, X.shape[1])
    assert np.allclose(phi.sum(1), m.decision_function(X[:5])[:, 0], atol=1e-5)
    rows = m.borders_in_original_units(); assert len(rows) == X.shape[1] * m.n_borders


# ---------------------------------------------------------------- v2 components (P3.5)
def test_psi_normed_limits():
    """tanh(kz)/tanh(k) -> z as k -> 0 (MABAC distance) and -> sign(z) as k -> inf (region count)."""
    import torch
    z = torch.linspace(-0.9, 0.9, 19)
    small, large = torch.tensor(1e-4), torch.tensor(200.0)
    assert torch.allclose(torch.tanh(small * z) / torch.tanh(small), z, atol=1e-5)
    nz = z.abs() > 0.05
    assert torch.allclose((torch.tanh(large * z) / torch.tanh(large))[nz], torch.sign(z)[nz], atol=1e-3)


def test_quantile_matched_p():
    from bamakan.core import quantile_matched_p
    rng = np.random.RandomState(0)
    N = rng.beta(0.5, 3, size=(2000, 2))  # strongly skewed
    S = np.array([[1.0, 0.0]])
    B = 4
    P = quantile_matched_p(N, S, B)
    for j in range(2):
        u = 1 + (N[:, j] if S[0, j] > 0.5 else 1 - N[:, j])
        for k in range(B):
            p = P[0, j, k]
            if abs(p) >= 60:  # target beyond reachable range -> clipped at the bound (near min / max)
                continue
            M = np.exp(np.log(u).mean()) if abs(p) < 1e-6 else np.mean(u ** p) ** (1 / p)
            assert abs(M - np.quantile(np.unique(u), (k + 0.5) / B)) < 1e-3


def test_v2_fit_and_monotone():
    X, y = data()
    m = BAMAKANClassifier(n_borders=4, kappa_mode="local", psi="normed", kappa0=3.0, p_init="quantile",
                          lr_schedule="cosine", epochs=100, refit=True).fit(X, y)
    assert m.score(X, y) > 0.75
    # T4 still holds per criterion in the monotone shape
    s = m.net_.s_full()[0].detach().numpy()
    j = 0
    Z = np.tile(X[:1], (50, 1)); Z[:, j] = np.linspace(X[:, j].min(), X[:, j].max(), 50)
    d = np.diff(m.decision_function(Z)[:, 0])
    assert (d >= -1e-6).all() or (d <= 1e-6).all() or not (np.all(s[j] > .5) or np.all(s[j] < .5))

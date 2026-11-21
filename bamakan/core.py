"""BAMAKAN global branch: learnable multi-border MABAC classifier.

See spec.md for the formal definition. scikit-learn compatible.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

DELTA = 0.5
# Ablation switches (module level so the public API is unchanged; defaults reproduce the paper's configuration)
KAPPA_MAX = 200.0      # upper clamp on kappa
FIX_KAPPA = False      # True: kappa frozen at kappa0
ORDER_BORDERS = True   # False: free (unordered) power-mean exponents
L1 = 0.0               # L1 penalty on criterion weights exp(alpha)


# ----------------------------------------------------------------------------------------------
# Normalisation (frozen on training data)
# ----------------------------------------------------------------------------------------------
class Normaliser:
    def __init__(self, kind: str = "minmax", q: float = 0.0):
        assert kind in ("minmax", "rank")
        self.kind, self.q = kind, q

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self.kind == "minmax":
            self.a_ = np.quantile(X, self.q, axis=0)
            self.b_ = np.quantile(X, 1 - self.q, axis=0)
            span = self.b_ - self.a_
            self.span_ = np.where(span > 0, span, 1.0)
        else:
            self.sorted_ = [np.sort(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.kind == "minmax":
            N = (X - self.a_) / self.span_
            return np.clip(N, -DELTA, 1 + DELTA)
        N = np.empty_like(X)
        for j, s in enumerate(self.sorted_):
            lo = np.searchsorted(s, X[:, j], side="left")
            hi = np.searchsorted(s, X[:, j], side="right")
            N[:, j] = (lo + hi) / (2.0 * len(s))  # mid-rank ECDF, invariant to strictly increasing maps
        return N

    def inverse(self, n, j):
        """Map a normalised value of feature j back to original units (for explanations)."""
        if self.kind == "minmax":
            return self.a_[j] + n * self.span_[j]
        s = self.sorted_[j]
        return float(np.quantile(s, np.clip(n, 0, 1)))


def entropy_weights(N):
    """Classical entropy weights on a normalised matrix with values in [0,1] (shifted by +1 as in MABAC)."""
    P = N + 1.0
    P = P / P.sum(axis=0, keepdims=True)
    m = N.shape[0]
    E = -(P * np.log(P + 1e-12)).sum(axis=0) / math.log(m)
    d = 1 - E
    if d.sum() <= 0:
        return np.full(N.shape[1], 1.0 / N.shape[1])
    return d / d.sum()


def corr_directions(N, Y):
    """Y: (m, H) targets per head. Returns s in {0,1}^(H,d): 1 = benefit, 0 = cost."""
    H, d = Y.shape[1], N.shape[1]
    S = np.ones((H, d))
    for h in range(H):
        for j in range(d):
            if np.std(N[:, j]) == 0 or np.std(Y[:, h]) == 0:
                continue
            r = spearmanr(N[:, j], Y[:, h]).statistic
            S[h, j] = 1.0 if (np.isnan(r) or r >= 0) else 0.0
    return S


def partial_directions(N, Y, alpha=1.0):
    """Conditional directions: sign of each criterion's coefficient in a ridge regression of the head target on all
    standardised criteria (controls for the other criteria; robust to correlated criteria). 1 = benefit, 0 = cost."""
    Z = (N - N.mean(0)) / np.where(N.std(0) > 0, N.std(0), 1.0)
    S = np.ones((Y.shape[1], N.shape[1]))
    for h in range(Y.shape[1]):
        y = Y[:, h] - Y[:, h].mean()
        b = np.linalg.solve(Z.T @ Z + alpha * len(Z) * 1e-3 * np.eye(Z.shape[1]), Z.T @ y)
        S[h] = (b >= 0).astype(float)
    return S


# ----------------------------------------------------------------------------------------------
# Power mean, differentiable in p (log-domain, Taylor-smoothed around p = 0)
# ----------------------------------------------------------------------------------------------
def log_power_mean(logu: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """logu: (m, ...) log of u over the reference set; p broadcastable to logu.shape[1:].
    Returns log M_p over axis 0. Exact for |p| >= 1e-3, second-order Taylor (log-normal) around p = 0."""
    m = logu.shape[0]
    if logu.dim() == p.dim():  # legacy call: logu (m,H,d), p (H,d,B)
        logu = logu.unsqueeze(-1)
    mu = logu.mean(0)
    var = logu.var(0, unbiased=False)
    small = p.abs() < 1e-3
    p_safe = torch.where(small, torch.full_like(p, 1e-3), p)
    lme = torch.logsumexp(p_safe.unsqueeze(0) * logu, dim=0) - math.log(m)  # log mean u^p
    exact = lme / p_safe
    taylor = mu + 0.5 * p * var
    return torch.where(small, taylor, exact)


def quantile_matched_p(N, S, B, lo=-60.0, hi=60.0, iters=50):
    """Initial power-mean exponents p (H,d,B) such that border k = M_p(u_j) sits at the (k+1/2)/B quantile of
    u_j = 1 + direction-adjusted n_j (unique values used, so discrete features get distinct borders).
    M_p is increasing in p, so each p is found by bisection. S: (H,d) or (H,d,B) directions in {0,1}."""
    N = np.clip(np.asarray(N, float), 0, 1)
    H, d = S.shape[0], S.shape[1]
    S = S if S.ndim == 3 else np.repeat(S[..., None], B, -1)
    P = np.zeros((H, d, B))
    for h in range(H):
        for j in range(d):
            for k in range(B):
                u = 1 + (N[:, j] if S[h, j, k] > 0.5 else 1 - N[:, j])
                uq = np.unique(u)
                if len(uq) < 2:
                    continue
                target = np.quantile(uq, (k + 0.5) / B)
                lu = np.log(u)
                def M(pp):
                    if abs(pp) < 1e-6:
                        return np.exp(lu.mean())
                    a = pp * lu; mx = a.max()
                    return np.exp((mx + np.log(np.mean(np.exp(a - mx)))) / pp)
                a_, b_ = lo, hi
                if target <= M(a_):
                    P[h, j, k] = a_; continue
                if target >= M(b_):
                    P[h, j, k] = b_; continue
                for _ in range(iters):
                    mid = (a_ + b_) / 2
                    a_, b_ = (mid, b_) if M(mid) < target else (a_, mid)
                P[h, j, k] = (a_ + b_) / 2
    return P


# ----------------------------------------------------------------------------------------------
# Torch module
# ----------------------------------------------------------------------------------------------
class BorderNet(nn.Module):
    def __init__(self, H, d, B, n_out_thresholds, s_init, w_init, direction, kappa0=1.0, p_init=None,
                 shape="monotone", tau0=10.0, kappa_mode="global", psi="scaled"):
        super().__init__()
        self.H, self.d, self.B = H, d, B
        self.direction = direction
        self.shape = shape
        self.alpha = nn.Parameter(torch.log(torch.as_tensor(w_init, dtype=torch.float32) + 1e-6))  # (H,d)
        s_init = np.asarray(s_init, dtype=float)
        if shape == "interval":
            # one direction per border: border 0 = correlation direction ("G+ above"), border 1 = opposite
            # ("G+ below"), ... -> unimodal / band-shaped contributions (for unordered classes)
            s_init = np.stack([s_init if k % 2 == 0 else 1 - s_init for k in range(B)], -1)  # (H,d,B)
            direction = self.direction = "learn"
        if direction == "learn":
            self.theta_s = nn.Parameter(torch.as_tensor(np.where(s_init > 0.5, 2.0, -2.0), dtype=torch.float32))
        else:
            self.register_buffer("s_fixed", torch.as_tensor(s_init, dtype=torch.float32))
        if isinstance(p_init, str):  # 'quantile' -> computed by the estimator and passed as an array
            raise ValueError("pass the array from quantile_matched_p")
        if p_init is None:
            # spread borders: B=1 -> geometric mean (p=0); B>1 -> spread between low and high power means
            p_init = np.linspace(-4, 4, B) if B > 1 else np.zeros(1)
        P0 = np.array(np.broadcast_to(p_init, (H, d, B)), dtype=np.float64)
        self.ordered = (shape != "interval") and B > 1 and ORDER_BORDERS
        if self.ordered:  # ordered exponents => ordered borders (M_p is non-decreasing in p for u > 0)
            P0 = np.sort(P0, -1)
            gap = np.maximum(np.diff(P0, axis=-1), 1e-3)
            self.p0 = nn.Parameter(torch.as_tensor(P0[..., :1], dtype=torch.float32))
            self.dp = nn.Parameter(torch.as_tensor(np.log(np.expm1(gap)), dtype=torch.float32))  # softplus^-1
        else:
            self.p_free = nn.Parameter(torch.as_tensor(P0, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.zeros(H, d, B))
        self.kappa_mode, self.psi = kappa_mode, psi
        kshape = (H, d, B) if kappa_mode == "local" else ()
        self.log_kappa = nn.Parameter(torch.full(kshape, math.log(kappa0)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau0)))
        self.n_thr = n_out_thresholds
        if n_out_thresholds == "per_head":
            self.theta = nn.Parameter(torch.zeros(H))
        else:  # ordinal: K-1 ordered thresholds
            K1 = n_out_thresholds
            self.theta0 = nn.Parameter(torch.tensor(-1.0 * (K1 - 1) / 2))
            self.dtheta = nn.Parameter(torch.zeros(max(K1 - 1, 0)))
        self.ref = None  # (m_ref, d) normalised reference set (training only)
        self.M_cached = None  # borders frozen after training

    # --- helpers
    def s(self):
        return torch.sigmoid(self.theta_s) if self.direction == "learn" else self.s_fixed

    def s_full(self):
        s = self.s()
        return s if s.dim() == 3 else s.unsqueeze(-1).expand(self.H, self.d, self.B)

    def u(self, N):  # N: (m,d) -> (m,H,d,B)
        s = self.s_full().unsqueeze(0)
        Nn = N[:, None, :, None]
        return s * Nn + (1 - s) * (1 - Nn) + 1.0

    def kappa(self):
        return torch.exp(self.log_kappa).clamp(1e-4, KAPPA_MAX)

    def weights(self):
        return torch.softmax(self.alpha, dim=-1)

    def borders(self):
        """(H,d,B) border values m_hjk in u-space."""
        if getattr(self, "M_cached", None) is not None:
            return self.M_cached
        logu = torch.log(self.u(self.ref).clamp_min(1e-6))  # (m,H,d,B)
        return torch.exp(log_power_mean(logu, self.p()))

    def p(self):
        """(H,d,B) power-mean exponents; ordered along the border axis for monotone shapes."""
        if self.ordered:
            return torch.cat([self.p0, self.p0 + torch.cumsum(F.softplus(self.dp), -1)], -1)
        return self.p_free

    def contributions(self, N):
        """Per-criterion signed contributions phi: (m,H,d)."""
        U = self.u(N)  # (m,H,d,B)
        M = self.borders().unsqueeze(0)  # (1,H,d,B)
        kappa = self.kappa()
        if self.psi == "normed":  # tanh(kz)/tanh(k): -> z as k->0, -> sign(z) as k->inf, amplitude ~ 1 for all k
            psi = torch.tanh(kappa * (U - M)) / torch.tanh(kappa)
        else:  # tanh(kz)/k (original): -> z as k->0, amplitude 1/k
            psi = torch.tanh(kappa * (U - M)) / kappa
        beta = torch.softmax(self.gamma, dim=-1).unsqueeze(0)
        return self.weights().unsqueeze(0) * (beta * psi).sum(-1)

    def score(self, N):
        return self.contributions(N).sum(-1)  # (m,H)

    def thresholds(self):
        if self.n_thr == "per_head":
            return self.theta
        return torch.cat([self.theta0.view(1), self.theta0 + torch.cumsum(F.softplus(self.dtheta), 0)])

    def logits(self, N):
        tau = torch.exp(self.log_tau)
        S = self.score(N)
        if self.n_thr == "per_head":
            return tau * S - self.theta  # (m,H)
        return tau * S[:, :1] - self.thresholds().unsqueeze(0)  # (m,K-1) cumulative logits


# ----------------------------------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------------------------------
class BAMAKANClassifier(BaseEstimator, ClassifierMixin):
    """Global branch of BAMAKAN (learnable multi-border MABAC).

    Parameters
    ----------
    task : 'auto' | 'binary' | 'ordinal' | 'nominal'
    normalisation : 'minmax' | 'rank'
    direction : 'corr' | 'learn' | 'given'   (benefit_mask used when 'given')
    n_borders : borders per criterion (B)
    weight_init : 'entropy' | 'uniform'
    shape : 'auto' | 'monotone' | 'interval'  ('auto' = monotone for binary/ordinal, interval for nominal)
    """

    def __init__(self, task="auto", normalisation="minmax", direction="learn", benefit_mask=None, n_borders=2,
                 weight_init="entropy", kappa0=1.0, lr=0.05, epochs=400, batch_size=512, patience=40,
                 val_size=0.15, l2=1e-4, ref_size=4096, shape="auto", tau0=10.0, kappa_mode="global",
                 psi="scaled", lr_schedule="constant", refit=False, p_init="spread", optimizer="adam", lbfgs_block=10,
                 random_state=0, verbose=False):
        self.task = task
        self.normalisation = normalisation
        self.direction = direction
        self.benefit_mask = benefit_mask
        self.n_borders = n_borders
        self.weight_init = weight_init
        self.kappa0 = kappa0
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_size = val_size
        self.l2 = l2
        self.ref_size = ref_size
        self.shape = shape
        self.tau0 = tau0
        self.kappa_mode = kappa_mode
        self.psi = psi
        self.lr_schedule = lr_schedule
        self.refit = refit
        self.p_init = p_init
        self.optimizer = optimizer
        self.lbfgs_block = lbfgs_block
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    def _targets(self, yi):
        if self.task_ == "nominal":
            return np.eye(len(self.classes_))[yi]
        return yi.reshape(-1, 1).astype(float)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = unique_labels(y)
        yi = np.searchsorted(self.classes_, y)
        C = len(self.classes_)
        task = self.task
        if task == "auto":
            task = "binary" if C == 2 else "nominal"
        self.task_ = task
        self.n_restores_ = 0
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        self.norm_ = Normaliser(self.normalisation).fit(X)
        N = self.norm_.transform(X)
        Y = self._targets(yi)
        H = Y.shape[1]
        d = X.shape[1]

        if self.direction == "given":
            s0 = np.tile(np.asarray(self.benefit_mask, dtype=float), (H, 1))
        elif self.direction == "partial":
            s0 = partial_directions(N, Y)
        else:
            s0 = corr_directions(N, Y)
        if self.weight_init == "entropy":
            Nd = np.clip(N, 0, 1)  # entropy on direction-adjusted values (as in MABAC), one vector per head
            w0 = np.stack([entropy_weights(np.where(s0[h] > 0.5, Nd, 1 - Nd)) for h in range(H)])
        else:
            w0 = np.full((H, d), 1.0 / d)
        thr = "per_head" if task in ("binary", "nominal") else C - 1
        shape = self.shape if self.shape != "auto" else ("interval" if task == "nominal" else "monotone")
        self.shape_ = shape
        ref_idx = rng.choice(len(N), size=min(self.ref_size, len(N)), replace=False)
        p0 = None
        if self.p_init == "quantile":
            if shape == "interval":
                S3 = np.stack([s0 if k % 2 == 0 else 1 - s0 for k in range(self.n_borders)], -1)
            else:
                S3 = s0
            p0 = quantile_matched_p(N[ref_idx], S3, self.n_borders)
        self.net_ = BorderNet(H, d, self.n_borders, thr, s0, w0, "learn" if self.direction == "learn" else "fixed",
                              kappa0=self.kappa0, shape=shape, tau0=self.tau0, kappa_mode=self.kappa_mode,
                              psi=self.psi, p_init=p0)
        self.net_.ref = torch.as_tensor(N[ref_idx], dtype=torch.float32)
        if FIX_KAPPA:
            self.net_.log_kappa.requires_grad_(False)

        # train/val split for early stopping
        strat = yi if np.min(np.bincount(yi)) >= 2 else None
        tr, va = train_test_split(np.arange(len(N)), test_size=self.val_size, random_state=self.random_state,
                                  stratify=strat)
        Nt = torch.as_tensor(N, dtype=torch.float32)
        yt = torch.as_tensor(yi, dtype=torch.long)
        init_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
        best_ep = self._train(Nt, yt, tr, va, rng, self.epochs)
        if self.refit:  # retrain from the same initialisation on all data for the selected number of epochs
            self.net_.load_state_dict(init_state)
            self._train(Nt, yt, np.arange(len(N)), None, rng, best_ep)
        # The borders are fixed after training: cache them and drop the reference sample, so the fitted model stores
        # only its parameters and inference does not recompute power means.
        with torch.no_grad():
            self.net_.M_cached = self.net_.borders().detach()
        self.net_.ref = None
        return self

    def _train(self, Nt, yt, tr, va, rng, epochs):
        """Adam with optional cosine schedule; early stopping on val NLL when va is given.
        Returns the best epoch count."""
        if self.optimizer == "lbfgs":
            return self._train_lbfgs(Nt, yt, tr, va, epochs)
        if self.optimizer == "hybrid":  # short mini-batch Adam warm-up (exploration), then full-batch L-BFGS
            warm = getattr(self, "warm_epochs", 30)
            opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
            for ep in range(warm):
                perm = rng.permutation(tr)
                for i in range(0, len(perm), self.batch_size):
                    b = perm[i:i + self.batch_size]
                    loss = self._loss(Nt[b], yt[b]) + self.l2 * (self.net_.alpha.pow(2).sum() + self.net_.gamma.pow(2).sum()) + L1 * self.net_.alpha.exp().sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            return self._train_lbfgs(Nt, yt, tr, va, epochs)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=self.lr * 0.02)
                 if self.lr_schedule == "cosine" else None)
        best, best_state, bad, best_ep = np.inf, None, 0, epochs
        for ep in range(epochs):
            self.net_.train()
            perm = rng.permutation(tr)
            for i in range(0, len(perm), self.batch_size):
                b = perm[i:i + self.batch_size]
                loss = self._loss(Nt[b], yt[b]) + self.l2 * (self.net_.alpha.pow(2).sum() + self.net_.gamma.pow(2).sum()) + L1 * self.net_.alpha.exp().sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if sched is not None:
                sched.step()
            if va is None:
                continue
            with torch.no_grad():
                vl = self._loss(Nt[va], yt[va]).item()
            if vl < best - 1e-5:
                best, bad, best_ep = vl, 0, ep + 1
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if va is not None:
            self.net_.load_state_dict(best_state)
        self.n_epochs_ = best_ep
        return best_ep

    def _train_lbfgs(self, Nt, yt, tr, va, epochs):
        """Full-batch L-BFGS (strong Wolfe line search); one 'epoch' = one block of `lbfgs_block` iterations.
        Early stopping on validation NLL between blocks. Far fewer Python-level steps than mini-batch Adam."""
        block = self.lbfgs_block
        opt = torch.optim.LBFGS(self.net_.parameters(), lr=1.0, max_iter=block, history_size=20,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)
        trt = torch.as_tensor(tr)
        Ntr, ytr = Nt[trt], yt[trt]

        def closure():
            opt.zero_grad()
            loss = self._loss(Ntr, ytr) + self.l2 * (self.net_.alpha.pow(2).sum() + self.net_.gamma.pow(2).sum()) + L1 * self.net_.alpha.exp().sum()
            loss.backward()
            return loss

        n_blocks = max(1, epochs // block) if va is None else max(1, self.epochs // block)
        best, best_state, bad, best_b = np.inf, None, 0, n_blocks
        pat = max(2, self.patience // block)
        snap = lambda: {k: v.detach().clone() for k, v in self.net_.state_dict().items()}  # noqa: E731
        last_good, n_done = snap(), 0
        with torch.no_grad():
            tl_prev = self._loss(Ntr, ytr).item()
        for b in range(n_blocks):
            self.net_.train()
            opt.step(closure)
            # divergence guard: on (nearly) separable data L-BFGS can push the scales to inf/NaN; roll back to the
            # last finite state and stop (this also protects the final refit, which has no validation set)
            with torch.no_grad():
                tl = self._loss(Ntr, ytr).item()
            finite = np.isfinite(tl) and all(torch.isfinite(p).all() for p in self.net_.parameters())
            # a strong-Wolfe L-BFGS block should not increase the training loss; a large increase signals divergence
            if not finite or tl > tl_prev + 1e-3 * (1 + abs(tl_prev)):
                self.net_.load_state_dict(last_good)
                self.n_restores_ = getattr(self, "n_restores_", 0) + 1
                if self.n_restores_ >= 3:
                    break
                # restart L-BFGS from the last good state with a fresh curvature history
                opt = torch.optim.LBFGS(self.net_.parameters(), lr=1.0, max_iter=block, history_size=20,
                                        line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)
                continue
            last_good, n_done, tl_prev = snap(), b + 1, tl
            if va is None:
                continue
            with torch.no_grad():
                vl = self._loss(Nt[va], yt[va]).item()
            if not np.isfinite(vl):
                break
            if vl < best - 1e-5:
                best, bad, best_b = vl, 0, b + 1
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= pat:
                    break
        if va is not None and best_state is not None:
            self.net_.load_state_dict(best_state)
        elif va is None:
            best_b = n_done
        self.n_epochs_ = max(best_b, 1) * block
        return max(best_b, 1) * block

    def _loss(self, N, y):
        z = self.net_.logits(N)
        if self.task_ == "binary":
            return F.binary_cross_entropy_with_logits(z[:, 0], y.float())
        if self.task_ == "nominal":
            return F.cross_entropy(z, y)
        P = self._ordinal_probs_from_logits(z)
        return F.nll_loss(torch.log(P.clamp_min(1e-9)), y)

    @staticmethod
    def _ordinal_probs_from_logits(z):
        g = torch.sigmoid(z)  # P(y > k), (m, K-1)
        ones = torch.ones(z.shape[0], 1)
        zeros = torch.zeros(z.shape[0], 1)
        cum = torch.cat([ones, g, zeros], 1)
        return (cum[:, :-1] - cum[:, 1:]).clamp_min(0)

    # ------------------------------------------------------------------
    def _N(self, X):
        return torch.as_tensor(self.norm_.transform(np.asarray(X, dtype=float)), dtype=torch.float32)

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            z = self.net_.logits(self._N(X))
            if self.task_ == "binary":
                p1 = torch.sigmoid(z[:, 0])
                P = torch.stack([1 - p1, p1], 1)
            elif self.task_ == "nominal":
                P = torch.softmax(z, 1)
            else:
                P = self._ordinal_probs_from_logits(z)
        P = P.numpy()
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

    def decision_function(self, X):
        self.net_.eval()
        with torch.no_grad():
            return self.net_.score(self._N(X)).numpy()

    # ------------------------------------------------------------------
    def explain(self, X, feature_names=None, head=0):
        """Per-criterion signed contributions (G+ > 0, G- < 0) for each sample."""
        self.net_.eval()
        with torch.no_grad():
            phi = self.net_.contributions(self._N(X))[:, head, :].numpy()
        return phi

    def feature_importances(self, X):
        """Global importance of each criterion on data X: mean absolute centred contribution on the logit scale
        (tau * phi_j), averaged over heads. Same definition as EBM's term importance, so the two are comparable.
        The raw MABAC weights w_j are not used: their scale trades off against the border shape (psi saturates
        when a border lies outside the data), so w_j alone is not identifiable."""
        self.net_.eval()
        with torch.no_grad():
            phi = self.net_.contributions(self._N(X)).numpy() * float(torch.exp(self.net_.log_tau))
        return np.abs(phi - phi.mean(0)).mean(0).mean(0)

    def diagnostics(self):
        """Fit diagnostics reported in the paper: L-BFGS restores, kappa at its bounds, coincident borders,
        and (for learned soft directions) the share of directions close to 1/2."""
        with torch.no_grad():
            k = self.net_.kappa().numpy(); M = np.sort(self.net_.borders().numpy(), -1); s = self.net_.s_full().numpy()
        return dict(n_restores=int(self.n_restores_), kappa_at_upper=float((k >= 199.9).mean()),
                    kappa_at_lower=float((k <= 1.01e-4).mean()),
                    borders_coincident=float((np.diff(M, axis=-1) < 1e-3).mean()) if M.shape[-1] > 1 else 0.0,
                    directions_soft=float(((s > 1e-6) & (s < 1 - 1e-6)).mean()),
                    directions_near_half=float((np.abs(s - 0.5) < 0.1).mean()))

    def borders_in_original_units(self, feature_names=None, head=0):
        """Border locations mapped back to the original feature scale."""
        with torch.no_grad():
            M = self.net_.borders()[head].numpy()  # (d,B) in u-space
            s = self.net_.s_full()[head].numpy()  # (d,B)
            w = self.net_.weights()[head].numpy()
            p = self.net_.p()[head].numpy()
        rows = []
        for j in range(M.shape[0]):
            for k in range(M.shape[1]):
                # u = s*n + (1-s)*(1-n) + 1  =>  n = (u - 2 + s) / (2s - 1); exact for hard directions s in {0,1}
                if abs(2 * s[j, k] - 1) < 0.2:
                    raise ValueError("border not mappable to original units: direction too close to 1/2")
                n = (M[j, k] - 2 + s[j, k]) / (2 * s[j, k] - 1)
                rows.append(dict(feature=feature_names[j] if feature_names is not None else j, border=k,
                                 weight=w[j], benefit=float(s[j, k]), p=p[j, k],
                                 value=self.norm_.inverse(n, j)))
        return rows

    # ------------------------------------------------------------------
    def scorecard(self, feature_names=None, max_points=5, integer=True, X=None, y=None, C=1.0):
        """If-then point rules from the fitted model (monotone tasks, hard directions).

        Each border is replaced by its sharp limit (kappa -> inf): psi -> sign(u - m). A border k of criterion j then
        contributes tau * w_j * beta_k * (+1 if the rule fires else -1), i.e. a constant plus `points` = 2 tau w_j beta_k
        logit units when the rule fires. Rules whose border lies outside the observed range never change and are
        absorbed into the intercept. With integer=True the points are rounded on a scale where the largest rule gets
        `max_points`; the scale (logit units per point) is returned so the card can be turned back into probabilities.
        Returns dict(rules=[...], intercept_points, unit, thresholds)."""
        if self.shape_ != "monotone":
            raise ValueError("scorecards are defined for binary and ordinal (monotone) tasks")
        net = self.net_
        with torch.no_grad():
            tau = float(torch.exp(net.log_tau)); w = net.weights()[0].numpy()
            beta = torch.softmax(net.gamma, dim=-1)[0].numpy()  # (d,B)
            rows = self.borders_in_original_units(feature_names)
            thr = net.thresholds().numpy().ravel()
        rules, const = [], 0.0
        for r in rows:
            j = r["feature"] if feature_names is None else list(feature_names).index(r["feature"])
            k = r["border"]; a = tau * w[j] * beta[j, k]
            const -= a
            vmin, vmax = self.norm_.inverse(0.0, j), self.norm_.inverse(1.0, j)
            if not (vmin < r["value"] < vmax):  # border outside the training range: the rule is constant
                const += 2 * a if ((r["benefit"] == 1) == (r["value"] <= vmin)) else 0.0
                continue
            rules.append(dict(feature=r["feature"], j=j, op=">" if r["benefit"] == 1 else "<", value=float(r["value"]),
                              logit=2 * a))
        # merge coincident rules (same criterion, operator and threshold)
        merged = {}
        for x in rules:
            key = (x["j"], x["op"], round(x["value"], 6))
            if key in merged: merged[key]["logit"] += x["logit"]
            else: merged[key] = dict(x)
        rules = list(merged.values())
        if X is not None and rules:  # merge rules that fire on exactly the same training cases (e.g. binary criteria)
            R0 = self._fire(np.asarray(X, float), rules); groups = {}
            for x, col in zip(rules, R0.T):
                key = (x["j"], col.tobytes())
                if key in groups: groups[key]["logit"] += x["logit"]
                else: groups[key] = dict(x)
            rules = list(groups.values())
            for x in rules:  # report binary criteria as "= 1" / "= 0"
                v = np.unique(np.asarray(X, float)[:, x["j"]])
                if len(v) == 2:
                    x["op"], x["value"] = ("=", float(v[1])) if x["op"] == ">" else ("=", float(v[0]))
        if X is not None and y is not None and rules:
            # optional refit: keep the learned rules (thresholds, directions), re-estimate their points and the
            # intercept/cut-points on the rule indicators by (ordinal) logistic regression with non-negative points
            R = self._fire(np.asarray(X, float), rules)
            yi = np.searchsorted(self.classes_, y)
            from scipy.optimize import minimize
            from scipy.special import expit
            K1 = len(thr); n = R.shape[1]
            def nll(v):
                b = v[:n]; c = np.sort(v[n:]); z = (R @ b)[:, None] - c[None, :]
                Fz = expit(z)  # P(y > k)
                Pc = np.concatenate([np.ones((len(z), 1)), Fz, np.zeros((len(z), 1))], 1)
                p = np.clip(Pc[np.arange(len(z)), yi] - Pc[np.arange(len(z)), yi + 1], 1e-12, 1)
                return -np.log(p).sum() + 0.5 / C * (b ** 2).sum()
            v0 = np.concatenate([[x["logit"] for x in rules], thr - const])
            res = minimize(nll, v0, method="L-BFGS-B", bounds=[(0, None)] * n + [(None, None)] * K1)
            if np.all(np.isfinite(res.x)):
                for x, b in zip(rules, res.x[:n]): x["logit"] = float(b)
                const, thr = 0.0, np.sort(res.x[n:])
        rules = [x for x in rules if x["logit"] > 1e-8]
        unit = (max(x["logit"] for x in rules) / max_points) if (integer and rules) else 1.0
        for x in rules:
            x["points"] = int(round(x["logit"] / unit)) if integer else x["logit"]
        rules = [x for x in rules if x["points"] != 0]
        return dict(rules=rules, intercept=const, unit=unit, thresholds=thr, tau=tau)

    @staticmethod
    def _fire(X, rules):
        ops = {">": np.greater, "<": np.less, "=": np.isclose}
        return np.column_stack([ops[x["op"]](X[:, x["j"]], x["value"]) for x in rules]).astype(float)

    def predict_proba_scorecard(self, X, card):
        """Class probabilities from a scorecard (same link as the full model)."""
        X = np.asarray(X, float)
        S = np.full(len(X), card["intercept"])
        if card["rules"]:
            S += self._fire(X, card["rules"]) @ np.array([x["points"] * card["unit"] for x in card["rules"]])
        z = torch.as_tensor(S[:, None] - card["thresholds"][None, :], dtype=torch.float32)
        if self.task_ == "binary":
            p1 = torch.sigmoid(z[:, 0]); P = torch.stack([1 - p1, p1], 1)
        else:
            P = self._ordinal_probs_from_logits(z)
        P = P.numpy(); return P / P.sum(1, keepdims=True)


# ----------------------------------------------------------------------------------------------
# Classical MABAC used as a classifier (baseline, no learning except the decision threshold)
# ----------------------------------------------------------------------------------------------
class MABACClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, normalisation="minmax"):
        self.normalisation = normalisation

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = unique_labels(y)
        yi = np.searchsorted(self.classes_, y)
        C = len(self.classes_)
        self.norm_ = Normaliser(self.normalisation).fit(X)
        N = np.clip(self.norm_.transform(X), 0, 1)
        Y = np.eye(C)[yi] if C > 2 else yi.reshape(-1, 1).astype(float)
        self.s_ = corr_directions(N, Y)
        self.w_ = entropy_weights(np.where(self.s_[0] > 0.5, N, 1 - N))
        U = self._u(N)  # (m,H,d)
        self.g_ = np.exp(np.log(U).mean(0))  # geometric mean border (H,d)
        S = self._S(N)
        if C == 2:  # threshold maximising balanced accuracy (0 = original MABAC rule)
            cands = np.unique(np.quantile(S[:, 0], np.linspace(0.01, 0.99, 99)))
            best = max(cands, key=lambda t: _bacc(yi, (S[:, 0] > t).astype(int)))
            self.t_ = best
        return self

    def _u(self, N):
        s = self.s_[None]
        return s * N[:, None, :] + (1 - s) * (1 - N[:, None, :]) + 1.0

    def _S(self, N):
        return (self.w_[None, None, :] * (self._u(N) - self.g_[None])).sum(-1)

    def decision_function(self, X):
        return self._S(np.clip(self.norm_.transform(X), 0, 1))

    def predict(self, X):
        S = self.decision_function(X)
        if len(self.classes_) == 2:
            return self.classes_[(S[:, 0] > self.t_).astype(int)]
        return self.classes_[np.argmax(S, 1)]

    def predict_proba(self, X):
        S = self.decision_function(X)
        if len(self.classes_) == 2:
            p = 1 / (1 + np.exp(-20 * (S[:, 0] - self.t_)))
            return np.stack([1 - p, p], 1)
        e = np.exp(20 * (S - S.max(1, keepdims=True)))
        return e / e.sum(1, keepdims=True)


def _bacc(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y, p)

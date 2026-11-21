"""Minimal GMLVQ (Schneider, Biehl & Hammer, 2009) in PyTorch: one prototype per class, full relevance matrix
Lambda = Omega^T Omega (trace-normalised), GLVQ cost mu = (d+ - d-)/(d+ + d-) through a sigmoid."""
import numpy as np, torch
from sklearn.preprocessing import StandardScaler

class GMLVQ:
    def __init__(self, seed=0, epochs=300, lr=0.01, beta=4.0):
        self.seed, self.epochs, self.lr, self.beta = seed, epochs, lr, beta

    def _dist(self, Z):
        O = self.Omega / torch.sqrt((self.Omega ** 2).sum())
        diff = Z[:, None, :] - self.W[None]                 # (m,C,d)
        return ((diff @ O.T) ** 2).sum(-1)                  # (m,C)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.sc = StandardScaler().fit(X)
        Z = torch.as_tensor(self.sc.transform(X), dtype=torch.float32)
        self.classes_ = np.unique(y)
        yi = torch.as_tensor(np.searchsorted(self.classes_, y))
        C, d = len(self.classes_), Z.shape[1]
        self.W = torch.stack([Z[yi == c].mean(0) for c in range(C)]).clone().requires_grad_(True)
        self.Omega = (torch.eye(d) / np.sqrt(d)).requires_grad_(True)
        opt = torch.optim.Adam([self.W, self.Omega], lr=self.lr)
        onehot = torch.nn.functional.one_hot(yi, C).bool()
        for _ in range(self.epochs):
            D = self._dist(Z)
            dp = D[onehot]
            dm = D.masked_fill(onehot, float("inf")).min(1).values
            loss = torch.sigmoid(self.beta * (dp - dm) / (dp + dm + 1e-9)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            D = self._dist(torch.as_tensor(self.sc.transform(X), dtype=torch.float32))
            return torch.softmax(-D, 1).numpy()

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

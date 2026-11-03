"""Model zoo, nested hyper-parameter tuning, CV+ conformal intervals and applicability domain."""
import itertools
import warnings
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from xgboost import XGBRegressor
from common import Prep, balanced_group_folds

warnings.filterwarnings("ignore")
import os
N_JOBS = int(os.environ.get("N_JOBS", 2))


class Scaled:
    """Wraps an estimator with train-fold preprocessing and optional target standardization."""

    def __init__(self, est, scale_X=True, scale_y=False):
        self.est, self.scale_X, self.scale_y = est, scale_X, scale_y

    def fit(self, X, y):
        self.prep = Prep(scale=self.scale_X).fit(X)
        y = np.asarray(y, float)
        self.ym, self.ys = (y.mean(), y.std()) if self.scale_y else (0.0, 1.0)
        self.est.fit(self.prep.transform(X), (y - self.ym) / self.ys)
        return self

    def predict(self, X, return_std=False):
        Z = self.prep.transform(X)
        if return_std:
            m, s = self.est.predict(Z, return_std=True)
            return m * self.ys + self.ym, s * self.ys
        return self.est.predict(Z) * self.ys + self.ym


def grid(d):
    keys = list(d)
    return [dict(zip(keys, v)) for v in itertools.product(*[d[k] for k in keys])]


SPACES = {
    'Mean': [dict()],
    'kNN': grid(dict(n_neighbors=[1, 3, 5, 10, 15])),
    'Ridge': grid(dict(alpha=list(np.logspace(-3, 3, 13)))),
    'SVR': grid(dict(C=[1, 10, 100], gamma=[0.002, 0.01, 0.05], epsilon=[0.05, 0.2])),
    'GPR': [dict()],
    'RF': grid(dict(max_features=[0.2, 0.4, 0.7], min_samples_leaf=[1, 3])),
    'XGB': grid(dict(max_depth=[3, 5, 7], learning_rate=[0.03, 0.08], colsample_bytree=[0.3, 0.7],
                     min_child_weight=[1, 4])),
    'MLP': grid(dict(hidden_layer_sizes=[(128,), (256, 128)], alpha=[1e-3, 1e-1, 1.0])),
}
N_ITER = {'SVR': 10, 'RF': 6, 'XGB': 10, 'MLP': 6, 'kNN': 5, 'Ridge': 13}


def build(name, p, seed=0):
    if name == 'Mean':
        return Scaled(DummyRegressor(), scale_X=False)
    if name == 'kNN':
        return Scaled(KNeighborsRegressor(weights='distance', **p))
    if name == 'Ridge':
        if 'alpha' in p:   # revision 2: alpha tuned with the deployment-specific inner folds
            return Scaled(Ridge(alpha=p['alpha']), scale_y=True)
        return Scaled(RidgeCV(alphas=np.logspace(-3, 3, 13)), scale_y=True)   # original (leave-one-out) variant
    if name == 'SVR':
        return Scaled(SVR(kernel='rbf', **p), scale_y=True)
    if name == 'GPR':
        k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=10.0, length_scale_bounds=(1e-1, 1e3), nu=2.5) \
            + WhiteKernel(0.1, (1e-4, 1.0))
        return Scaled(GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=1,
                                               random_state=seed), scale_y=True)
    if name == 'RF':
        return Scaled(RandomForestRegressor(n_estimators=300, n_jobs=N_JOBS, random_state=seed, **p), scale_X=False)
    if name == 'XGB':
        return Scaled(XGBRegressor(n_estimators=500, subsample=0.8, reg_lambda=1.0, tree_method='hist',
                                   n_jobs=N_JOBS, random_state=seed, verbosity=0, **p), scale_X=False)
    if name == 'MLP':
        return Scaled(MLPRegressor(learning_rate_init=1e-3, max_iter=1500, early_stopping=True,
                                   validation_fraction=0.15, n_iter_no_change=30, random_state=seed, **p),
                      scale_y=True)
    raise ValueError(name)


def inner_folds(n, groups, k, seed):
    rng = np.random.default_rng(seed)
    if groups is None:
        f = np.empty(n, int); f[rng.permutation(n)] = np.arange(n) % k
        return f
    return balanced_group_folds(groups, k, rng)


def tune(name, X, y, groups, seed, k=3):
    space = SPACES[name]
    if len(space) == 1:
        return space[0], np.nan
    rng = np.random.default_rng(seed)
    n_iter = min(N_ITER.get(name, len(space)), len(space))
    cand = [space[i] for i in rng.choice(len(space), n_iter, replace=False)]
    fid = inner_folds(len(y), groups, k, seed + 7)
    best, best_s = None, np.inf
    for p in cand:
        se = 0.0
        for f in range(k):
            tr, va = fid != f, fid == f
            m = build(name, p, seed).fit(X[tr], y[tr])
            se += ((m.predict(X[va]) - y[va]) ** 2).sum()
        s = np.sqrt(se / len(y))
        if s < best_s:
            best, best_s = p, s
    return best, best_s


def knn_distance(Ztr, Zte, k=5, exclude_self=False):
    nn = NearestNeighbors(n_neighbors=k + int(exclude_self)).fit(Ztr)
    d, _ = nn.kneighbors(Zte)
    if exclude_self:
        d = d[:, 1:]
    return d.mean(1)


def cv_plus(name, p, X, y, Xte, groups, seed, alphas=(0.1, 0.32), k=5, Zall=None):
    """CV+ (Barber et al., 2021) prediction intervals plus a normalized (difficulty-scaled)
    cross-conformal variant. groups=None -> random inner folds; otherwise grouped inner folds.
    Zall: (Ztr, Zte) standardized descriptor matrices used for the kNN difficulty score."""
    fid = inner_folds(len(y), groups, k, seed + 11)
    mu_te = np.zeros((len(y), len(Xte)))  # mu_{-k(i)}(x_test) for each calibration point i
    R = np.zeros(len(y))
    s_cal = np.zeros(len(y))
    Ztr, Zte = Zall
    for f in range(k):
        tr, va = fid != f, fid == f
        m = build(name, p, seed).fit(X[tr], y[tr])
        R[va] = np.abs(m.predict(X[va]) - y[va])
        mu_te[va] = m.predict(Xte)[None, :]
        s_cal[va] = knn_distance(Ztr[tr], Ztr[va])
    out = {}
    n = len(y)
    for a in alphas:
        lo_k = int(np.floor(a * (n + 1)))
        hi_k = int(np.ceil((1 - a) * (n + 1)))
        lo_k, hi_k = max(lo_k, 1), min(hi_k, n)
        lo = np.sort(mu_te - R[:, None], axis=0)[lo_k - 1]
        hi = np.sort(mu_te + R[:, None], axis=0)[hi_k - 1]
        out[a] = (lo, hi)
    # normalized cross-conformal: scores R_i / s_i ; s(x) for test relative to the full training set
    beta = np.median(s_cal)
    sc = R / (s_cal + beta)
    s_te = knn_distance(Ztr, Zte) + beta
    outn = {}
    for a in alphas:
        q = np.sort(sc)[min(int(np.ceil((1 - a) * (n + 1))), n) - 1]
        outn[a] = (q * s_te,)
    return out, outn


def tune_folds(name, X, y, folds, seed):
    """Random search over SPACES[name]; score = pooled RMSE over the held-out records of the given inner folds."""
    space = SPACES[name]
    if len(space) == 1:
        return space[0], np.nan
    rng = np.random.default_rng(seed)
    n_iter = min(N_ITER.get(name, len(space)), len(space))
    cand = [space[i] for i in rng.choice(len(space), n_iter, replace=False)]
    best, best_s = None, np.inf
    for p in cand:
        se, nv = 0.0, 0
        for tr, va in folds:
            m = build(name, p, seed).fit(X[tr], y[tr])
            se += ((m.predict(X[va]) - y[va]) ** 2).sum(); nv += len(va)
        s = np.sqrt(se / nv)
        if s < best_s:
            best, best_s = p, s
    return best, best_s

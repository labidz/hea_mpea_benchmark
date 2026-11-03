"""Deployment-specific inner folds and cross-conformal prediction intervals (revision 2).

Inner fold schemes (used both for hyper-parameter tuning and for conformal calibration) mirror the outer protocol:
  random       -> random K-fold
  composition  -> composition-grouped K-fold
  system       -> system-grouped K-fold
  element      -> leave-one-element-out inner folds (greedy, see element_folds)
  temporal     -> forward-chaining by publication year (see forward_chaining_folds)
  external     -> leave-one-element-out inner folds (primary; the external set is defined by element novelty)
Each scheme returns a list of (train_idx, cal_idx) pairs; the cal_idx sets are disjoint but need not cover every
record (records never held out contribute no calibration residual).

Intervals (all use fold-specific models mu_{-k}; Barber et al. 2021, Ann. Stat. 49, 486-507):
  CV+            [ q^-_a{ mu_{-k(i)}(x) - R_i },  q^+_a{ mu_{-k(i)}(x) + R_i } ],   R_i = |y_i - mu_{-k(i)}(x_i)|
  normalized CV+ [ q^-_a{ mu_{-k(i)}(x) - (R_i/s_i) s_{-k(i)}(x) }, q^+_a{ ... + ... } ],
                 s_i = d5(x_i; D_-k) + beta_-k, s_{-k}(x) = d5(x; D_-k) + beta_-k, where D_-k is the fold-k training
                 set, d5 is computed in the winsorized descriptor space fitted on D_-k, and beta_-k is the median
                 leave-self-out d5 of the D_-k records (every quantity is a fold-specific fit)
  group-weighted CV+: as CV+, but each calibration record carries weight 1/n_g of its alloy system g, so that every
                 system contributes equally (two-layer hierarchical exchangeability; Dunn et al. 2023, JASA).
q^-_a: the floor(a(n+1))-th smallest value; q^+_a: the ceil((1-a)(n+1))-th smallest value of the n calibration values;
if ceil((1-a)(n+1)) > n the upper end is +inf (and symmetrically the lower end is -inf).
For weighted quantiles the record count n is replaced by the total weight G (number of systems) and the test point
carries weight 1.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from common import Prep, balanced_group_folds

ZMAX = 5.0


# --------------------------------------------------------------------------- inner fold schemes
def random_folds(n, k, seed):
    rng = np.random.default_rng(seed)
    f = np.empty(n, int); f[rng.permutation(n)] = np.arange(n) % k
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def group_folds(groups, k, seed):
    f = balanced_group_folds(np.asarray(groups), k, np.random.default_rng(seed))
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def element_folds(sys_keys, k, min_share=0.05, max_share=0.6, min_cal=10):
    """Leave-one-element-out inner folds. Eligible elements occur in >= min_share and <= max_share of the records
    (the outer P4 rule, rescaled to the smaller inner set). Elements are taken in order of decreasing frequency;
    fold j holds out the not-yet-assigned records containing element e_j, and its model is trained only on records
    that do not contain e_j (so every calibration residual is an unseen-element residual)."""
    els = [set(s.split('-')) for s in sys_keys]
    n = len(els)
    cnt = pd.Series([e for s in els for e in s]).value_counts()
    elig = [e for e, c in cnt.items() if min_share * n <= c <= max_share * n]
    assigned = np.zeros(n, bool)
    out, used = [], []
    for e in elig:
        has = np.array([e in s for s in els])
        cal = np.where(has & ~assigned)[0]
        if len(cal) < min_cal:
            continue
        out.append((np.where(~has)[0], cal)); used.append(e)
        assigned[cal] = True
        if len(out) == k:
            break
    return out, used


def forward_chaining_folds(years, k):
    """Forward-chaining (rolling-origin) folds: for each of the last k publication years v present in the training
    data, calibrate on the records of year v with a model trained on all records published before v."""
    years = np.asarray(years)
    uy = np.sort(np.unique(years))[::-1][:k]
    return [(np.where(years < v)[0], np.where(years == v)[0]) for v in sorted(uy)], [int(v) for v in sorted(uy)]


def scheme(rec_tr, protocol, k, seed):
    """Inner folds mirroring the deployment scenario. Returns (folds, description)."""
    n = len(rec_tr)
    if protocol == 'random':
        return random_folds(n, k, seed), 'random'
    if protocol == 'composition':
        return group_folds(rec_tr.comp_key.values, k, seed), 'composition-grouped'
    if protocol in ('system', 'external_system'):
        return group_folds(rec_tr.sys_key.values, k, seed), 'system-grouped'
    if protocol in ('element', 'external'):
        f, used = element_folds(rec_tr.sys_key.values, k)
        return f, 'leave-element-out: ' + ','.join(used)
    if protocol == 'temporal':
        f, used = forward_chaining_folds(rec_tr.year.values, k)
        return f, 'forward-chaining: ' + ','.join(map(str, used))
    raise ValueError(protocol)


# --------------------------------------------------------------------------- AD distance
class ADSpace:
    """Composition-descriptor space: train-set standardization, z-scores winsorized to [-ZMAX, ZMAX]."""

    def __init__(self, zmax=ZMAX):
        self.zmax = zmax

    def fit(self, X):
        self.p = Prep(clip=False).fit(X)
        return self

    def transform(self, X):
        Z = self.p.transform(X)
        return np.clip(Z, -self.zmax, self.zmax) if self.zmax else Z


def knn_dist(Ztr, Zq, k=5, exclude_self=False):
    nn = NearestNeighbors(n_neighbors=min(k + int(exclude_self), len(Ztr))).fit(Ztr)
    d, _ = nn.kneighbors(Zq)
    return (d[:, 1:] if exclude_self else d).mean(1)


# --------------------------------------------------------------------------- quantiles
def _lower_upper(lo_vals, hi_vals, a):
    """lo_vals, hi_vals: (n_cal, n_test). Unweighted CV+ order statistics with +-inf when undefined."""
    n = lo_vals.shape[0]
    kl = int(np.floor(a * (n + 1)))
    ku = int(np.ceil((1 - a) * (n + 1)))
    lo = np.sort(lo_vals, axis=0)[kl - 1] if kl >= 1 else np.full(lo_vals.shape[1], -np.inf)
    hi = np.sort(hi_vals, axis=0)[ku - 1] if ku <= n else np.full(hi_vals.shape[1], np.inf)
    return lo, hi


def _weighted_lower_upper(lo_vals, hi_vals, w, a):
    """Weighted analogue: total weight G = sum(w) (one per alloy system), test point weight 1.
    upper = smallest v with sum_{i: hi_i <= v} w_i >= (1-a)(G+1); lower symmetric."""
    G = w.sum(); target = (1 - a) * (G + 1)
    out = []
    for vals, sign in ((hi_vals, 1), (lo_vals, -1)):
        v = sign * vals
        o = np.argsort(v, axis=0)
        vs = np.take_along_axis(v, o, axis=0)
        cw = np.cumsum(w[o], axis=0)
        idx = (cw >= target - 1e-12).argmax(axis=0)
        ok = cw[-1] >= target - 1e-12
        res = np.where(ok, vs[idx, np.arange(v.shape[1])], np.inf)
        out.append(sign * res)
    hi, lo = out
    return lo, hi


# --------------------------------------------------------------------------- cross-conformal intervals
def cross_conformal(fit_predict, X, y, Xte, folds, alphas=(0.1, 0.32), Xc=None, Xcte=None, groups=None, k_ad=5):
    """fit_predict(Xtr, ytr, Xquery_list) -> list of predictions (one per query matrix).
    Returns dict variant -> {alpha: (lo, hi)} for variants 'cvp', 'ncvp' (if Xc given) and 'gw' (if groups given),
    plus diagnostics."""
    cal_idx, R, mu_te, s_cal, s_te, betas = [], [], [], [], [], []
    for tr, cal in folds:
        p_cal, p_te = fit_predict(X[tr], y[tr], [X[cal], Xte])
        cal_idx.append(cal); R.append(np.abs(p_cal - y[cal]))
        mu_te.append(np.repeat(p_te[None, :], len(cal), axis=0))
        if Xc is not None:
            A = ADSpace().fit(Xc[tr]); Ztr = A.transform(Xc[tr])
            beta_k = np.median(knn_dist(Ztr, Ztr, k_ad, exclude_self=True))   # fold-specific offset
            s_cal.append(knn_dist(Ztr, A.transform(Xc[cal]), k_ad) + beta_k)
            s_te.append(np.repeat(knn_dist(Ztr, A.transform(Xcte), k_ad)[None, :] + beta_k, len(cal), axis=0))
            betas.append(beta_k)
    cal_idx = np.concatenate(cal_idx); R = np.concatenate(R); mu_te = np.vstack(mu_te)
    out = {'cvp': {}, 'diag': dict(n_cal=len(cal_idx), n_train=len(y))}
    for a in alphas:
        out['cvp'][a] = _lower_upper(mu_te - R[:, None], mu_te + R[:, None], a)
    if Xc is not None:
        s_cal = np.concatenate(s_cal); s_te = np.vstack(s_te)
        sc = R / s_cal
        half = sc[:, None] * s_te
        out['ncvp'] = {a: _lower_upper(mu_te - half, mu_te + half, a) for a in alphas}
        out['diag']['beta'] = float(np.mean(betas))
    if groups is not None:
        g = np.asarray(groups)[cal_idx]
        _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv]
        out['gw'] = {a: _weighted_lower_upper(mu_te - R[:, None], mu_te + R[:, None], w, a) for a in alphas}
        out['diag']['n_groups'] = len(cnt)
    return out

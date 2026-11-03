"""Unit tests for conformal.py (run: python -m pytest tests/)."""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/tests/', 1)[0])
from sklearn.linear_model import LinearRegression
from conformal import (_lower_upper, _weighted_lower_upper, cross_conformal, random_folds, element_folds,
                       forward_chaining_folds)


def fp(a, b, qs):
    m = LinearRegression().fit(a, b)
    return [m.predict(q) for q in qs]


def test_weighted_equals_unweighted_for_unit_weights():
    rng = np.random.default_rng(0)
    lo = rng.normal(size=(50, 7)); hi = lo + rng.uniform(0, 2, (50, 7))
    for a in (0.1, 0.32):
        l1, h1 = _lower_upper(lo, hi, a); l2, h2 = _weighted_lower_upper(lo, hi, np.ones(50), a)
        assert np.allclose(l1, l2) and np.allclose(h1, h2)


def test_infinite_endpoints_for_tiny_calibration_sets():
    lo = np.zeros((5, 1)); hi = np.ones((5, 1))
    l, h = _lower_upper(lo, hi, 0.1)          # ceil(0.9 * 6) = 6 > 5
    assert np.isinf(h).all() and np.isinf(l).all()


def test_marginal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(1); cov = {'cvp': [], 'ncvp': [], 'gw': []}
    for r in range(300):
        X = rng.normal(size=(120, 3)); y = X @ [1, 2, 3] + rng.normal(size=120) * (1 + np.abs(X[:, 2]))
        Xt = rng.normal(size=(1, 3)); yt = Xt @ [1, 2, 3] + rng.normal(size=1) * (1 + np.abs(Xt[:, 2]))
        o = cross_conformal(fp, X, y, Xt, random_folds(120, 5, r), alphas=(0.1,), Xc=X, Xcte=Xt, groups=np.arange(120) // 3)
        for k in cov:
            l, h = o[k][0.1]; cov[k].append(bool((l <= yt) & (yt <= h)))
    for k, v in cov.items():
        assert np.mean(v) >= 0.85, (k, np.mean(v))   # nominal 0.90; K-fold CV+ bound 1 - 2a - sqrt(2/n) is lower


def test_element_folds_hold_out_whole_elements():
    keys = ['Al-Co-Cr', 'Co-Cr-Ni', 'Al-Ni-Ti', 'Cr-Fe-Ni', 'Fe-Ni-Ti', 'Al-Fe-Ti'] * 10
    folds, used = element_folds(keys, 3, min_share=0.1, max_share=0.7, min_cal=1)
    for (tr, cal), e in zip(folds, used):
        assert all(e not in keys[i].split('-') for i in tr) and all(e in keys[i].split('-') for i in cal)


def test_forward_chaining_trains_on_the_past_only():
    years = np.array([2010, 2011, 2012, 2013, 2014] * 4)
    folds, used = forward_chaining_folds(years, 3)
    for (tr, cal), v in zip(folds, used):
        assert years[tr].max() < v and (years[cal] == v).all()

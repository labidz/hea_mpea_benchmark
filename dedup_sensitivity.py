"""Sensitivity: does de-duplication alone remove the optimism of random CV?
Hardness records are aggregated to one record per (composition, processing) (median HV), then evaluated with
random and system-grouped 5-fold CV (3 repeats) using fixed-hyper-parameter XGB and GPR."""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from common import load, feature_columns, DEFAULT_BLOCKS, Prep, balanced_group_folds, metrics
from ablation import xgb_fixed


def main():
    rec, X = load('HV')
    cols = feature_columns(X, DEFAULT_BLOCKS)
    df = pd.concat([rec.reset_index(drop=True), X[cols].reset_index(drop=True)], axis=1)
    agg = df.groupby(['comp_key', 'proc'], as_index=False).agg({**{c: 'first' for c in cols}, 'target': 'median', 'sys_key': 'first'})
    out = []
    for tag, d in [('original', df), ('deduplicated', agg)]:
        Xv, y = d[cols].values.astype(float), d.target.values.astype(float)
        for proto in ('random', 'system'):
            for r in range(3):
                rng = np.random.default_rng(2026 + r)
                if proto == 'random':
                    f = np.empty(len(y), int); f[rng.permutation(len(y))] = np.arange(len(y)) % 5
                else:
                    f = balanced_group_folds(d.sys_key.values, 5, rng)
                pred = {'XGB': np.zeros(len(y)), 'GPR': np.zeros(len(y))}
                for k in range(5):
                    tr, te = f != k, f == k
                    pred['XGB'][te] = xgb_fixed(2).fit(Xv[tr], y[tr]).predict(Xv[te])
                    p = Prep().fit(Xv[tr]); ym, ys = y[tr].mean(), y[tr].std()
                    kern = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(10.0, (1e-1, 1e3), nu=2.5) + WhiteKernel(0.1, (1e-4, 1.0))
                    g = GaussianProcessRegressor(kern, n_restarts_optimizer=1, random_state=0).fit(p.transform(Xv[tr]), (y[tr] - ym) / ys)
                    pred['GPR'][te] = g.predict(p.transform(Xv[te])) * ys + ym
                for m, pv in pred.items():
                    mt = metrics(y, pv)
                    out.append(dict(data=tag, n=len(y), protocol=proto, rep=r, model=m, RMSE=mt['RMSE'], R2=mt['R2']))
    res = pd.DataFrame(out)
    s = res.groupby(['data', 'n', 'protocol', 'model'])[['RMSE', 'R2']].agg(['mean', 'std']).round(3)
    print(s)
    res.to_csv('/home/claude/hea/results/tables/dedup_sensitivity.csv', index=False)


if __name__ == '__main__':
    main()

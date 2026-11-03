"""Feature-set ablation on the frozen splits (fixed hyper-parameters, no tuning).
Models: XGB (fixed), GPR (ML-II kernel fit), kNN (k=5, distance-weighted).
Prints one line per (dataset, variant, model, split): A|ds|variant|model|label|protocol|n|MAE|RMSE|R2|rho
"""
import json
import sys
import time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from xgboost import XGBRegressor

VARIANTS = {
    'FRAC': ('FRAC', 'PROC', 'TEST'),
    'PHYS': ('PHYS', 'PROC', 'TEST'),
    'MAG': ('MAG', 'PROC', 'TEST'),
    'PHYS+MAG': ('PHYS', 'MAG', 'PROC', 'TEST'),
    'ALL': ('FRAC', 'PHYS', 'MAG', 'PROC', 'TEST'),
    'ALL-noPROC': ('FRAC', 'PHYS', 'MAG', 'TEST'),
    'ALL+PHASE': ('FRAC', 'PHYS', 'MAG', 'PROC', 'TEST', 'PHASE'),
}


def xgb_fixed(n_jobs):
    return XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                        min_child_weight=2, reg_lambda=1.0, tree_method='hist', n_jobs=n_jobs, random_state=0,
                        verbosity=0)


def run_ablation(load, feature_columns, Prep, metrics, data_dir, datasets=('HV', 'YS'), n_jobs=4, out=print):
    splits = json.load(open(f"{data_dir}/splits.json"))
    for ds in datasets:
        rec, Xdf = load(ds, data_dir)
        y = rec.target.values.astype(float)
        rid2i = {r: i for i, r in enumerate(rec.rid.values)}
        for vname, blocks in VARIANTS.items():
            cols = feature_columns(Xdf, blocks)
            X = Xdf[cols].values.astype(float)
            t0 = time.time()
            for sp in splits[ds]:
                te = np.array([rid2i[r] for r in sp['test']])
                mask = np.ones(len(y), bool); mask[te] = False
                tr = np.where(mask)[0]
                prep = Prep().fit(X[tr])
                Ztr, Zte = prep.transform(X[tr]), prep.transform(X[te])
                ym, ys = y[tr].mean(), y[tr].std()
                preds = {}
                m = xgb_fixed(n_jobs).fit(X[tr], y[tr]); preds['XGB'] = m.predict(X[te])
                k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=10.0, length_scale_bounds=(1e-1, 1e3), nu=2.5) \
                    + WhiteKernel(0.1, (1e-4, 1.0))
                g = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=1, random_state=0).fit(Ztr, (y[tr] - ym) / ys)
                preds['GPR'] = g.predict(Zte) * ys + ym
                kn = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(Ztr, y[tr]); preds['kNN'] = kn.predict(Zte)
                for mn, p in preds.items():
                    mt = metrics(y[te], p)
                    out("A|" + "|".join([ds, vname, mn, sp['label'], sp['protocol'], str(mt['n'])] +
                                        [f"{mt[k]:.4f}" for k in ('MAE', 'RMSE', 'R2', 'rho')]))
            out(f"# {ds} {vname} n_features={len(cols)} done in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    sys.path.insert(0, '/home/claude/hea')
    from common import load, feature_columns, Prep, metrics, DATA
    ds = tuple(sys.argv[1].split(',')) if len(sys.argv) > 1 else ('HV',)
    run_ablation(load, feature_columns, Prep, metrics, DATA, datasets=ds, n_jobs=2)

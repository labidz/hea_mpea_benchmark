"""External validation: train on all MPEA hardness records, predict the independent GE refractory dataset
(Detor et al., Data in Brief 2022). Same tuning / CV+ machinery as bench.py (inner folds grouped by alloy system)."""
import json
import os
import time
import numpy as np
import pandas as pd
from common import load, feature_columns, DEFAULT_BLOCKS, Prep
from models import build, tune, cv_plus, knn_distance
from bench import UQ_MODELS, ALPHAS, RES

MODELS = ['Mean', 'kNN', 'Ridge', 'SVR', 'GPR', 'RF', 'XGB', 'MLP']


def main():
    rec, Xdf = load('HV')
    rge, Xge = load('GE')
    cols = feature_columns(Xdf, DEFAULT_BLOCKS)
    comp_cols = feature_columns(Xdf, ('FRAC', 'PHYS', 'MAG'))
    X, Xte = Xdf[cols].values, Xge[cols].values
    y, yte = rec.target.values.astype(float), rge.target.values.astype(float)
    grp = rec.sys_key.values
    pc = Prep(clip=False).fit(Xdf[comp_cols].values)
    Ztr, Zte = pc.transform(Xdf[comp_cols].values), pc.transform(Xge[comp_cols].values)
    ad = knn_distance(Ztr, Zte)
    seed = (sum(map(ord, 'external-GE')) * 7919) % (2 ** 31)
    rows = []
    for name in MODELS:
        t0 = time.time()
        p, _ = tune(name, X, y, grp, seed)
        m = build(name, p, seed).fit(X, y)
        out = pd.DataFrame({'dataset': 'HV', 'protocol': 'external', 'label': 'external-GE', 'rep': 0, 'fold': 0,
                            'rid': rge.rid.values, 'model': name, 'y': yte, 'ad_knn': ad, 'n_train': len(y)})
        if name == 'GPR':
            out['yhat'], out['gp_sd'] = m.predict(Xte, return_std=True)
        else:
            out['yhat'] = m.predict(Xte)
        if name in UQ_MODELS:
            for vname, g in [('naive', None), ('matched', grp)]:
                iv, ivn = cv_plus(name, p, X, y, Xte, g, seed, alphas=ALPHAS, Zall=(Ztr, Zte))
                for a in ALPHAS:
                    tag = int(round((1 - a) * 100))
                    out[f'cvp_{vname}_lo{tag}'], out[f'cvp_{vname}_hi{tag}'] = iv[a]
                    out[f'ncp_{vname}_hw{tag}'] = ivn[a][0]
        out['sec'] = time.time() - t0
        rows.append(out)
        e = out.yhat - out.y
        print(f"external-GE {name:<6} RMSE={np.sqrt((e**2).mean()):7.1f} MAE={e.abs().mean():7.1f} "
              f"bias={e.mean():7.1f} params={p} t={time.time()-t0:.1f}s", flush=True)
    pd.concat(rows).to_csv(f"{RES}/HV_external_pred.csv", index=False)


if __name__ == '__main__':
    main()

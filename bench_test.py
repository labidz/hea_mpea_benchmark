"""Main benchmark: models x evaluation protocols with nested tuning, CV+ conformal intervals and AD.

usage: python bench.py HV [models...]
Writes results/{ds}_pred.csv.gz (per test record), results/{ds}_params.jsonl; resumable.
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from common import load, make_splits, feature_columns, DEFAULT_BLOCKS, inner_groups, Prep
from models import build, tune, cv_plus, knn_distance

RES = "/home/claude/hea/results_test"
os.makedirs(RES, exist_ok=True)
UQ_MODELS = ('XGB', 'RF', 'GPR')
ALPHAS = (0.1, 0.32)


def run(ds, models, only_protocols=None):
    rec, Xdf = load(ds)
    cols = feature_columns(Xdf, DEFAULT_BLOCKS)
    comp_cols = feature_columns(Xdf, ('FRAC', 'PHYS', 'MAG'))
    X = Xdf[cols].values
    Xc = Xdf[comp_cols].values
    y = rec.target.values.astype(float)
    splits = make_splits(rec, ds)
    pred_path = f"{RES}/{ds}_pred.csv"
    done = set()
    if os.path.exists(pred_path):
        d = pd.read_csv(pred_path, usecols=['label', 'model'])
        done = set(zip(d.label, d.model))
    print(f"[{ds}] n={len(y)} features={len(cols)} splits={len(splits)}", flush=True)
    for sp in splits:
        if only_protocols and sp['protocol'] not in only_protocols:
            continue
        tr, te = sp['train_idx'], sp['test_idx']
        rtr = rec.iloc[tr]
        grp = inner_groups(rtr, sp['protocol'])
        pc = Prep(clip=False).fit(Xc[tr])
        Ztr, Zte = pc.transform(Xc[tr]), pc.transform(Xc[te])
        ad = knn_distance(Ztr, Zte)
        seed = abs(hash(sp['label'])) % (2 ** 31) if False else (sum(map(ord, sp['label'])) * 7919) % (2 ** 31)
        for name in models:
            if (sp['label'], name) in done:
                continue
            t0 = time.time()
            p, inner_rmse = tune(name, X[tr], y[tr], grp, seed)
            m = build(name, p, seed).fit(X[tr], y[tr])
            out = pd.DataFrame({'dataset': ds, 'protocol': sp['protocol'], 'label': sp['label'], 'rep': sp['rep'],
                                'fold': sp['fold'], 'rid': rec.rid.values[te], 'model': name, 'y': y[te],
                                'ad_knn': ad, 'n_train': len(tr)})
            if name == 'GPR':
                mu, sd = m.predict(X[te], return_std=True)
                out['yhat'] = mu
                out['gp_sd'] = sd
            else:
                out['yhat'] = m.predict(X[te])
            if name in UQ_MODELS:
                variants = [('naive', None)]
                if sp['protocol'] != 'random':
                    variants.append(('matched', grp))
                for vname, g in variants:
                    iv, ivn = cv_plus(name, p, X[tr], y[tr], X[te], g, seed, alphas=ALPHAS, Zall=(Ztr, Zte))
                    for a in ALPHAS:
                        tag = int(round((1 - a) * 100))
                        out[f'cvp_{vname}_lo{tag}'], out[f'cvp_{vname}_hi{tag}'] = iv[a]
                        out[f'ncp_{vname}_hw{tag}'] = ivn[a][0]
            out['sec'] = time.time() - t0
            out.to_csv(pred_path, mode='a', header=not os.path.exists(pred_path), index=False)
            with open(f"{RES}/{ds}_params.jsonl", 'a') as fh:
                fh.write(json.dumps({'label': sp['label'], 'model': name, 'params': {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
                                     'inner_rmse': None if np.isnan(inner_rmse) else float(inner_rmse)}) + "\n")
            e = out.yhat - out.y
            print(f"  {sp['label']:<22} {name:<6} n_te={len(te):4d} RMSE={np.sqrt((e**2).mean()):7.1f} "
                  f"MAE={e.abs().mean():7.1f} t={time.time()-t0:6.1f}s", flush=True)


if __name__ == '__main__':
    ds = sys.argv[1]
    models = sys.argv[2].split(',') if len(sys.argv) > 2 else ['Mean', 'kNN', 'Ridge', 'SVR', 'GPR', 'RF', 'XGB', 'MLP']
    prot = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    run(ds, models, prot)

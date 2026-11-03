"""Revision-2 benchmark runs with deployment-specific inner folds and corrected cross-conformal intervals.

usage: python bench_v2.py DS JOB      JOB in {shift, ridge, xgbuq}
  shift : P4, P5 (and P6 for HV) for all tuned models, with element-grouped (P4, P6) or forward-chaining (P5) inner
          folds for tuning and calibration; P6 additionally calibrated with system-grouped folds (sensitivity).
  ridge : Ridge on P1-P3 with alpha tuned on the protocol-matched inner folds (previously leave-one-out RidgeCV).
  xgbuq : XGBoost on P1-P3 (hyper-parameters as in the original runs; tuning folds unchanged) with the corrected
          normalized CV+ and group-weighted CV+ intervals.
Output: results/v2/{DS}_pred.csv (appended, resumable) and results/v2/{DS}_params.jsonl.
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from common import load, make_splits, feature_columns, DEFAULT_BLOCKS
from models import build, tune_folds
from conformal import scheme, random_folds, cross_conformal, ADSpace, knn_dist

RES = "/home/claude/hea/results"
OUT = f"{RES}/v2"
os.makedirs(OUT, exist_ok=True)
ALPHAS = (0.1, 0.32)
UQ = {'HV': ('XGB', 'RF', 'GPR'), 'YS': ('XGB', 'GPR')}
TUNED = ['kNN', 'Ridge', 'SVR', 'GPR', 'MLP', 'RF', 'XGB']


def seed_of(label):
    return (sum(map(ord, label)) * 7919) % (2 ** 31)


def v1_params(ds):
    P = {}
    for l in open(f"{RES}/{ds}_params.jsonl"):
        j = json.loads(l)
        p = j['params']
        if 'hidden_layer_sizes' in p:
            p['hidden_layer_sizes'] = tuple(p['hidden_layer_sizes'])
        P[(j['label'], j['model'])] = p
    return P


def jobs_for(ds, job):
    rec, Xdf = load(ds)
    sp = make_splits(rec, ds)
    if job == 'shift':
        J = [(s['label'], s['protocol'], s['train_idx'], s['test_idx'], None) for s in sp if s['protocol'] in ('element', 'temporal')]
        if ds == 'HV':
            J.append(('external-GE', 'external', np.arange(len(rec)), None, 'GE'))
        return J, TUNED
    J = [(s['label'], s['protocol'], s['train_idx'], s['test_idx'], None) for s in sp
         if s['protocol'] in ('random', 'composition', 'system')]
    return J, (['Ridge'] if job == 'ridge' else ['XGB'])


def run(ds, job):
    rec, Xdf = load(ds)
    cols = feature_columns(Xdf, DEFAULT_BLOCKS); ccols = feature_columns(Xdf, ('FRAC', 'PHYS', 'MAG'))
    X, Xc, y = Xdf[cols].values, Xdf[ccols].values, rec.target.values.astype(float)
    if ds == 'HV':
        rge, Xge = load('GE')
    P1 = v1_params(ds)
    J, models = jobs_for(ds, job)
    path = f"{OUT}/{ds}_pred.csv"
    from v2io import done_pairs
    done = done_pairs(path) if os.path.exists(path) else set()
    for label, proto, tr, te, ext in J:
        if ext is None:
            Xte, Xcte, yte, rid = X[te], Xc[te], y[te], rec.rid.values[te]
        else:
            Xte, Xcte, yte, rid = Xge[cols].values, Xge[ccols].values, rge.target.values.astype(float), rge.rid.values
        rtr = rec.iloc[tr]
        seed = seed_of(label)
        tune_f, tdesc = scheme(rtr, proto, 3, seed + 7)
        cal_f, cdesc = scheme(rtr, proto, 5, seed + 11)
        naive_f = random_folds(len(tr), 5, seed + 11)
        variants = [('naive', naive_f)] + ([('matched', cal_f)] if proto != 'random' else [])
        if proto == 'external':
            sys_f, _ = scheme(rtr, 'external_system', 5, seed + 11)
            variants.append(('msys', sys_f))
        A = ADSpace().fit(Xc[tr]); ad = knn_dist(A.transform(Xc[tr]), A.transform(Xcte))
        for name in models:
            if (label, name) in done:
                continue
            t0 = time.time()
            if job == 'xgbuq':
                p, inner = P1[(label, name)], np.nan
            else:
                p, inner = tune_folds(name, X[tr], y[tr], tune_f, seed)
            m = build(name, p, seed).fit(X[tr], y[tr])
            out = pd.DataFrame({'dataset': ds, 'protocol': proto, 'label': label, 'rid': rid, 'model': name, 'y': yte,
                                'ad': ad, 'n_train': len(tr)})
            if name == 'GPR':
                out['yhat'], out['gp_sd'] = m.predict(Xte, return_std=True)
            else:
                out['yhat'] = m.predict(Xte)
            diag = {}
            if name in UQ[ds] and (job != 'ridge'):
                def fp(a, b, qs, name=name, p=p):
                    mm = build(name, p, seed).fit(a, b)
                    return [mm.predict(q) for q in qs]
                for vn, folds in variants:
                    o = cross_conformal(fp, X[tr], y[tr], Xte, folds, ALPHAS, Xc[tr], Xcte, groups=rtr.sys_key.values)
                    diag[vn] = o['diag']
                    for a in ALPHAS:
                        tag = int(round((1 - a) * 100))
                        for kind in ('cvp', 'ncvp', 'gw'):
                            lo, hi = o[kind][a]
                            out[f'{vn}_{kind}_lo{tag}'], out[f'{vn}_{kind}_hi{tag}'] = lo, hi
            out['sec'] = time.time() - t0
            out.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
            with open(f"{OUT}/{ds}_params.jsonl", 'a') as fh:
                fh.write(json.dumps({'label': label, 'model': name, 'job': job,
                                     'params': {k: (list(v) if isinstance(v, tuple) else (float(v) if isinstance(v, np.floating) else v)) for k, v in p.items()},
                                     'inner_rmse': None if np.isnan(inner) else float(inner), 'tune_folds': tdesc,
                                     'cal_folds': cdesc, 'diag': diag}, default=float) + "\n")
            e = out.yhat - out.y
            print(f"{ds} {label:<20} {name:<6} RMSE={np.sqrt((e ** 2).mean()):7.1f} p={p} cal=[{cdesc}] "
                  f"t={time.time() - t0:5.1f}s", flush=True)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])

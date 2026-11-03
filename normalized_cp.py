"""Applicability-domain distance (winsorized) and normalized cross-conformal intervals for XGB.

AD space: composition features (FRAC, PHYS, MAG), train-fold standardization, then z-scores winsorized to [-5, 5]
so that near-constant descriptors (an element present in one or two training alloys) cannot dominate the distance.
Normalized CV+: residuals of the K=5 inner fold models (same folds and tuned hyper-parameters as bench.py) are
scaled by s_i = d5_i + beta; interval half-width = q_{1-a}(R/s) * s(x).
Output: results/{ds}_ncp.csv with ad_w, ad_raw and ncp_{naive,matched}_hw{90,68}."""
import json
import sys
import numpy as np
import pandas as pd
from common import load, feature_columns, DEFAULT_BLOCKS, Prep, make_splits, inner_groups
from models import build, inner_folds, knn_distance

RES = "/home/claude/hea/results"
ZMAX = 5.0


class ADSpace:
    def fit(self, X):
        self.p = Prep(clip=False).fit(X)
        return self

    def transform(self, X):
        return np.clip(self.p.transform(X), -ZMAX, ZMAX)


def params_for(ds):
    P = {}
    for l in open(f"{RES}/{ds}_params.jsonl"):
        j = json.loads(l)
        if j['model'] == 'XGB':
            p = j['params']
            P[j['label']] = p
    return P


def run(ds, alphas=(0.1, 0.32)):
    rec, Xdf = load(ds)
    cols = feature_columns(Xdf, DEFAULT_BLOCKS); ccols = feature_columns(Xdf, ('FRAC', 'PHYS', 'MAG'))
    X, Xc, y = Xdf[cols].values, Xdf[ccols].values, rec.target.values.astype(float)
    splits = make_splits(rec, ds)
    jobs = [(sp['label'], sp['protocol'], sp['train_idx'], sp['test_idx'], None) for sp in splits]
    if ds == 'HV':
        rge, Xge = load('GE')
        jobs.append(('external-GE', 'external', np.arange(len(y)), None, (Xge[cols].values, Xge[ccols].values, rge.rid.values)))
    P = params_for(ds)
    if ds == 'HV':
        import ast
        for l in open(f"{RES}/external_log.txt"):
            if l.startswith('external-GE XGB'):
                P['external-GE'] = ast.literal_eval(l.split('params=')[1].split(' t=')[0])
    out = []
    for label, proto, tr, te, ext in jobs:
        if ext is None:
            Xte, Xcte, rid = X[te], Xc[te], rec.rid.values[te]
        else:
            Xte, Xcte, rid = ext
        p = P[label]
        seed = (sum(map(ord, label)) * 7919) % (2 ** 31)
        A = ADSpace().fit(Xc[tr]); Ztr, Zte = A.transform(Xc[tr]), A.transform(Xcte)
        raw = Prep(clip=False).fit(Xc[tr]); d_raw = knn_distance(raw.transform(Xc[tr]), raw.transform(Xcte))
        d_te = knn_distance(Ztr, Zte)
        row = pd.DataFrame({'dataset': ds, 'label': label, 'protocol': proto, 'rid': rid, 'ad_w': d_te, 'ad_raw': d_raw})
        full = build('XGB', p, seed).fit(X[tr], y[tr])
        row['yhat_check'] = full.predict(Xte)
        gproto = 'system' if proto == 'external' else proto
        variants = [('naive', None)] + ([('matched', inner_groups(rec.iloc[tr], gproto))] if proto != 'random' else [])
        for vn, g in variants:
            fid = inner_folds(len(tr), g, 5, seed + 11)
            R = np.zeros(len(tr)); s = np.zeros(len(tr))
            for f in range(5):
                a, b = fid != f, fid == f
                m = build('XGB', p, seed).fit(X[tr][a], y[tr][a])
                R[b] = np.abs(m.predict(X[tr][b]) - y[tr][b])
                s[b] = knn_distance(Ztr[a], Ztr[b])
            beta = np.median(s)
            sc = np.sort(R / (s + beta)); n = len(sc)
            for al in alphas:
                q = sc[min(int(np.ceil((1 - al) * (n + 1))), n) - 1]
                row[f'ncp_{vn}_hw{int(round((1 - al) * 100))}'] = q * (d_te + beta)
        out.append(row)
        print(f"{ds} {label:<22} median ad_w={np.median(d_te):.2f}", flush=True)
    pd.concat(out).to_csv(f"{RES}/{ds}_ncp.csv", index=False)


if __name__ == '__main__':
    for ds in sys.argv[1].split(','):
        run(ds)

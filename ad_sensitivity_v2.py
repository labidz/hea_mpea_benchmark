"""Revision 2: sensitivity of the applicability-domain (AD) analysis to the neighbourhood size k and to the descriptor
representation. For every split, the mean distance from each test record to its k nearest training records is computed
in six representations (fitted on the training records only) and related to the absolute XGBoost error of the final
(revision-2) predictions.
Reported per (dataset, representation, k):
  rho_pooled      Spearman rho(distance, |error|) pooled over the non-random protocols
  rho_<protocol>  Spearman rho within each protocol (pooled over its folds)
  rho_population  Spearman rho between the protocol-median distance and the protocol RMSE (P1-P6)
  retained50      RMSE of the 50% of P3-P5 predictions with the smallest distance, divided by the target SD
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from common import load, feature_columns, Prep

R = "/home/claude/hea/results"
KS = (1, 3, 5, 10, 20)
SPACES = {
    'all composition, winsorized (default)': ('FRAC', 'PHYS', 'MAG'),
    'all composition, unwinsorized': ('FRAC', 'PHYS', 'MAG'),
    'alloy theory (PHYS)': ('PHYS',),
    'Magpie (MAG)': ('MAG',),
    'element fractions (L1)': ('FRAC',),
    'PCA-10 of all composition': ('FRAC', 'PHYS', 'MAG'),
}


def dists(A_tr, A_te, name):
    if name.startswith('element fractions'):
        nn = NearestNeighbors(n_neighbors=max(KS), metric='manhattan').fit(A_tr)
        d = nn.kneighbors(A_te)[0]
    else:
        pr = Prep(clip=False).fit(A_tr); Ztr, Zte = pr.transform(A_tr), pr.transform(A_te)
        if 'winsorized (default)' in name:
            Ztr, Zte = np.clip(Ztr, -5, 5), np.clip(Zte, -5, 5)
        if name.startswith('PCA'):
            pca = PCA(10, random_state=0).fit(Ztr); Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)
        d = NearestNeighbors(n_neighbors=max(KS)).fit(Ztr).kneighbors(Zte)[0]
    return {k: d[:, :k].mean(1) for k in KS}


def run(ds):
    rec, X = load(ds)
    acc = pd.read_csv(f"{R}/final/{ds}_acc.csv"); acc = acc[acc.model == 'XGB']
    rid2i = {r: i for i, r in enumerate(rec.rid.values)}
    splits = json.load(open("/home/claude/hea/data/splits.json"))[ds]
    jobs = [(s['label'], s['protocol'], np.array([rid2i[r] for r in s['test']])) for s in splits]
    if ds == 'HV':
        jobs.append(('external-GE', 'external', None))
        rge, Xge = load('GE')
    rows = []
    for label, proto, te in jobs:
        if te is None:
            tr = np.arange(len(rec)); te_rid = rge.rid.values; Xq = Xge
        else:
            m = np.ones(len(rec), bool); m[te] = False; tr = np.where(m)[0]; te_rid = rec.rid.values[te]; Xq = X.iloc[te]
        p = acc[acc.label == label].set_index('rid').loc[te_rid]
        err = (p.yhat - p.y).abs().values
        for name, blocks in SPACES.items():
            cols = feature_columns(X, blocks)
            D = dists(X[cols].values[tr].astype(float), Xq[cols].values.astype(float), name)
            for k, d in D.items():
                rows.append(pd.DataFrame({'protocol': proto, 'label': label, 'space': name, 'k': k, 'rid': te_rid,
                                          'd': d, 'abs_err': err}))
    df = pd.concat(rows)
    sd = rec.target.std()
    out = []
    for (space, k), g in df.groupby(['space', 'k']):
        r = dict(dataset=ds, space=space, k=k)
        nr = g[g.protocol != 'random']
        r['rho_pooled'] = spearmanr(nr.d, nr.abs_err).correlation
        for proto, h in g.groupby('protocol'):
            r[f'rho_{proto}'] = spearmanr(h.d, h.abs_err).correlation
        med = g.groupby('protocol').d.median()
        rm = g.groupby('protocol').apply(lambda h: np.sqrt((h.abs_err ** 2).mean()), include_groups=False)
        r['rho_population'] = spearmanr(med, rm[med.index]).correlation
        s = g[g.protocol.isin(['system', 'element', 'temporal'])].sort_values('d')
        keep = s.iloc[: len(s) // 2]
        r['retained50'] = np.sqrt((keep.abs_err ** 2).mean()) / sd
        r['all_P3P5'] = np.sqrt((s.abs_err ** 2).mean()) / sd
        r['oracle50'] = np.sqrt((np.sort(s.abs_err.values)[: len(s) // 2] ** 2).mean()) / sd
        out.append(r)
    return pd.DataFrame(out)


if __name__ == '__main__':
    import sys
    res = pd.concat([run(ds) for ds in sys.argv[1].split(',')])
    res.to_csv(f"{R}/tables/ad_sensitivity_v2.csv", index=False)
    pd.set_option('display.width', 250)
    print(res.round(3).to_string())

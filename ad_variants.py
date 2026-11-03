"""Applicability-domain variants: which distance best ranks prediction errors?
For every split, distances from each test record to its 5 nearest training records are computed in several
spaces (fit on the training fold only) and correlated (Spearman) with the absolute XGB error, within protocol."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from common import load, feature_columns, Prep
from models import knn_distance
from io_utils import read_pred

RES = "/home/claude/hea/results"
SPACES = {
    'all composition, winsorized |z|<=5 (reported)': ('FRAC', 'PHYS', 'MAG'),
    'all composition, unwinsorized': ('FRAC', 'PHYS', 'MAG'),
    'alloy-theory (PHYS)': ('PHYS',),
    'Magpie (MAG)': ('MAG',),
    'composition fractions (L1)': ('FRAC',),
    'PCA-10 of all': ('FRAC', 'PHYS', 'MAG'),
}


def run(ds):
    rec, X = load(ds)
    pred = read_pred(f"{RES}/{ds}_pred.csv")
    pred = pred[pred.model == 'XGB']
    rid2i = {r: i for i, r in enumerate(rec.rid.values)}
    splits = json.load(open("/home/claude/hea/data/splits.json"))[ds]
    rows = []
    for sp in splits:
        te = np.array([rid2i[r] for r in sp['test']]); mask = np.ones(len(rec), bool); mask[te] = False; tr = np.where(mask)[0]
        p = pred[pred.label == sp['label']].set_index('rid').loc[rec.rid.values[te]]
        err = (p.yhat - p.y).abs().values
        for name, blocks in SPACES.items():
            cols = feature_columns(X, blocks)
            A = X[cols].values.astype(float)
            if name.startswith('composition fractions'):
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=5, metric='manhattan').fit(A[tr]); d = nn.kneighbors(A[te])[0].mean(1)
            else:
                pr = Prep(clip=False).fit(A[tr]); Ztr, Zte = pr.transform(A[tr]), pr.transform(A[te])
                if 'winsorized' in name and 'unwinsorized' not in name:
                    Ztr, Zte = np.clip(Ztr, -5, 5), np.clip(Zte, -5, 5)
                if name.startswith('PCA'):
                    pca = PCA(10, random_state=0).fit(Ztr); Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)
                d = knn_distance(Ztr, Zte)
            rows.append(pd.DataFrame({'dataset': ds, 'protocol': sp['protocol'], 'label': sp['label'], 'space': name,
                                      'rid': rec.rid.values[te], 'd': d, 'abs_err': err}))
    df = pd.concat(rows)
    df.to_csv(f"{RES}/ad_variants_{ds}.csv", index=False)
    out = []
    for (proto, space), g in df.groupby(['protocol', 'space']):
        per = [spearmanr(x.d, x.abs_err).correlation for _, x in g.groupby('label') if len(x) > 10]
        out.append(dict(dataset=ds, protocol=proto, space=space, rho_pooled=spearmanr(g.d, g.abs_err).correlation,
                        rho_within_split_mean=np.nanmean(per)))
    # pooled over all non-random protocols (one record may appear several times; it is a pooled association)
    for space, g in df[df.protocol != 'random'].groupby('space'):
        out.append(dict(dataset=ds, protocol='all non-random', space=space,
                        rho_pooled=spearmanr(g.d, g.abs_err).correlation, rho_within_split_mean=np.nan))
    return pd.DataFrame(out)


if __name__ == '__main__':
    import sys
    res = pd.concat([run(ds) for ds in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['HV', 'YS'])])
    res.to_csv(f"{RES}/tables/ad_variants.csv", index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
    print(res.pivot_table(index=['dataset', 'space'], columns='protocol', values='rho_pooled').round(3))
    print(res.pivot_table(index=['dataset', 'space'], columns='protocol', values='rho_within_split_mean').round(3))

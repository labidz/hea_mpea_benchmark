"""SHAP interpretation of the XGB model and stability of feature importance across system-grouped folds."""
import json
import collections
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from common import load, feature_columns, DEFAULT_BLOCKS
from models import build

RES = "/home/claude/hea/results"


def modal_params(ds, model='XGB', protocol='system'):
    ps = [json.loads(l) for l in open(f"{RES}/{ds}_params.jsonl")]
    ps = [p['params'] for p in ps if p['model'] == model and p['label'].startswith(protocol)]
    c = collections.Counter(json.dumps(p, sort_keys=True) for p in ps)
    best, n = c.most_common(1)[0]
    return json.loads(best), n, len(ps)


def run(ds):
    rec, Xdf = load(ds)
    cols = feature_columns(Xdf, DEFAULT_BLOCKS)
    X = Xdf[cols].values; y = rec.target.values.astype(float)
    p, n, tot = modal_params(ds)
    print(ds, "modal XGB params", p, f"({n}/{tot} system folds)")
    m = build('XGB', p, seed=0).fit(X, y)
    Z = m.prep.transform(X)
    names = np.array(cols)[m.prep.keep]
    ex = shap.TreeExplainer(m.est)
    sv = ex.shap_values(Z)
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    imp.to_csv(f"{RES}/tables/shap_importance_{ds}.csv", header=['mean_abs_shap'])
    np.save(f"{RES}/shap_values_{ds}.npy", sv)
    pd.DataFrame(Z, columns=names).to_csv(f"{RES}/shap_X_{ds}.csv", index=False)
    print(imp.head(15).round(2).to_string())
    # stability across system-grouped training folds
    splits = json.load(open("/home/claude/hea/data/splits.json"))[ds]
    rid2i = {r: i for i, r in enumerate(rec.rid.values)}
    vecs, tops = [], collections.Counter()
    for sp in splits:
        if sp['protocol'] != 'system':
            continue
        te = np.array([rid2i[r] for r in sp['test']]); mask = np.ones(len(y), bool); mask[te] = False
        mf = build('XGB', p, seed=0).fit(X[mask], y[mask])
        nm = np.array(cols)[mf.prep.keep]
        s = np.abs(shap.TreeExplainer(mf.est).shap_values(mf.prep.transform(X[mask]))).mean(0)
        v = pd.Series(s, index=nm).reindex(imp.index).fillna(0.0)
        vecs.append(v.values); tops.update(v.sort_values(ascending=False).index[:10])
    V = np.array(vecs)
    rhos = [spearmanr(V[i], V[j]).correlation for i in range(len(V)) for j in range(i + 1, len(V))]
    rho_full = [spearmanr(V[i], imp.values).correlation for i in range(len(V))]
    stab = pd.DataFrame({'feature': imp.index[:20], 'mean_abs_shap_full': imp.values[:20],
                         'top10_frequency': [tops[f] / len(V) for f in imp.index[:20]],
                         'fold_mean': V.mean(0)[:20], 'fold_sd': V.std(0)[:20]})
    stab.to_csv(f"{RES}/tables/shap_stability_{ds}.csv", index=False)
    summ = dict(dataset=ds, n_folds=len(V), pairwise_spearman_mean=float(np.mean(rhos)),
                pairwise_spearman_min=float(np.min(rhos)), vs_full_mean=float(np.mean(rho_full)), params=p)
    json.dump(summ, open(f"{RES}/tables/shap_stability_{ds}.json", 'w'), indent=1)
    print(summ); print(stab.head(12).round(3).to_string())


if __name__ == '__main__':
    import sys
    for ds in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['HV', 'YS']):
        run(ds)

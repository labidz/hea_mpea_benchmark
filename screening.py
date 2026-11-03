"""Retrospective screening analysis: can models rank alloys they have never seen?
(i) within-system Spearman rho for held-out systems (system-grouped protocol, systems with >=5 test records)
(ii) top-quintile recall on the temporal and external test sets when candidates are ranked by the point
     prediction, by an optimistic upper bound (UCB) or by a conservative lower bound (LCB) of the
     protocol-matched 68% CV+ interval."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from analysis_v2 import load_all, RES, TAB   # revision 2: merged per-record tables


def within_system(ds, d):
    rec = pd.read_csv(f"/home/claude/hea/data/{'hv' if ds == 'HV' else 'ys'}_records.csv")
    key = rec.set_index('rid').sys_key
    g = d[d.protocol == 'system'].copy()
    g['sys'] = g.rid.map(key)
    if ds == 'YS':  # condition-matched ranking: compare alloys at room temperature in compression only
        cond = rec.set_index('rid')
        g = g[(g.rid.map(cond.test_T) == 25) & (g.rid.map(cond.test) == 'C')]
    out = []
    for (model, rep, sysk), s in g.groupby(['model', 'rep', 'sys']):
        if len(s) >= 5 and s.y.std() > 0:
            out.append(dict(dataset=ds, model=model, rep=rep, system=sysk, n=len(s),
                            rho=spearmanr(s.y, s.yhat).correlation))
    return pd.DataFrame(out)


def topk_recall(ds, d, labels=('temporal-2019', 'external-GE'), frac=0.2):
    out = []
    for lab in labels:
        for model, g in d[d.label == lab].groupby('model'):
            g = g.drop_duplicates('rid')
            k = max(1, int(round(frac * len(g))))
            true_top = set(g.nlargest(k, 'y').rid)
            scores = {'mean': g.yhat}
            if 'cvp_matched_lo68' in g and g.cvp_matched_lo68.notna().all():
                scores['UCB'] = g.cvp_matched_hi68
                scores['LCB'] = g.cvp_matched_lo68
            for sname, s in scores.items():
                sel = set(g.assign(s=s.values).nlargest(k, 's').rid)
                out.append(dict(dataset=ds, label=lab, model=model, score=sname, k=k, n=len(g),
                                recall=len(sel & true_top) / k, random_expectation=frac))
    return pd.DataFrame(out)


if __name__ == '__main__':
    W, T = [], []
    for ds in ('HV', 'YS'):
        try:
            d = load_all(ds)
        except FileNotFoundError:
            continue
        W.append(within_system(ds, d)); T.append(topk_recall(ds, d))
    W = pd.concat(W); T = pd.concat(T)
    W = W[W.model != 'Mean']; T = T[T.model != 'Mean']
    W.to_csv(f"{TAB}/within_system_rho.csv", index=False); T.to_csv(f"{TAB}/topk_recall.csv", index=False)
    print(W.groupby(['dataset', 'model']).rho.agg(['median', 'mean', 'count']).round(3))
    print(T.pivot_table(index=['dataset', 'label', 'model'], columns='score', values='recall').round(3))

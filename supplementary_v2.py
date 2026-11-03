"""Revision-2 supplementary analyses.
  noise    : experimental variability (pooled within-group SD) with group-bootstrap CIs and its decomposition
  temporal : quantified overlap between the <=2018 training literature and the 2019 test literature (P5)
  external : external-set results by subset (overlap excluded; Ru-containing; seen vs unseen alloy systems)
"""
import json
import re
import numpy as np
import pandas as pd

D = "/home/claude/hea/data"; R = "/home/claude/hea/results"; T = f"{R}/tables"
B = 2000


def _pooled(groups):
    """groups: list of arrays with >= 2 values -> pooled within-group SD."""
    num = sum((len(g) - 1) * np.var(g, ddof=1) for g in groups); den = sum(len(g) - 1 for g in groups)
    return np.sqrt(num / den)


def noise():
    out = {}
    hv = pd.read_csv(f"{D}/hv_records.csv"); ys = pd.read_csv(f"{D}/ys_records.csv")
    rng = np.random.default_rng(7)
    for name, df, keys in [('HV', hv, ['comp_key', 'proc']), ('YS', ys, ['comp_key', 'proc', 'test', 'test_T'])]:
        res = {}
        for tag, sub, k in [('same_processing', df, keys), ('same_processing_cross_publication',
                                                              df[df.groupby(keys).doi.transform('nunique') >= 2], keys),
                            ('same_processing_same_publication', df, keys + ['doi']),
                            ('same_composition_any_processing', df, ['comp_key'] + (['test', 'test_T'] if name == 'YS' else []))]:
            G = [g.target.values for _, g in sub.groupby(k) if len(g) >= 2]
            if not G:
                continue
            sds = np.array([np.std(g, ddof=1) for g in G])
            boots = [_pooled([G[i] for i in rng.integers(0, len(G), len(G))]) for _ in range(B)]
            res[tag] = dict(pooled_sd=_pooled(G), ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                            groups=len(G), records=int(sum(len(g) for g in G)), median_group_sd=float(np.median(sds)),
                            iqr_group_sd=[float(np.percentile(sds, 25)), float(np.percentile(sds, 75))])
        res['target_sd'] = float(df.target.std())
        out[name] = res
    json.dump(out, open(f"{T}/noise_floor_v2.json", 'w'), indent=1)
    print(json.dumps(out, indent=1))
    return out


def _els(k):
    return set(re.findall(r'([A-Z][a-z]?)[0-9.]', k))


def temporal():
    nn = pd.read_csv(f"{T}/nn_comp_distance.csv")
    rows = []
    for ds, f in (('HV', 'hv_records.csv'), ('YS', 'ys_records.csv')):
        df = pd.read_csv(f"{D}/{f}")
        tr, te = df[df.year <= 2018], df[df.year == 2019]
        seen_el = set().union(*[_els(k) for k in tr.comp_key])
        new_el = te.comp_key.map(lambda k: len(_els(k) - seen_el) > 0)
        pc_tr = tr.proc.value_counts(normalize=True); pc_te = te.proc.value_counts(normalize=True)
        tvd = 0.5 * sum(abs(pc_tr.get(c, 0) - pc_te.get(c, 0)) for c in set(pc_tr.index) | set(pc_te.index))
        d = nn[(nn.dataset == ds) & (nn.protocol == 'temporal')].dnn
        r = dict(dataset=ds, n_train=len(tr), n_test=len(te), test_publications=te.doi.nunique(),
                 test_pubs_also_in_train=int(len(set(te.doi) & set(tr.doi))),
                 frac_comp_seen=te.comp_key.isin(set(tr.comp_key)).mean(),
                 frac_comp_and_proc_seen=pd.Series(list(zip(te.comp_key, te.proc))).isin(set(zip(tr.comp_key, tr.proc))).mean(),
                 frac_system_seen=te.sys_key.isin(set(tr.sys_key)).mean(),
                 frac_new_element=new_el.mean(), new_elements=','.join(sorted(set().union(*[_els(k) for k in te.comp_key]) - seen_el)),
                 median_nn_at_pct=float(d.median()), frac_nn_le_2at=float((d <= 2).mean()),
                 proc_total_variation=tvd, mean_target_train=tr.target.mean(), mean_target_test=te.target.mean(),
                 frac_unknown_proc_test=(te.proc == 'UNKNOWN').mean(), frac_unknown_proc_train=(tr.proc == 'UNKNOWN').mean())
        if ds == 'YS':
            r.update(frac_tension_train=(tr.test == 'T').mean(), frac_tension_test=(te.test == 'T').mean(),
                     median_T_train=tr.test_T.median(), median_T_test=te.test_T.median())
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(f"{T}/temporal_overlap_v2.csv", index=False)
    print(out.T.to_string())
    return out


def external():
    from inference_v2 import system_weights
    acc = pd.read_csv(f"{R}/final/HV_acc.csv"); acc = acc[acc.protocol == 'external']
    it = pd.read_csv(f"{R}/final/HV_int.csv"); it = it[(it.protocol == 'external') & (it.model == 'XGB') & (it.level == 90)]
    hv = pd.read_csv(f"{D}/hv_records.csv"); ge = pd.read_csv(f"{D}/ext_ge_records.csv")
    meta = ge.set_index('rid')
    seen_comp = set(hv.comp_key); seen_sys = set(hv.sys_key); seen_el = set().union(*[_els(k) for k in hv.comp_key])
    meta['overlap_comp'] = meta.comp_key.isin(seen_comp)
    meta['seen_system'] = meta.sys_key.isin(seen_sys)
    meta['has_Ru'] = meta.comp_key.str.contains('Ru')
    meta['unseen_element'] = meta.comp_key.map(lambda k: len(_els(k) - seen_el) > 0)
    subsets = {'all': meta.index, 'overlap compositions excluded': meta.index[~meta.overlap_comp],
               'Ru-containing': meta.index[meta.has_Ru], 'Ru-free': meta.index[~meta.has_Ru],
               'previously represented systems': meta.index[meta.seen_system],
               'unseen systems': meta.index[~meta.seen_system],
               'unseen systems, Ru-free': meta.index[(~meta.seen_system) & (~meta.has_Ru)]}
    rows = []
    for sname, ids in subsets.items():
        ids = set(ids)
        for m, g in acc[acc.rid.isin(ids)].groupby('model'):
            systems = np.array(sorted(g.sys_key.unique()))
            W = system_weights(systems, B, 99)[:, [list(systems).index(s) for s in g.sys_key]]
            e2 = ((g.yhat - g.y) ** 2).values
            bs = np.sqrt((W @ e2) / W.sum(1))
            ss = ((g.y - g.y.mean()) ** 2).sum()
            from scipy.stats import spearmanr
            rows.append(dict(subset=sname, model=m, n=len(g), n_systems=len(systems), RMSE=np.sqrt(e2.mean()),
                             RMSE_lo=np.percentile(bs, 2.5), RMSE_hi=np.percentile(bs, 97.5),
                             R2=1 - e2.sum() / ss if ss > 0 else np.nan, rho=spearmanr(g.y, g.yhat).correlation,
                             bias=(g.yhat - g.y).mean()))
        for meth, h in it[it.rid.isin(ids)].groupby('method'):
            rows.append(dict(subset=sname, model='XGB', method=meth, n=len(h),
                             coverage90=((h.y >= h.lo) & (h.y <= h.hi)).mean(), mean_width90=(h.hi - h.lo).mean()))
    out = pd.DataFrame(rows)
    out.to_csv(f"{T}/external_subsets_v2.csv", index=False)
    print({k: len(v) for k, v in subsets.items()})
    print(out.round(3).to_string())
    return out


if __name__ == '__main__':
    import sys
    for job in sys.argv[1].split(','):
        globals()[job]()

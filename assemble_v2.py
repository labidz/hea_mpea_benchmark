"""Assemble the revision-2 per-record prediction and interval tables from the original runs (unchanged parts) and the
revision runs (results/v2), plus TabPFN per-record predictions transferred from Kaggle.

Accuracy table  results/final/{DS}_acc.csv : dataset, protocol, label, rep, rid, sys_key, model, y, yhat, ad
Interval table  results/final/{DS}_int.csv : dataset, protocol, label, rep, rid, sys_key, model, method, level, lo, hi, yhat, y, ad
Rules
  * P1-P3 (random/composition/system): original runs for kNN, SVR, GPR, MLP, RF, XGB, Mean (inner folds unchanged);
    Ridge from the revision (alpha now tuned on the protocol-matched inner folds).
  * P4-P6 (element/temporal/external): revision runs (deployment-specific inner folds) for every tuned model; Mean
    from the original runs (no tuning).
  * Intervals: XGBoost from the revision for all protocols (corrected normalized CV+ and group-weighted CV+); RF/GPR
    CV+ from the original runs for P1-P3 (unchanged method) and from the revision for P4-P6; GP-native everywhere.
"""
import os
import numpy as np
import pandas as pd
from io_utils import read_pred

R = "/home/claude/hea/results"; V2 = f"{R}/v2"; OUT = f"{R}/final"
os.makedirs(OUT, exist_ok=True)
SHIFT = ('element', 'temporal', 'external')
Z = {90: 1.6448536, 68: 0.9944579}


def rep_of(label):
    return int(label.split('-r')[1][0]) if '-r' in label and label.split('-')[0] in ('random', 'composition', 'system') else 0


def load_tabpfn(ds):
    p = f"{R}/remote/tabpfn_v2_records.csv"
    if not os.path.exists(p):
        return None
    t = pd.read_csv(p)
    return t[t.dataset == ds]


def main(ds):
    rec = pd.read_csv(f"/home/claude/hea/data/{'hv' if ds == 'HV' else 'ys'}_records.csv")
    sysmap = dict(zip(rec.rid, rec.sys_key))
    if ds == 'HV':
        ge = pd.read_csv("/home/claude/hea/data/ext_ge_records.csv"); sysmap.update(dict(zip(ge.rid, ge.sys_key)))
    v1 = read_pred(f"{R}/{ds}_pred.csv")
    if ds == 'HV':
        v1 = pd.concat([v1, read_pred(f"{R}/HV_external_pred.csv")], ignore_index=True)
    v1 = v1.drop_duplicates(['label', 'rid', 'model'], keep='last')
    from v2io import read_v2
    v2 = read_v2(f"{V2}/{ds}_pred.csv").drop_duplicates(['label', 'rid', 'model'], keep='last')
    # ------------------------------------------------------------ accuracy
    keep1 = v1[((~v1.protocol.isin(SHIFT)) & (v1.model != 'Ridge')) | (v1.model == 'Mean')]
    keep1 = keep1[~((keep1.model == 'XGB') & (~keep1.protocol.isin(SHIFT)) & keep1.label.isin(v2[v2.model == 'XGB'].label))] \
        if False else keep1
    keep2 = v2[(v2.protocol.isin(SHIFT)) | (v2.model == 'Ridge')]
    acc = pd.concat([keep1[['protocol', 'label', 'rid', 'model', 'y', 'yhat']].assign(src='v1'),
                     keep2[['protocol', 'label', 'rid', 'model', 'y', 'yhat']].assign(src='v2')], ignore_index=True)
    t = load_tabpfn(ds)
    if t is not None:
        acc = pd.concat([acc, t[['protocol', 'label', 'rid', 'y', 'yhat']].assign(model='TabPFN', src='kaggle')], ignore_index=True)
    acc['dataset'] = ds; acc['rep'] = acc.label.map(rep_of); acc['sys_key'] = acc.rid.map(sysmap)
    # XGB consistency: revision P1-P3 refits (same hyper-parameters) must reproduce the original predictions
    x2 = v2[(v2.model == 'XGB') & (~v2.protocol.isin(SHIFT))][['label', 'rid', 'yhat']]
    if len(x2):
        m = x2.merge(v1[v1.model == 'XGB'][['label', 'rid', 'yhat']], on=['label', 'rid'], suffixes=('_v2', '_v1'))
        print(ds, 'XGB P1-P3 refit max |diff| =', float(np.abs(m.yhat_v2 - m.yhat_v1).max()), 'n =', len(m))
    assert not acc.duplicated(['label', 'rid', 'model']).any()
    adv = v2[v2.model == 'XGB'][['label', 'rid', 'ad']].drop_duplicates(['label', 'rid'])
    acc = acc.merge(adv, on=['label', 'rid'], how='left')
    acc.to_csv(f"{OUT}/{ds}_acc.csv", index=False)
    # ------------------------------------------------------------ intervals
    rows = []

    def add(df, model, method, lo, hi):
        for lvl in (90, 68):
            if lo.format(lvl) not in df or df[lo.format(lvl)].isna().all():
                continue
            d_ = df[df[lo.format(lvl)].notna() & df[hi.format(lvl)].notna()]
            rows.append(pd.DataFrame({'protocol': d_.protocol.values, 'label': d_.label.values, 'rid': d_.rid.values,
                                      'model': model, 'method': method, 'level': lvl, 'y': d_.y.values,
                                      'yhat': d_.yhat.values, 'lo': d_[lo.format(lvl)].values,
                                      'hi': d_[hi.format(lvl)].values}))
    # XGB: revision everywhere
    x = v2[v2.model == 'XGB']
    for v, nm in (('naive', 'naive'), ('matched', 'matched'), ('msys', 'matched-system')):
        add(x, 'XGB', f'CV+ {nm}', f'{v}_cvp_lo{{}}', f'{v}_cvp_hi{{}}')
        add(x, 'XGB', f'nCV+ {nm}', f'{v}_ncvp_lo{{}}', f'{v}_ncvp_hi{{}}')
        add(x, 'XGB', f'gwCV+ {nm}', f'{v}_gw_lo{{}}', f'{v}_gw_hi{{}}')
    for model in ('RF', 'GPR'):
        a = v1[(v1.model == model) & (~v1.protocol.isin(SHIFT))]
        add(a, model, 'CV+ naive', 'cvp_naive_lo{}', 'cvp_naive_hi{}')
        add(a, model, 'CV+ matched', 'cvp_matched_lo{}', 'cvp_matched_hi{}')
        b = v2[(v2.model == model) & (v2.protocol.isin(SHIFT))]
        for v, nm in (('naive', 'naive'), ('matched', 'matched'), ('msys', 'matched-system')):
            add(b, model, f'CV+ {nm}', f'{v}_cvp_lo{{}}', f'{v}_cvp_hi{{}}')
    g = acc[acc.model == 'GPR'][['label', 'rid']].merge(pd.concat([v1[(v1.model == 'GPR') & (~v1.protocol.isin(SHIFT))],
                                                                     v2[(v2.model == 'GPR') & (v2.protocol.isin(SHIFT))]]),
                                                          on=['label', 'rid'])
    g = g[g.gp_sd.notna()]
    for lvl in (90, 68):
        rows.append(pd.DataFrame({'protocol': g.protocol.values, 'label': g.label.values, 'rid': g.rid.values, 'model': 'GPR',
                                  'method': 'GP-native', 'level': lvl, 'y': g.y.values, 'yhat': g.yhat.values,
                                  'lo': (g.yhat - Z[lvl] * g.gp_sd).values, 'hi': (g.yhat + Z[lvl] * g.gp_sd).values}))
    it = pd.concat(rows, ignore_index=True)
    it['dataset'] = ds; it['rep'] = it.label.map(rep_of); it['sys_key'] = it.rid.map(sysmap)
    it = it.merge(adv, on=['label', 'rid'], how='left')
    it.to_csv(f"{OUT}/{ds}_int.csv", index=False)
    print(ds, 'acc', acc.groupby(['protocol', 'model']).size().unstack().fillna(0).astype(int).to_string())
    print(ds, 'int', it.groupby(['model', 'method']).protocol.unique().apply(lambda z: ','.join(sorted(z))).to_string())


if __name__ == '__main__':
    import sys
    for ds in sys.argv[1].split(','):
        main(ds)

"""Near-duplicate mechanism: composition distance from each test record to its nearest training record, and
random-split errors for test records whose composition is / is not present in the training fold."""
import json
import re
import numpy as np
import pandas as pd
from io_utils import read_pred

D = "/home/claude/hea/data"
OUT = "/home/claude/hea/results/tables"


def parse(key):
    return {m.group(1): float(m.group(2)) for m in re.finditer(r'([A-Z][a-z]?)([0-9.]+)', key)}


def comp_matrix(keys, elements):
    M = np.zeros((len(keys), len(elements)))
    for i, k in enumerate(keys):
        for el, f in parse(k).items():
            M[i, elements.index(el)] = f
    return M / M.sum(1, keepdims=True)


def main():
    rec = {'HV': pd.read_csv(f"{D}/hv_records.csv"), 'YS': pd.read_csv(f"{D}/ys_records.csv")}
    ge = pd.read_csv(f"{D}/ext_ge_records.csv")
    sp = json.load(open(f"{D}/splits.json"))
    els = sorted({e for df in list(rec.values()) + [ge] for k in df.comp_key for e in parse(k)})
    rows = []
    for ds, df in rec.items():
        C = comp_matrix(df.comp_key.tolist(), els)
        idx = {r: i for i, r in enumerate(df.rid)}
        for s in sp[ds]:
            te = np.array([idx[r] for r in s['test']])
            tr = np.setdiff1d(np.arange(len(df)), te)
            d = 0.5 * np.abs(C[te][:, None, :] - C[tr][None, :, :]).sum(-1).min(1) * 100   # at.% to change
            rows += [dict(dataset=ds, protocol=s['protocol'], label=s['label'], rid=df.rid.iloc[i], dnn=v)
                     for i, v in zip(te, d)]
        if ds == 'HV':
            Ce = comp_matrix(ge.comp_key.tolist(), els)
            d = 0.5 * np.abs(Ce[:, None, :] - C[None, :, :]).sum(-1).min(1) * 100
            rows += [dict(dataset='HV', protocol='external', label='external', rid=r, dnn=v) for r, v in zip(ge.rid, d)]
    nn = pd.DataFrame(rows)
    nn.to_csv(f"{OUT}/nn_comp_distance.csv", index=False)
    summ = nn.groupby(['dataset', 'protocol']).dnn.agg(median='median', frac_zero=lambda v: (v < 0.05).mean(),
                                                       frac_le2=lambda v: (v <= 2).mean()).reset_index()
    print(summ.to_string())
    summ.to_csv(f"{OUT}/nn_comp_distance_summary.csv", index=False)

    # random-split errors split by whether the test composition occurs in the training fold
    out = []
    for ds in ('HV', 'YS'):
        P = read_pred(f"/home/claude/hea/results/{ds}_pred.csv")
        P = P[(P.protocol == 'random')].merge(nn[(nn.dataset == ds) & (nn.protocol == 'random')][['label', 'rid', 'dnn']],
                                               on=['label', 'rid'])
        for m in ('XGB', 'GPR', 'RF', 'kNN'):
            g = P[P.model == m]
            if g.empty:
                continue
            for seen, h in g.groupby(g.dnn < 0.05):
                r = h.groupby('rep').apply(lambda x: np.sqrt(((x.y - x.yhat) ** 2).mean()), include_groups=False)
                out.append(dict(dataset=ds, model=m, seen=bool(seen), n=len(h) / h.rep.nunique(), RMSE=r.mean()))
    o = pd.DataFrame(out)
    print(o.to_string())
    o.to_csv(f"{OUT}/random_seen_unseen.csv", index=False)


if __name__ == '__main__':
    main()

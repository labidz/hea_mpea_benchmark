"""Ingest the revision-2 TabPFN Kaggle output (copied from the notebook log into results/remote/tabpfn_v2_log.txt):
  C| lines : CV+ coverage/width with deployment-specific calibration (P4-P6)
  R| lines : per-split RMSE (checked against the first TabPFN run and against the decoded records)
  base64   : gzip of 'DS|label|yhat;yhat;...' lines in the frozen test order of each split
Writes results/remote/tabpfn_v2_records.csv and results/remote/tabpfn_v2_cvplus.csv."""
import base64
import gzip
import hashlib
import json
import numpy as np
import pandas as pd

R = "/home/claude/hea/results"


def main(path=f"{R}/remote/tabpfn_v2_log.txt"):
    txt = open(path).read()
    lines = txt.splitlines()
    C = [l.split('|') for l in lines if l.startswith('C|')]
    Rl = [l.split('|') for l in lines if l.startswith('R|')]
    meta = next(l for l in lines if l.startswith('B64_LEN'))
    b64_md5 = meta.split('B64_MD5')[1].strip(); raw_md5 = meta.split('RAW_MD5')[1].split()[0]
    i0 = lines.index('=====B64_BEGIN=====') + 1; i1 = lines.index('=====B64_END=====')
    b64 = ''.join(l.strip() for l in lines[i0:i1])
    assert hashlib.md5(b64.encode()).hexdigest() == b64_md5, 'base64 checksum mismatch'
    raw = gzip.decompress(base64.b64decode(b64))
    assert hashlib.md5(raw).hexdigest() == raw_md5, 'raw checksum mismatch'
    splits = json.load(open("/home/claude/hea/data/splits.json"))
    ge = pd.read_csv("/home/claude/hea/data/ext_ge_records.csv")
    rec = {'HV': pd.read_csv("/home/claude/hea/data/hv_records.csv"), 'YS': pd.read_csv("/home/claude/hea/data/ys_records.csv")}
    order = {(ds, s['label']): (s['protocol'], s['test']) for ds in splits for s in splits[ds]}
    order[('HV', 'external-GE')] = ('external', list(ge.rid))
    yv = {**{r: t for ds in rec for r, t in zip(rec[ds].rid, rec[ds].target)}, **dict(zip(ge.rid, ge.target))}
    rows = []
    for l in raw.decode().strip().splitlines():
        ds, lab, vals = l.split('|')
        proto, rids = order[(ds, lab)]
        v = np.array([float(x) for x in vals.split(';')])
        assert len(v) == len(rids), (ds, lab)
        rows.append(pd.DataFrame({'dataset': ds, 'protocol': proto, 'label': lab, 'rid': rids, 'yhat': v,
                                  'y': [yv[r] for r in rids]}))
    out = pd.concat(rows, ignore_index=True)
    # consistency: RMSE from decoded records vs R| lines (values rounded to 0.1 before transfer)
    chk = out.groupby(['dataset', 'label']).apply(lambda g: np.sqrt(((g.yhat - g.y) ** 2).mean()), include_groups=False)
    rr = {(a[1], a[2]): float(a[4]) for a in Rl}
    dev = max(abs(chk[k] - rr[k]) for k in rr)
    print('records', len(out), 'splits', len(rr), 'max |RMSE(decoded) - RMSE(Kaggle)| =', round(dev, 4))
    assert dev < 0.1
    # reproducibility vs the first TabPFN run (per-fold metrics)
    old = pd.read_csv(f"{R}/ext_metrics.csv")
    old = old[old.model == 'TabPFN'] if 'model' in old else old
    m = pd.DataFrame([dict(dataset=k[0], label=k[1], RMSE_v2=v) for k, v in rr.items()]).merge(old[['dataset', 'label', 'RMSE']], on=['dataset', 'label'])
    print('first-run comparison: n =', len(m), 'max |dRMSE| =', round(float((m.RMSE_v2 - m.RMSE).abs().max()), 3))
    out.to_csv(f"{R}/remote/tabpfn_v2_records.csv", index=False)
    cv = pd.DataFrame([dict(dataset=a[1], label=a[2], variant={'m': 'matched', 's': 'matched-system'}[a[3]], level=int(a[4]),
                            n=int(a[5]), n_cal=int(a[6]), coverage=float(a[7]), mean_width=float(a[8]), folds=a[9]) for a in C])
    cv.to_csv(f"{R}/remote/tabpfn_v2_cvplus.csv", index=False)
    print(cv.groupby(['dataset', 'variant', 'level']).apply(lambda g: pd.Series({'coverage': np.average(g.coverage, weights=g.n),
                                                                                 'width': np.average(g.mean_width, weights=g.n)}), include_groups=False))
    m.to_csv(f"{R}/remote/tabpfn_v2_repro.csv", index=False)


if __name__ == '__main__':
    main()

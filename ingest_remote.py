"""Parse result blocks copied from Kaggle logs (results/remote/*.txt) into tidy CSVs.
- tabpfn.txt  : M| lines (dataset|label|protocol|...) and P| lines (per-record temporal/external predictions)
- ys_gpr.txt, ys_rfmlp.txt : M| (model|label|...), W| (within-system rho) and K| (top-quintile recall) lines
Integrity checks: per-fold n must equal the frozen split sizes; TabPFN temporal/external metrics recomputed from
P| lines must match the M| lines."""
import glob
import json
import numpy as np
import pandas as pd

RES = "/home/claude/hea/results"


def parse():
    splits = json.load(open("/home/claude/hea/data/splits.json"))
    size = {(ds, s['label']): len(s['test']) for ds in splits for s in splits[ds]}
    size[('HV', 'external-GE')] = len(pd.read_csv("/home/claude/hea/data/ext_ge_records.csv"))
    M, W, K, P = [], [], [], []
    for f in sorted(glob.glob(f"{RES}/remote/*.txt")):
        lines = [l.strip() for l in open(f) if l.strip()]
        is_tab = 'tabpfn' in f
        hdr = None
        for l in lines:
            if l.startswith('M|'):
                parts = l.split('|')[1:]
                if parts[0] in ('dataset', 'model'):
                    hdr = parts; continue
                row = dict(zip(hdr, parts))
                if is_tab:
                    row['model'] = 'TabPFN'
                else:
                    row['dataset'] = 'YS'
                M.append(row)
            elif l.startswith('W|'):
                m, rep, sk, n, rho = l.split('|')[1:]
                W.append(dict(dataset='YS', model=m, rep=int(rep), system=sk, n=int(n), rho=float(rho)))
            elif l.startswith('K|'):
                p = l.split('|')[1:]
                if len(p) == 7:   # K|model|dataset|label|score|k|n|recall (TabPFN, computed in-browser)
                    m, ds_, lab, score, k, n, rec = p
                else:             # K|model|label|score|k|n|recall (YS CPU runs)
                    m, lab, score, k, n, rec = p; ds_ = 'YS'
                K.append(dict(dataset=ds_, label=lab, model=m, score=score, k=int(k), n=int(n), recall=float(rec), random_expectation=0.2))
            elif l.startswith('P|') and not l.startswith('P|dataset'):
                ds, lab, rid, yhat, q05, q95, lo, hi = l.split('|')[1:]
                P.append(dict(dataset=ds, label=lab, rid=rid, yhat=float(yhat), q05=float(q05), q95=float(q95),
                              cvp_matched_lo90=float(lo), cvp_matched_hi90=float(hi)))
    M = pd.DataFrame(M)
    for c in M.columns:
        if c not in ('dataset', 'label', 'protocol', 'model'):
            M[c] = pd.to_numeric(M[c].replace('nan', np.nan))
    bad = [(r.dataset, r.model, r.label, r.n, size.get((r.dataset, r.label))) for r in M.itertuples() if size.get((r.dataset, r.label)) != r.n]
    assert not bad, f"fold size mismatch: {bad[:5]}"
    M.to_csv(f"{RES}/ext_metrics.csv", index=False)
    pd.DataFrame(W).to_csv(f"{RES}/remote_within.csv", index=False)
    pd.DataFrame(K).to_csv(f"{RES}/remote_topk.csv", index=False)
    if P:
        P = pd.DataFrame(P)
        for ds in ('HV', 'YS'):
            rec = pd.read_csv(f"/home/claude/hea/data/{'hv' if ds == 'HV' else 'ys'}_records.csv").set_index('rid').target
            ge = pd.read_csv("/home/claude/hea/data/ext_ge_records.csv").set_index('rid').target
            p = P[P.dataset == ds].copy()
            if p.empty:
                continue
            p['y'] = [ge[r] if r.startswith('GE') else rec[r] for r in p.rid]
            p['protocol'] = p.label.map(lambda s: s.split('-')[0])
            p['model'] = 'TabPFN'
            # integrity: recompute RMSE vs M lines
            for lab, g in p.groupby('label'):
                rm = np.sqrt(((g.yhat - g.y) ** 2).mean())
                ref = M[(M.dataset == ds) & (M.label == lab) & (M.model == 'TabPFN')].RMSE.iloc[0]
                assert abs(rm - ref) < 0.5, (ds, lab, rm, ref)
                print(f"check {ds} {lab}: RMSE from records {rm:.2f} vs metrics {ref:.2f}")
            p.to_csv(f"{RES}/tabpfn_records_{ds}.csv", index=False)
    print(M.groupby(['dataset', 'model']).size())


if __name__ == '__main__':
    parse()

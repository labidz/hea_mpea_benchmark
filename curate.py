"""Curate the MPEA dataset (Borg et al., Sci. Data 2020) into two ML-ready tables.

Outputs (in data/):
  hv_records.csv  - room-temperature Vickers hardness records
  ys_records.csv  - yield strength records with test temperature and test type
Each record carries: normalized composition, composition key, element-set (system) key,
processing route, publication year, DOI and the observed phase label (kept only for the
'post-synthesis information' ablation, never used as a default input).
"""
import re
import sys
import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/data_raw/mpea_dataset/MPEA_dataset.csv"
OUT = "/home/claude/hea/data"

COLS = ['ref', 'formula', 'micro', 'proc', 'phase3', 'grain', 'expdens', 'caldens', 'HV', 'test', 'T',
        'YS', 'UTS', 'El', 'Elp', 'E_exp', 'E_cal', 'O', 'N', 'C', 'doi', 'year', 'title']


def parse_formula(f):
    toks = re.findall(r'([A-Z][a-z]?)\s*([0-9.]*)', f)
    comp = {}
    for el, amt in toks:
        comp[el] = comp.get(el, 0.0) + (float(amt) if amt else 1.0)
    tot = sum(comp.values())
    return {k: v / tot for k, v in comp.items()}


def comp_key(c, nd=3):
    return " ".join(f"{k}{round(v, nd):.3f}" for k, v in sorted(c.items()) if round(v, nd) > 0)


def sys_key(c):
    return "-".join(sorted(k for k, v in c.items() if v > 0))


def base_table(df):
    df = df.copy()
    df['comp'] = df.formula.apply(parse_formula)
    df['comp_key'] = df.comp.apply(comp_key)
    df['sys_key'] = df.comp.apply(sys_key)
    df['n_elem'] = df.comp.apply(len)
    df['proc'] = df.proc.fillna('UNKNOWN')
    df['phase3'] = df.phase3.fillna('other')
    df['micro'] = df.micro.fillna('UNKNOWN')
    df['year'] = df.year.astype(int)
    return df


def main():
    raw = pd.read_csv(SRC)
    raw.columns = COLS
    df = base_table(raw)

    # ---- Hardness: room temperature only (T = 25 C or not reported) ----
    hv = df[df.HV.notna() & (df['T'].isna() | (df['T'] == 25))].copy()
    hv = hv[['formula', 'comp_key', 'sys_key', 'n_elem', 'proc', 'phase3', 'micro', 'HV', 'year', 'doi']]
    hv = hv.rename(columns={'HV': 'target'})
    hv.insert(0, 'rid', [f"HV{i:04d}" for i in range(len(hv))])

    # ---- Yield strength: requires test temperature and test type ----
    ys = df[df.YS.notna() & df['T'].notna() & df.test.isin(['T', 'C'])].copy()
    ys = ys[['formula', 'comp_key', 'sys_key', 'n_elem', 'proc', 'phase3', 'micro', 'T', 'test', 'YS', 'year', 'doi']]
    ys = ys.rename(columns={'YS': 'target', 'T': 'test_T'})
    ys.insert(0, 'rid', [f"YS{i:04d}" for i in range(len(ys))])

    import os
    os.makedirs(OUT, exist_ok=True)
    hv.to_csv(f"{OUT}/hv_records.csv", index=False)
    ys.to_csv(f"{OUT}/ys_records.csv", index=False)

    for name, d in [('HV', hv), ('YS', ys)]:
        dup = d.groupby('comp_key').size()
        print(f"{name}: records={len(d)} unique_comp={d.comp_key.nunique()} systems={d.sys_key.nunique()} "
              f"refs={d.doi.nunique()} comps_with_repeats={int((dup > 1).sum())} "
              f"records_in_repeated_comps={int(dup[dup > 1].sum())} years={d.year.min()}-{d.year.max()} "
              f"n2019={int((d.year == 2019).sum())}")
        print("  target:", d.target.describe().round(1).to_dict())
        print("  proc:", d.proc.value_counts().to_dict())
        els = pd.Series([e for s in d.sys_key for e in s.split('-')]).value_counts()
        print("  elements:", els.to_dict())


if __name__ == '__main__':
    main()


def curate_external_ge(path="/home/claude/hea/data/external/GE_RefractoryAlloyScreeningDataset_FINAL.csv"):
    """Detor et al., Data in Brief 45 (2022) 108582; figshare 10.6084/m9.figshare.17032760 (CC BY 4.0).
    Arc-melted, as-cast refractory alloys; Vickers microhardness (500 gf) reported in GPa."""
    g = pd.read_csv(path, encoding='utf-8-sig')
    els = [c.split('(')[0] for c in g.columns[:9]]
    recs = []
    for i, row in g.iterrows():
        comp = {el: float(row.iloc[j]) for j, el in enumerate(els) if float(row.iloc[j]) > 0}
        tot = sum(comp.values()); comp = {k: v / tot for k, v in comp.items()}
        recs.append({'rid': f"GE{i:03d}", 'formula': " ".join(f"{k}{v:.4f}" for k, v in sorted(comp.items())),
                     'comp_key': comp_key(comp), 'sys_key': sys_key(comp), 'n_elem': len(comp), 'proc': 'CAST',
                     'phase3': 'other', 'micro': 'UNKNOWN',
                     'target': float(row['Hardness (GPa)']) * 1000.0 / 9.80665, 'year': 2022,
                     'doi': '10.1016/j.dib.2022.108582'})
    ext = pd.DataFrame(recs)
    ext.to_csv(f"{OUT}/ext_ge_records.csv", index=False)
    hv = pd.read_csv(f"{OUT}/hv_records.csv")
    print(f"GE external: records={len(ext)} unique_comp={ext.comp_key.nunique()} systems={ext.sys_key.nunique()} "
          f"HV range={ext.target.min():.0f}-{ext.target.max():.0f}; comps also in MPEA-HV: "
          f"{len(set(ext.comp_key) & set(hv.comp_key))}; systems also in MPEA-HV: {len(set(ext.sys_key) & set(hv.sys_key))}; "
          f"records with Ru: {ext.sys_key.str.contains('Ru').sum()}, with Re: {ext.sys_key.str.contains('Re').sum()}")


if __name__ == '__main__':
    curate_external_ge()

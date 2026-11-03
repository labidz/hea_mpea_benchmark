"""Composition featurization for MPEA records.

Feature blocks
  FRAC : atomic fractions of every element present in the union of all datasets
  PHYS : alloy-theory descriptors (Wen et al. 2019 set via matminer WenAlloys:
         VEC, atomic-size mismatch, mixing enthalpy/entropy, Omega, electronegativity
         mismatch, shear-modulus descriptors, cohesive energy, ...)
  MAGPIE : weighted statistics of 22 elemental properties (Ward et al. 2016)
Context blocks (not composition)
  PROC : one-hot processing route (CAST, ANNEAL, WROUGHT, POWDER, OTHER, UNKNOWN)
  TEST : test temperature (C) and test type (tension=1/compression=0) -- yield strength only
  PHASE: observed phase label one-hot (FCC/BCC/other) -- post-synthesis, ablation only
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition.alloy import WenAlloys

warnings.filterwarnings("ignore")
DATA = "/home/claude/hea/data"
PROC_LEVELS = ['CAST', 'ANNEAL', 'WROUGHT', 'POWDER', 'OTHER', 'UNKNOWN']


def comp_from_key(k):
    # comp_key like "Al0.111 Co0.222 ..."
    import re
    d = {el: float(x) for el, x in re.findall(r'([A-Z][a-z]?)([0-9.]+)', k)}
    return Composition(d)


def featurize_unique(keys):
    wen = WenAlloys()
    mag = ElementProperty.from_preset("magpie", impute_nan=True)
    rows = []
    for k in keys:
        c = comp_from_key(k)
        r = {'comp_key': k}
        try:
            v = wen.featurize(c)
            r.update({f"PHYS:{n}": x for n, x in zip(wen.feature_labels(), v)})
        except Exception as e:  # elements outside WenAlloys tables
            r.update({f"PHYS:{n}": np.nan for n in wen.feature_labels()})
            r['PHYS_FAIL'] = repr(e)[:80]
        v = mag.featurize(c)
        r.update({f"MAG:{n.replace('MagpieData ', '')}": x for n, x in zip(mag.feature_labels(), v)})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    hv = pd.read_csv(f"{DATA}/hv_records.csv")
    ys = pd.read_csv(f"{DATA}/ys_records.csv")
    extra = []
    try:
        ext = pd.read_csv(f"{DATA}/ext_ge_records.csv")
        extra = list(ext.comp_key)
    except FileNotFoundError:
        pass
    keys = sorted(set(hv.comp_key) | set(ys.comp_key) | set(extra))
    F = featurize_unique(keys)
    # element fraction block over the union of elements
    import re
    elems = sorted({el for k in keys for el in re.findall(r'([A-Z][a-z]?)', k)})
    for el in elems:
        F[f"FRAC:{el}"] = [dict((e, float(x)) for e, x in re.findall(r'([A-Z][a-z]?)([0-9.]+)', k)).get(el, 0.0)
                           for k in F.comp_key]
    F.to_csv(f"{DATA}/composition_features.csv", index=False)
    phys = [c for c in F.columns if c.startswith('PHYS:')]
    print("unique compositions:", len(F), "elements:", len(elems))
    print("PHYS features:", len(phys), "MAG:", sum(c.startswith('MAG:') for c in F.columns))
    if 'PHYS_FAIL' in F:
        bad = F[F.PHYS_FAIL.notna()]
        print("PHYS failures:", len(bad)); print(bad[['comp_key', 'PHYS_FAIL']].head(20).to_string())
    print(F[phys].isna().sum().to_string())


if __name__ == '__main__':
    main()

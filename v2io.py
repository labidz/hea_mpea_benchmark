"""Reader for results/v2/*_pred.csv. Rows were appended with model-dependent column sets under a 10-column header;
the layout of every row is fully determined by its model and length:
  dataset, protocol, label, rid, model, y, ad, n_train, yhat, [gp_sd if GPR], 12 interval columns per calibration
  variant (naive, matched, msys in that order; for each level 90, 68 and each kind cvp, ncvp, gw: lo, hi), sec."""
import csv
import pandas as pd

BASE = ['dataset', 'protocol', 'label', 'rid', 'model', 'y', 'ad', 'n_train', 'yhat']
VARIANTS = ['naive', 'matched', 'msys']


def interval_cols(variant):
    return [f'{variant}_{kind}_{lh}{tag}' for tag in (90, 68) for kind in ('cvp', 'ncvp', 'gw') for lh in ('lo', 'hi')]


def read_v2(path):
    rows = list(csv.reader(open(path)))
    assert rows[0][:9] == BASE
    recs = []
    for r in rows[1:]:
        if r[:9] == BASE:          # a repeated header line
            continue
        gp = r[4] == 'GPR'
        nvar = (len(r) - 10 - int(gp)) // 12
        assert len(r) == 10 + int(gp) + 12 * nvar, (len(r), r[:5])
        cols = BASE + (['gp_sd'] if gp else []) + [c for v in VARIANTS[:nvar] for c in interval_cols(v)] + ['sec']
        recs.append(dict(zip(cols, r)))
    df = pd.DataFrame(recs)
    for c in df.columns:
        if c not in ('dataset', 'protocol', 'label', 'rid', 'model'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def done_pairs(path):
    out = set()
    for r in list(csv.reader(open(path)))[1:]:
        if len(r) > 4 and r[:9] != BASE:
            out.add((r[2], r[4]))
    return out


def filter_rows(path, keep):
    """Rewrite the raw file keeping rows for which keep(row_list) is True (format preserved)."""
    rows = list(csv.reader(open(path)))
    kept = [rows[0]] + [r for r in rows[1:] if keep(r)]
    with open(path, 'w', newline='') as fh:
        csv.writer(fh).writerows(kept)
    return len(rows) - len(kept)

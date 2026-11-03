"""Robust reader for prediction files. Early HV rows were appended with model-dependent column sets
(no common header); their column order is deterministic, so rows are re-mapped field by field."""
import csv
import pandas as pd

BASE = ['dataset', 'protocol', 'label', 'rep', 'fold', 'rid', 'model', 'y', 'ad_knn', 'n_train']
UQ_MODELS = ('XGB', 'RF', 'GPR')


def legacy_columns(model, protocol):
    cols = BASE + (['yhat', 'gp_sd'] if model == 'GPR' else ['yhat'])
    if model in UQ_MODELS:
        variants = ['naive'] + (['matched'] if protocol != 'random' else [])
        for v in variants:
            for t in (90, 68):
                cols += [f'cvp_{v}_lo{t}', f'cvp_{v}_hi{t}', f'ncp_{v}_hw{t}']
    return cols + ['sec']


def read_pred(path):
    with open(path) as fh:
        rd = csv.reader(fh)
        header = next(rd)
        fixed = 'gp_sd' in header and 'ncp_matched_hw68' in header  # new-format file
        rows = []
        for r in rd:
            if fixed:
                rows.append(dict(zip(header, r)))
            else:
                cols = legacy_columns(r[6], r[1])
                assert len(cols) == len(r), (len(cols), len(r), r[:8])
                rows.append(dict(zip(cols, r)))
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ('dataset', 'protocol', 'label', 'rid', 'model'):
            df[c] = pd.to_numeric(df[c].replace('', None), errors='coerce')
    return df

"""Collect every number quoted in the manuscript from data/results files (single source of truth)."""
import hashlib
import json
import numpy as np
import pandas as pd

H = "/home/claude/hea"


def data_stats():
    R = {}
    raw = pd.read_csv(f"{H}/../data_raw/mpea_dataset/MPEA_dataset.csv") if False else pd.read_csv("/home/claude/data_raw/mpea_dataset/MPEA_dataset.csv")
    R['mpea_records'] = len(raw)
    R['mpea_commit'] = 'd31e2510'
    hvraw = raw[raw['PROPERTY: HV'].notna()]
    t = hvraw[[c for c in raw.columns if 'Test temperature' in c][0]]
    R['HV_excl_T'] = int(((t.notna()) & (t != 25)).sum())
    R['HV_excl_Tmin'], R['HV_excl_Tmax'] = float(t[(t.notna()) & (t != 25)].min()), float(t[(t.notna()) & (t != 25)].max())
    for ds, fn in (('HV', 'hv_records.csv'), ('YS', 'ys_records.csv'), ('GE', 'ext_ge_records.csv')):
        d = pd.read_csv(f"{H}/data/{fn}")
        R[f'{ds}_n'] = len(d); R[f'{ds}_comp'] = d.comp_key.nunique(); R[f'{ds}_sys'] = d.sys_key.nunique()
        R[f'{ds}_refs'] = d.doi.nunique(); R[f'{ds}_min'] = d.target.min(); R[f'{ds}_max'] = d.target.max()
        R[f'{ds}_sd'] = d.target.std(); R[f'{ds}_mean'] = d.target.mean()
        rep = d.groupby('comp_key').size()
        R[f'{ds}_rep_comp'] = int((rep > 1).sum()); R[f'{ds}_rep_rec'] = int(rep[rep > 1].sum())
        if ds == 'YS':
            R['YS_Tmin'], R['YS_Tmax'] = d.test_T.min(), d.test_T.max()
            R['YS_tension'] = int((d.test == 'T').sum()); R['YS_compression'] = int((d.test == 'C').sum())
        if ds != 'GE':
            R[f'{ds}_2019'] = int((d.year == 2019).sum())
    hv = pd.read_csv(f"{H}/data/hv_records.csv"); ge = pd.read_csv(f"{H}/data/ext_ge_records.csv")
    R['GE_comp_overlap'] = len(set(ge.comp_key) & set(hv.comp_key)); R['GE_sys_overlap'] = len(set(ge.sys_key) & set(hv.sys_key))
    R['GE_Ru'] = int(ge.sys_key.str.contains('Ru').sum()); R['GE_Re'] = int(ge.sys_key.str.contains('Re').sum())
    R['ge_md5'] = hashlib.md5(open(f"{H}/data/external/GE_RefractoryAlloyScreeningDataset_FINAL.csv", 'rb').read()).hexdigest()
    R['HV_Re_train'] = int(hv.sys_key.str.split('-').apply(lambda s: 'Re' in s).sum())
    sp = json.load(open(f"{H}/data/splits.json"))
    for ds in ('HV', 'YS'):
        d = pd.read_csv(f"{H}/data/{'hv' if ds == 'HV' else 'ys'}_records.csv").set_index('rid')
        R[f'{ds}_loeo'] = sum(s['protocol'] == 'element' for s in sp[ds])
        R[f'{ds}_loeo_elems'] = [s['label'].split('-')[1] for s in sp[ds] if s['protocol'] == 'element']
        R[f'{ds}_temporal_test'] = len([s for s in sp[ds] if s['protocol'] == 'temporal'][0]['test'])
        ov = []
        for s in sp[ds]:
            if s['protocol'] != 'random':
                continue
            te = set(s['test']); tr_comp = set(d.loc[[r for r in d.index if r not in te]].comp_key)
            ov.append(np.mean([d.loc[r].comp_key in tr_comp for r in s['test']]))
        R[f'{ds}_random_overlap'] = float(np.mean(ov))
    for ds, fn in (('HV', 'hv_records.csv'), ('YS', 'ys_records.csv')):
        d = pd.read_csv(f"{H}/data/{fn}"); tr = d[d.year <= 2018]; te = d[d.year == 2019]
        R[f'temporal_{ds}_comp_seen'] = float(te.comp_key.isin(tr.comp_key).mean())
        R[f'temporal_{ds}_sys_seen'] = float(te.sys_key.isin(tr.sys_key).mean())
        R[f'temporal_{ds}_ytr'], R[f'temporal_{ds}_yte'] = tr.target.mean(), te.target.mean()
        for p in ('POWDER', 'WROUGHT'):
            R[f'temporal_{ds}_{p}_tr'], R[f'temporal_{ds}_{p}_te'] = float((tr.proc == p).mean()), float((te.proc == p).mean())
    from common_shim import nfeat
    R['HV_nfeat'], R['YS_nfeat'] = nfeat('HV'), nfeat('YS')
    R['noise'] = json.load(open(f"{H}/results/tables/noise_floor.json"))
    return R

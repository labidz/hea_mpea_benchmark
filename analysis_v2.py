"""Revision-2 derived tables (results/tables_v2) from the merged per-record tables in results/final.
Keeps the column conventions of the original analysis so that the text, figures and SI builders can read them.
  load_all(ds)        wide per-record frame (accuracy + interval columns) for every model and protocol
  perf_summary.csv    cluster-bootstrap accuracy (from inference_v2)
  perf_folds.csv      per-fold / per-element metrics; element_tasks.csv
  pairwise.csv        paired cluster-bootstrap RMSE differences to the most accurate model
  ranking.csv         Kendall tau vs the random-split ranking with cluster-bootstrap CI
  uq_summary.csv      coverage (with cluster-bootstrap CI), width, Winkler, per-system coverage; TabPFN from Kaggle
  uq_by_ad.csv        coverage within applicability-domain tertiles (XGBoost)
  ad_corr.csv         median AD distance and Spearman rho(AD, |error|) per protocol (XGBoost)
"""
import os
import shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES = "/home/claude/hea/results"; FIN = f"{RES}/final"; T1 = f"{RES}/tables"; TAB = f"{RES}/tables_v2"
os.makedirs(TAB, exist_ok=True)
METHOD_COL = {'CV+ naive': 'cvp_naive', 'CV+ matched': 'cvp_matched', 'nCV+ naive': 'ncvp_naive', 'nCV+ matched': 'ncvp_matched',
              'gwCV+ naive': 'gw_naive', 'gwCV+ matched': 'gw_matched', 'CV+ matched-system': 'cvp_msys',
              'nCV+ matched-system': 'ncvp_msys', 'gwCV+ matched-system': 'gw_msys', 'GP-native': 'gp'}
OLDNAME = {'CV+ naive': 'CV+naive', 'CV+ matched': 'CV+matched', 'nCV+ naive': 'nCV+naive', 'nCV+ matched': 'nCV+matched',
           'gwCV+ naive': 'gwCV+naive', 'gwCV+ matched': 'gwCV+matched', 'CV+ matched-system': 'CV+matched-sys',
           'nCV+ matched-system': 'nCV+matched-sys', 'gwCV+ matched-system': 'gwCV+matched-sys', 'GP-native': 'GP-native'}


def load_all(ds):
    acc = pd.read_csv(f"{FIN}/{ds}_acc.csv")
    it = pd.read_csv(f"{FIN}/{ds}_int.csv")
    it['col'] = it.method.map(METHOD_COL)
    w = it.pivot_table(index=['label', 'rid', 'model'], columns=['col', 'level'], values=['lo', 'hi'], aggfunc='first')
    w.columns = [f"{c}_{lh}{lvl}" for lh, c, lvl in w.columns]
    w = w.reset_index()
    d = acc.merge(w, on=['label', 'rid', 'model'], how='left')
    d['ad_knn'] = d['ad']
    for t in (90, 68):
        if f'ncvp_matched_hi{t}' in d:
            d[f'ncp_matched_hw{t}'] = (d[f'ncvp_matched_hi{t}'] - d[f'ncvp_matched_lo{t}']) / 2
    if 'gp_hi90' in d:
        d['gp_sd'] = (d['gp_hi90'] - d['gp_lo90']) / (2 * 1.6448536)
    return d


def metrics(g):
    e = g.yhat - g.y; ss = ((g.y - g.y.mean()) ** 2).sum()
    return dict(n=len(g), MAE=e.abs().mean(), RMSE=np.sqrt((e ** 2).mean()), R2=1 - (e ** 2).sum() / ss if ss > 0 else np.nan,
                rho=spearmanr(g.y, g.yhat).correlation if len(g) > 2 else np.nan, bias=e.mean())


def tabpfn_uq():
    """TabPFN interval summaries: native quantiles and naive CV+ (first Kaggle run, all protocols), matched CV+ for P2-P3
    (first run) and for P4-P6 with deployment-specific calibration (revision run)."""
    old = pd.read_csv(f"{RES}/ext_metrics.csv"); old = old[old.model == 'TabPFN']
    rows = []
    for (ds, proto), g in old.groupby(['dataset', 'protocol']):
        for key, meth in (('nat', 'TabPFN-native'), ('cvn', 'CV+naive'), ('cvm', 'CV+matched')):
            if key == 'cvm' and proto not in ('composition', 'system'):
                continue
            for lvl in (90, 68):
                c, w = f'cov_{key}{lvl}', f'w_{key}{lvl}'
                if g[c].isna().all():
                    continue
                rows.append(dict(dataset=ds, protocol=proto, model='TabPFN', method=f'{meth}{lvl}', nominal=lvl / 100,
                                 coverage=np.average(g[c], weights=g.n), mean_width=np.average(g[w], weights=g.n), n=int(g.n.sum())))
    p = f"{RES}/remote/tabpfn_v2_cvplus.csv"
    if os.path.exists(p):
        cv = pd.read_csv(p); lab2p = lambda l: 'external' if l.startswith('external') else l.split('-')[0]
        cv['protocol'] = cv.label.map(lab2p)
        for (ds, proto, var, lvl), g in cv.groupby(['dataset', 'protocol', 'variant', 'level']):
            rows.append(dict(dataset=ds, protocol=proto, model='TabPFN',
                             method=('CV+matched' if var == 'matched' else 'CV+matched-sys') + str(lvl), nominal=lvl / 100,
                             coverage=np.average(g.coverage, weights=g.n), mean_width=np.average(g.mean_width, weights=g.n),
                             n=int(g.n.sum())))
    return pd.DataFrame(rows)


def main():
    perf = pd.read_csv(f"{T1}/perf_v2.csv")
    perf.to_csv(f"{TAB}/perf_summary.csv", index=False)
    pd.read_csv(f"{T1}/pairwise_v2.csv").to_csv(f"{TAB}/pairwise.csv", index=False)
    rk = pd.read_csv(f"{T1}/ranking_v2.csv").rename(columns={'tau': 'kendall_tau_vs_random'})
    rk.to_csv(f"{TAB}/ranking.csv", index=False)
    folds = []
    for ds in ('HV', 'YS'):
        acc = pd.read_csv(f"{FIN}/{ds}_acc.csv")
        for (proto, model, lab), g in acc.groupby(['protocol', 'model', 'label']):
            folds.append(dict(dataset=ds, protocol=proto, model=model, label=lab, **metrics(g)))
    folds = pd.DataFrame(folds)
    folds.to_csv(f"{TAB}/perf_folds.csv", index=False)
    folds[folds.protocol == 'element'].to_csv(f"{TAB}/element_tasks.csv", index=False)
    # uncertainty
    uq = pd.read_csv(f"{T1}/uq_v2.csv")
    uq['method'] = uq.method.map(OLDNAME) + uq.level.astype(str)
    uq['nominal'] = uq.level / 100
    uq = pd.concat([uq, tabpfn_uq()], ignore_index=True)
    uq.to_csv(f"{TAB}/uq_summary.csv", index=False)
    byad, adc = [], []
    for ds in ('HV', 'YS'):
        d = load_all(ds); x = d[d.model == 'XGB']
        for proto, g in x.groupby('protocol'):
            adc.append(dict(dataset=ds, protocol=proto, model='XGB', median_ad=g.ad_knn.median(),
                            spearman_ad_abs_err=spearmanr(g.ad_knn, (g.yhat - g.y).abs()).correlation))
            q = g.ad_knn.quantile([1 / 3, 2 / 3]).values; b = np.digitize(g.ad_knn.values, q)
            for col, meth in (('cvp_naive', 'CV+naive'), ('cvp_matched', 'CV+matched'), ('ncvp_matched', 'nCV+matched'),
                              ('gw_matched', 'gwCV+matched')):
                for t in (90, 68):
                    lo, hi = f'{col}_lo{t}', f'{col}_hi{t}'
                    if lo not in g or g[lo].isna().all():
                        continue
                    for k in range(3):
                        s = g[b == k]
                        byad.append(dict(dataset=ds, protocol=proto, model='XGB', method=f'{meth}{t}', ad_tertile=k, n=len(s),
                                         coverage=float(((s.y >= s[lo]) & (s.y <= s[hi])).mean()),
                                         mean_width=float((s[hi] - s[lo]).mean()),
                                         RMSE=float(np.sqrt(((s.yhat - s.y) ** 2).mean()))))
    pd.DataFrame(byad).to_csv(f"{TAB}/uq_by_ad.csv", index=False)
    pd.DataFrame(adc).to_csv(f"{TAB}/ad_corr.csv", index=False)
    # unchanged analyses are carried over
    if os.path.exists(f"{T1}/uq_system_cov_v2.csv"):
        shutil.copy(f"{T1}/uq_system_cov_v2.csv", f"{TAB}/uq_system_cov.csv")
    for f in ('ablation_summary.csv', 'dedup_sensitivity.csv', 'noise_floor.json', 'shap_importance_HV.csv', 'shap_importance_YS.csv',
              'shap_stability_HV.csv', 'shap_stability_YS.csv', 'shap_stability_HV.json', 'shap_stability_YS.json', 'shap_summary.json', 'nn_comp_distance.csv',
              'nn_comp_distance_summary.csv', 'random_seen_unseen.csv', 'noise_floor_v2.json', 'temporal_overlap_v2.csv',
              'external_subsets_v2.csv', 'ad_sensitivity_v2.csv', 'tuning_scheme_v2.csv', 'tabpfn_config.json'):
        if os.path.exists(f"{T1}/{f}"):
            shutil.copy(f"{T1}/{f}", f"{TAB}/{f}")
    print(perf.pivot_table(index=['dataset', 'model'], columns='protocol', values='RMSE').round(1))
    print(uq[(uq.model == 'XGB') & uq.method.str.endswith('90')].pivot_table(index=['dataset', 'method'], columns='protocol',
                                                                              values='coverage').round(3))


if __name__ == '__main__':
    main()

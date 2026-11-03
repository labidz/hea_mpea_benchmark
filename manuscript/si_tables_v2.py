"""Revision-2 Supplementary tables."""
import json
import numpy as np


def _f(v, fmt='{:.2f}'):
    return '–' if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
import pandas as pd
from tables import PNAME

MODELS = ['Mean', 'kNN', 'Ridge', 'SVR', 'GPR', 'MLP', 'RF', 'XGB', 'TabPFN']
PROTOS = ['random', 'composition', 'system', 'element', 'temporal', 'external']
DSN = {'HV': 'Hardness', 'YS': 'Yield strength'}


def s_tabpfn(cfg):
    ic = cfg['inference_config (v2 defaults)']
    rows = [['Package', cfg['package']], ['Model', 'TabPFNRegressor.create_default_for_version(ModelVersion.V2), device = CUDA, '
                                                 'random_state = split seed mod 10,000'],
            ['Checkpoint', cfg['checkpoint']], ['Ensemble size', cfg['n_estimators']],
            ['Softmax temperature', 'checkpoint default ("auto")'], ['Inference precision', 'automatic'],
            ['Inference configuration', f"outlier removal {ic['OUTLIER_REMOVAL_STD']}; feature shift {ic['FEATURE_SHIFT_METHOD']}; "
                                        f"fingerprint feature on; polynomial features {ic['POLYNOMIAL_FEATURES']}; no sample "
                                        f"subsampling; target transforms none / safe power; limits {ic['MAX_NUMBER_OF_SAMPLES']} "
                                        f"samples, {ic['MAX_NUMBER_OF_FEATURES']} features"],
            ['Inputs', cfg['input']], ['Point prediction', 'predictive mean'],
            ['Native intervals', cfg['native intervals']], ['Tuning', cfg['tuning / fine-tuning']], ['Hardware', cfg['hardware']]]
    return dict(caption="**Table S2b.** Complete TabPFN configuration.", header=['Setting', 'Value'], rows=rows, widths=[1.6, 5.6])


def s_selected(P1, P2):
    """Most frequently selected hyper-parameter configuration per model and protocol."""
    rows = []
    for ds in ('HV', 'YS'):
        recs = []
        for path, job in ((P1[ds], 'v1'), (P2[ds], 'v2')):
            for l in open(path):
                j = json.loads(l)
                lab = j['label']; proto = 'external' if lab.startswith('external') else lab.split('-')[0]
                if job == 'v1' and (proto in ('element', 'temporal', 'external') or j['model'] == 'Ridge'):
                    continue
                if job == 'v2' and j.get('job') == 'xgbuq':
                    continue
                if not j['params']:
                    continue
                p = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in j['params'].items()}
                recs.append(dict(ds=ds, proto=proto, model=j['model'], cfg=json.dumps(p, sort_keys=True)))
        R = pd.DataFrame(recs).drop_duplicates()
        for m in ['kNN', 'Ridge', 'SVR', 'RF', 'XGB', 'MLP']:
            for proto in PROTOS:
                g = R[(R.model == m) & (R.proto == proto)]
                if g.empty:
                    continue
                vc = g.cfg.value_counts()
                txt = vc.index[0].replace('"', '').replace('{', '').replace('}', '').replace('hidden_layer_sizes', 'layers') \
                    .replace('learning_rate', 'η').replace('colsample_bytree', 'colsample').replace('min_child_weight', 'mcw') \
                    .replace('max_features', 'max_feat').replace('min_samples_leaf', 'min_leaf').replace('n_neighbors', 'k')
                rows.append([ds, m, PNAME[proto], txt, f"{vc.iloc[0]}/{len(g)}"])
    return dict(caption="**Table S2c.** Most frequently selected hyper-parameters (selected in *n* of *N* outer folds or tasks).",
                header=['Data', 'Model', 'Protocol', 'Configuration', '*n*/*N*'], rows=rows, widths=[0.6, 0.7, 1.4, 4.6, 0.6])


def s_perf(perf, ds, n):
    g = perf[perf.dataset == ds]; rows = []
    for p in PROTOS:
        for m in MODELS:
            s = g[(g.protocol == p) & (g.model == m)]
            if s.empty:
                continue
            r = s.iloc[0]
            rows.append([PNAME[p], m, f"{r.MAE:.0f}", f"{r.RMSE:.0f} [{r.RMSE_lo:.0f}, {r.RMSE_hi:.0f}]", f"{r.R2:.2f}",
                         _f(r.rho), f"{r.bias:+.0f}"])
    unit = 'HV' if ds == 'HV' else 'MPa'
    return dict(caption=f"**Table S{n}.** Complete accuracy metrics for {DSN[ds].lower()} ({unit}) with cluster-bootstrap 95% "
                        f"confidence intervals of the RMSE.",
                header=['Protocol', 'Model', 'MAE', 'RMSE [95% CI]', '*R*^{2}', 'Spearman *ρ*', 'Bias'], rows=rows,
                widths=[1.4, 0.8, 0.8, 1.6, 0.8, 1.0, 0.8],
                notes="P1–P3: metrics of the pooled out-of-fold predictions of each repeat, averaged over three repeats; P4: mean "
                      "over held-out elements; P5–P6: single test set. CIs: 2,000 cluster-bootstrap replicates with alloy systems "
                      "as resampling units. Bias = mean(predicted − measured).")


def s_element(folds, ds, n):
    g = folds[(folds.dataset == ds) & (folds.protocol == 'element')].copy()
    g['element'] = g.label.str.replace('element-', '')
    P = g.pivot_table(index='element', columns='model', values='RMSE')
    N = g.groupby('element').n.first()
    cols = [m for m in MODELS if m in P.columns]
    rows = [[f"{e} ({int(N[e])})"] + [f"{P.loc[e, m]:.0f}" for m in cols] for e in P.index]
    return dict(caption=f"**Table S{n}.** Leave-one-element-out RMSE for {DSN[ds].lower()} "
                        f"({'HV' if ds == 'HV' else 'MPa'}); number of held-out test records in parentheses.",
                header=['Held-out element'] + cols, rows=rows, widths=[1.4] + [0.7] * len(cols))


def s_pairwise(pair, n):
    rows = []
    for r in pair.itertuples():
        if r.model == r.best:
            continue
        rows.append([r.dataset, PNAME[r.protocol], r.best, r.model, f"{r.diff:+.0f}", f"[{r.diff_lo:+.0f}, {r.diff_hi:+.0f}]",
                     'yes' if r.diff_lo <= 0 else 'no', f"{r.p_best:.2f}"])
    return dict(caption=f"**Table S{n}.** Paired cluster-bootstrap comparison of every model with the most accurate model.",
                header=['Data', 'Protocol', 'Best', 'Model', 'ΔRMSE', '95% CI', 'Indist.', '*P*(best)'], rows=rows,
                widths=[0.55, 1.3, 0.75, 0.75, 0.7, 1.1, 0.6, 0.7],
                notes="ΔRMSE = RMSE(model) − RMSE(best); 2,000 replicates resampling alloy systems, identical draws for all "
                      "models. Indist.: CI contains 0. *P*(best): fraction of replicates in which the model had the lowest RMSE.")


def s_uq(uq, n):
    rows = []
    U = uq[uq.method.str.endswith('90') | uq.method.str.endswith('68')].copy()
    for r in U.sort_values(['dataset', 'model', 'method', 'protocol']).itertuples():
        ci = f"[{r.cov_lo:.2f}, {r.cov_hi:.2f}]" if not pd.isna(getattr(r, 'cov_lo', np.nan)) else '–'
        sysc = f"{r.sys_cov_median:.2f}" if not pd.isna(getattr(r, 'sys_cov_median', np.nan)) else '–'
        wk = '–' if pd.isna(getattr(r, 'winkler', np.nan)) else f"{r.winkler:.0f}"
        rows.append([r.dataset, r.model, r.method, PNAME.get(r.protocol, r.protocol), f"{r.coverage:.2f}", ci,
                     f"{r.mean_width:.0f}", wk, sysc])
    return dict(caption=f"**Table S{n}.** Empirical coverage (with cluster-bootstrap 95% CI), mean width, Winkler score and median "
                        f"per-system coverage of all prediction intervals (nominal level in the method name).",
                header=['Data', 'Model', 'Method', 'Protocol', 'Cov.', '95% CI', 'Width', 'Winkler', 'Sys. cov.'], rows=rows,
                widths=[0.5, 0.7, 1.45, 1.3, 0.55, 0.95, 0.6, 0.7, 0.65],
                notes="CV+: cross-conformal intervals with random (naive), deployment-matched (matched; system-grouped for P6 as "
                      "'matched-sys') calibration folds; nCV+: normalized CV+; gwCV+: group-weighted CV+; GP-native: Gaussian "
                      "posterior interval; TabPFN-native: predicted quantiles. TabPFN rows come from per-fold summaries (no "
                      "per-record intervals), hence no CI or Winkler score. Sys. cov.: median coverage over alloy systems with ≥ 5 "
                      "test records in a fold or test set.")


def s_external(ext, n):
    a = ext[ext.method.isna()] if 'method' in ext else ext
    rows = []
    for r in a.itertuples():
        rows.append([r.subset, r.model, int(r.n), int(r.n_systems), f"{r.RMSE:.0f} [{r.RMSE_lo:.0f}, {r.RMSE_hi:.0f}]",
                     f"{r.R2:.2f}", _f(r.rho), f"{r.bias:+.0f}"])
    return dict(caption=f"**Table S{n}.** External validation (P6) by subset of the GE refractory alloys.",
                header=['Subset', 'Model', '*n*', 'Systems', 'RMSE [95% CI]', '*R*^{2}', '*ρ*', 'Bias'], rows=rows,
                widths=[2.2, 0.7, 0.45, 0.65, 1.5, 0.6, 0.55, 0.6],
                notes="Overlap compositions: the external compositions also present in the MPEA hardness data. Previously "
                      "represented systems: element sets present in the MPEA hardness data. CIs: cluster bootstrap over GE alloy "
                      "systems (2,000 replicates).")


def s_temporal(tov, n):
    lab = [('n_train', 'Training records (≤ 2018)', '{:.0f}'), ('n_test', 'Test records (2019)', '{:.0f}'),
           ('test_publications', 'Test publications', '{:.0f}'), ('test_pubs_also_in_train', 'Test publications also in training', '{:.0f}'),
           ('frac_comp_seen', 'Test records with a composition reported before 2019', '{:.0%}'),
           ('frac_comp_and_proc_seen', '… with the same composition and processing class', '{:.0%}'),
           ('frac_system_seen', 'Test records from a previously reported alloy system', '{:.0%}'),
           ('frac_new_element', 'Test records containing an element absent before 2019', '{:.0%}'),
           ('new_elements', 'Elements absent before 2019', '{}'),
           ('median_nn_at_pct', 'Median composition change to nearest earlier alloy (at.%)', '{:.1f}'),
           ('frac_nn_le_2at', 'Test records within 2 at.% of an earlier alloy', '{:.0%}'),
           ('proc_total_variation', 'Shift of processing-class mix (total variation distance)', '{:.2f}'),
           ('frac_unknown_proc_train', 'Records without processing information, training', '{:.0%}'),
           ('frac_unknown_proc_test', 'Records without processing information, test', '{:.0%}'),
           ('mean_target_train', 'Mean target, training', '{:.0f}'), ('mean_target_test', 'Mean target, test', '{:.0f}')]
    t = tov.set_index('dataset')
    rows = [[name] + [fmt.format(t.loc[ds, k]) if not pd.isna(t.loc[ds, k]) else '–' for ds in ('HV', 'YS')] for k, name, fmt in lab]
    return dict(caption=f"**Table S{n}.** Overlap between the training literature (≤ 2018) and the temporal test set (2019).",
                header=['Quantity', 'Hardness', 'Yield strength'], rows=rows, widths=[4.6, 1.2, 1.2])


def s_noise(nz, n):
    names = {'same_processing': 'Same composition and processing class (reported floor)',
             'same_processing_cross_publication': '… groups spanning ≥ 2 publications',
             'same_processing_same_publication': '… within a single publication',
             'same_composition_any_processing': 'Same composition, any processing'}
    rows = []
    for ds in ('HV', 'YS'):
        for k, nm in names.items():
            v = nz[ds].get(k)
            if v is None:
                continue
            rows.append([DSN[ds], nm, f"{v['pooled_sd']:.0f} [{v['ci'][0]:.0f}, {v['ci'][1]:.0f}]", v['groups'], v['records'],
                         f"{v['median_group_sd']:.0f}"])
    return dict(caption=f"**Table S{n}.** Experimental variability of repeated records (pooled within-group SD with group-bootstrap 95% CI).",
                header=['Data', 'Grouping', 'Pooled SD [95% CI]', 'Groups', 'Records', 'Median group SD'], rows=rows,
                widths=[0.9, 3.0, 1.4, 0.6, 0.65, 0.9],
                notes="Yield-strength groups also share test type and temperature. Units: HV or MPa.")


def s_adsens(ads, n):
    rows = []
    for r in ads.sort_values(['dataset', 'space', 'k']).itertuples():
        rows.append([r.dataset, r.space, int(r.k), f"{r.rho_pooled:.2f}", f"{r.rho_population:.2f}",
                     f"{getattr(r, 'rho_system', np.nan):.2f}", f"{getattr(r, 'rho_element', np.nan):.2f}",
                     f"{getattr(r, 'rho_temporal', np.nan):.2f}",
                     '–' if pd.isna(getattr(r, 'rho_external', np.nan)) else f"{r.rho_external:.2f}",
                     f"{r.retained50:.2f}"])
    return dict(caption=f"**Table S{n}.** Sensitivity of the applicability-domain analysis to the neighbourhood size *k* and to "
                        f"the descriptor representation (XGBoost errors).",
                header=['Data', 'Representation', '*k*', '*ρ* pooled', '*ρ* popul.', '*ρ* P3', '*ρ* P4', '*ρ* P5', '*ρ* P6', 'Ret. 50%'],
                rows=rows, widths=[0.45, 2.3, 0.35, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.65],
                notes="*ρ* pooled: Spearman correlation between distance and absolute error pooled over P2–P6; *ρ* popul.: correlation "
                      "between the protocol-median distance and the protocol RMSE; Ret. 50%: RMSE/SD of the half of the P3–P5 "
                      "predictions with the smallest distance (all P3–P5 predictions and an error oracle in the main text).")


def s_external_cov(ext, n):
    """Coverage of the XGBoost 90% intervals by subset of the external alloys."""
    u = ext[ext.method.notna()]
    order = ['CV+ naive', 'CV+ matched', 'CV+ matched-system', 'gwCV+ matched', 'nCV+ naive', 'nCV+ matched', 'nCV+ matched-system']
    subsets = list(dict.fromkeys(ext.subset))
    P = u.pivot_table(index='subset', columns='method', values='coverage90')
    W = u.pivot_table(index='subset', columns='method', values='mean_width90')
    N = u.groupby('subset').n.first()
    rows = [[f"{s} ({int(N[s])})"] + [f"{P.loc[s, m]:.2f}" for m in order] for s in subsets if s in P.index]
    rows.append(['Mean width, all alloys (HV)'] + [f"{W.loc['all', m]:.0f}" for m in order])
    return dict(caption=f"**Table S{n}b.** Empirical coverage of nominal 90% XGBoost intervals on subsets of the external alloys "
                        f"(number of alloys in parentheses).",
                header=['Subset', 'CV+ naive', 'CV+ elem.', 'CV+ sys.', 'gwCV+ elem.', 'nCV+ naive', 'nCV+ elem.', 'nCV+ sys.'],
                rows=rows, widths=[2.2, 0.72, 0.72, 0.72, 0.8, 0.8, 0.8, 0.8],
                notes="elem.: leave-one-element-out calibration folds (primary P6 scheme, fixed before evaluation); sys.: "
                      "system-grouped calibration folds (sensitivity analysis); naive: random folds.")


def s_tuning(ts, n):
    rows = [[DSN[r.dataset], PNAME[r.protocol], r.model, f"{r.rmse_system_grouped:.0f}", f"{r.rmse_deployment:.0f}",
             f"{r.change_pct:+.1f}"] for r in ts.sort_values(['dataset', 'protocol', 'model']).itertuples()]
    return dict(caption=f"**Table S{n}.** Effect of the inner-validation scheme used for hyper-parameter selection on the test RMSE "
                        f"under P4–P6.",
                header=['Data', 'Protocol', 'Model', 'System-grouped tuning', 'Deployment-specific tuning', 'Change (%)'],
                rows=rows, widths=[1.1, 1.5, 0.7, 1.3, 1.5, 0.9],
                notes="Deployment-specific: leave-one-element-out inner folds for P4 and P6, forward-chaining for P5. "
                      "System-grouped: the P3 scheme. Ridge regression (different search space in the system-grouped run) and "
                      "GPR (kernel fitted by marginal likelihood under both schemes) are omitted. Metrics as in Table S3.")

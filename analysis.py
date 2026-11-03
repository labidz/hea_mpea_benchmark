"""Aggregate benchmark predictions into tables and statistics.
Outputs (results/tables/): perf_summary.csv, perf_folds.csv, element_tasks.csv, uq_summary.csv,
uq_by_ad.csv, ad_corr.csv, friedman.csv, ranking.csv"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon, kendalltau
from io_utils import read_pred

RES = "/home/claude/hea/results"
TAB = f"{RES}/tables"
os.makedirs(TAB, exist_ok=True)
MODELS = ['Mean', 'kNN', 'Ridge', 'SVR', 'GPR', 'RF', 'XGB', 'MLP', 'TabPFN']
PROTOS = ['random', 'composition', 'system', 'element', 'temporal', 'external']


def m_(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); e = p - y
    ss = ((y - y.mean()) ** 2).sum()
    return dict(n=len(y), MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), R2=1 - (e ** 2).sum() / ss,
                rho=spearmanr(y, p).correlation, bias=e.mean())


REMOTE = {'YS': ('GPR', 'RF', 'MLP')}  # models whose yield-strength runs were executed on Kaggle CPU


def tabpfn_folds(ds):
    """Per-fold metrics of models run on Kaggle (TabPFN on GPU; YS GPR/RF/MLP on CPU) from results/ext_metrics.csv."""
    p = f"{RES}/ext_metrics.csv"
    if not os.path.exists(p):
        return None
    t = pd.read_csv(p)
    return t[t.dataset == ds].copy()


def load_all(ds):
    d = read_pred(f"{RES}/{ds}_pred.csv")
    d = d[~d.model.isin(REMOTE.get(ds, ()))]
    if ds == 'HV' and os.path.exists(f"{RES}/HV_external_pred.csv"):
        d = pd.concat([d, read_pred(f"{RES}/HV_external_pred.csv")], ignore_index=True)
    npth = f"{RES}/{ds}_ncp.csv"
    if os.path.exists(npth):
        n = pd.read_csv(npth)[['label', 'rid', 'ad_w', 'ad_raw', 'ncp_naive_hw90', 'ncp_naive_hw68', 'ncp_matched_hw90', 'ncp_matched_hw68']]
        n = n.rename(columns={c: c + '_w' for c in n.columns if c.startswith('ncp_')})
        d = d.merge(n, on=['label', 'rid'], how='left')
        d['ad_knn'] = d['ad_w']  # winsorized AD distance is the reported AD score
        for c in ('ncp_naive_hw90', 'ncp_naive_hw68', 'ncp_matched_hw90', 'ncp_matched_hw68'):
            d[c] = np.where(d.model == 'XGB', d[c + '_w'], np.nan)  # normalized CV+ reported for XGB only
    tp = f"{RES}/tabpfn_records_{ds}.csv"
    if os.path.exists(tp):
        t = pd.read_csv(tp)
        d = pd.concat([d, t], ignore_index=True)
    return d


def bootstrap_ci(y, p, stat='RMSE', B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); vals = []
    for _ in range(B):
        i = rng.integers(0, n, n); vals.append(m_(y[i], p[i])[stat])
    return np.percentile(vals, [2.5, 97.5])


def perf_tables(ds, d, tab_folds=None):
    """Per-fold metrics -> summary. P1-P4: mean and SD over folds/tasks; P5-P6: single test set with bootstrap CI.
    tab_folds: optional per-fold metrics of models run elsewhere (TabPFN on Kaggle)."""
    folds = []
    for (proto, model, lab), gl in d.groupby(['protocol', 'model', 'label']):
        folds.append(dict(dataset=ds, protocol=proto, model=model, label=lab, **m_(gl.y, gl.yhat)))
    folds = pd.DataFrame(folds)
    if tab_folds is not None and len(tab_folds):
        folds = pd.concat([folds, tab_folds[folds.columns.intersection(tab_folds.columns)]], ignore_index=True)
    rows = []
    for (proto, model), g in folds.groupby(['protocol', 'model']):
        if proto in ('random', 'composition', 'system', 'element'):
            r = {k: g[k].mean() for k in ('MAE', 'RMSE', 'R2', 'rho', 'bias')}
            r.update({k + '_sd': g[k].std(ddof=1) for k in ('MAE', 'RMSE', 'R2', 'rho')})
            r['R2_median'] = g.R2.median(); r['n'] = int(g.n.sum()); r['tasks'] = len(g)
        else:
            r = g.iloc[0][['n', 'MAE', 'RMSE', 'R2', 'rho', 'bias']].to_dict(); r['tasks'] = 1
            gl = d[(d.protocol == proto) & (d.model == model)]
            if len(gl):
                for st in ('RMSE', 'R2'):
                    lo, hi = bootstrap_ci(gl.y.values, gl.yhat.values, st); r[st + '_lo'], r[st + '_hi'] = lo, hi
        rows.append(dict(dataset=ds, protocol=proto, model=model, **r))
    return pd.DataFrame(rows), folds


def friedman(folds):
    out = []
    for (ds, proto), g in folds.groupby(['dataset', 'protocol']):
        if g.label.nunique() < 5:
            continue
        P = g.pivot_table(index='label', columns='model', values='RMSE').dropna(axis=1)
        P = P[[m for m in MODELS if m in P.columns and m != 'Mean']]
        stat, p = friedmanchisquare(*[P[c].values for c in P.columns])
        ranks = P.rank(axis=1).mean().sort_values()
        best = ranks.index[0]
        row = dict(dataset=ds, protocol=proto, n_tasks=len(P), friedman_chi2=stat, friedman_p=p, best=best,
                   mean_ranks="; ".join(f"{k}:{v:.2f}" for k, v in ranks.items()))
        # Wilcoxon signed-rank (best vs each), Holm-corrected
        pv = []
        for c in P.columns:
            if c == best:
                continue
            pv.append((c, wilcoxon(P[best], P[c]).pvalue))
        pv.sort(key=lambda x: x[1]); m = len(pv); adj = []
        running = 0
        for i, (c, pval) in enumerate(pv):
            running = max(running, min(1.0, pval * (m - i)))
            adj.append(f"{c}:{running:.3g}")
        row['holm_vs_best'] = "; ".join(adj)
        out.append(row)
    return pd.DataFrame(out)


def ranking_stability(perf):
    out = []
    for ds, g in perf.groupby('dataset'):
        base = g[g.protocol == 'random'].set_index('model').RMSE
        for proto in PROTOS[1:]:
            o = g[g.protocol == proto].set_index('model').RMSE
            common = [m for m in base.index if m in o.index and m != 'Mean']
            if len(common) < 4:
                continue
            tau = kendalltau(base[common], o[common]).correlation
            out.append(dict(dataset=ds, protocol=proto, kendall_tau_vs_random=tau,
                            best_random=base[common].idxmin(), best_here=o[common].idxmin()))
    return pd.DataFrame(out)


def coverage(y, lo, hi):
    return float(((y >= lo) & (y <= hi)).mean())


def uq_tables(ds, d):
    rows, byad = [], []
    for (proto, model), g in d.groupby(['protocol', 'model']):
        if model == 'TabPFN':
            continue
        methods = {}
        if model == 'GPR':
            for t, z in ((90, 1.6449), (68, 0.9945)):
                methods[f'GP-native{t}'] = (g.yhat - z * g.gp_sd, g.yhat + z * g.gp_sd)
        if model == 'TabPFN':
            methods['TabPFN-native90'] = (g.q05, g.q95)
            methods['TabPFN-native68'] = (g.q16, g.q84)
        for v in ('naive', 'matched'):
            for t in (90, 68):
                lo, hi = f'cvp_{v}_lo{t}', f'cvp_{v}_hi{t}'
                if lo in g and g[lo].notna().all():
                    methods[f'CV+{v}{t}'] = (g[lo], g[hi])
                hw = f'ncp_{v}_hw{t}'
                if hw in g and g[hw].notna().all():
                    methods[f'nCV+{v}{t}'] = (g.yhat - g[hw], g.yhat + g[hw])
        if not methods:
            continue
        # AD tertile edges defined on the pooled test distances of this protocol
        q = g.ad_knn.quantile([1 / 3, 2 / 3]).values if 'ad_knn' in g and g.ad_knn.notna().all() else None
        for name, (lo, hi) in methods.items():
            nominal = int(name[-2:]) / 100
            per_task = [coverage(gl.y.values, lo[gl.index].values, hi[gl.index].values) for _, gl in g.groupby('label')]
            w = (hi - lo)
            err = np.abs(g.yhat - g.y)
            alpha = 1 - nominal
            # interval (Winkler) score
            wink = w + (2 / alpha) * ((lo - g.y).clip(lower=0) + (g.y - hi).clip(lower=0))
            rows.append(dict(dataset=ds, protocol=proto, model=model, method=name, nominal=nominal,
                             coverage=coverage(g.y.values, lo.values, hi.values),
                             coverage_task_min=np.min(per_task), coverage_task_sd=np.std(per_task),
                             mean_width=w.mean(), median_width=w.median(), winkler=wink.mean(),
                             spearman_width_abs_err=spearmanr(w, err).correlation if w.std() > 0 else np.nan))
            if q is not None:
                bins = np.digitize(g.ad_knn.values, q)
                for b in range(3):
                    sel = bins == b
                    byad.append(dict(dataset=ds, protocol=proto, model=model, method=name, ad_tertile=b,
                                     n=int(sel.sum()), coverage=coverage(g.y.values[sel], lo.values[sel], hi.values[sel]),
                                     mean_width=float(w.values[sel].mean()),
                                     RMSE=float(np.sqrt(((g.yhat.values[sel] - g.y.values[sel]) ** 2).mean()))))
    return pd.DataFrame(rows), pd.DataFrame(byad)


def tabpfn_uq(ds):
    """UQ summary rows for TabPFN from per-fold Kaggle metrics (coverage and width are n-weighted means = pooled values)."""
    t = tabpfn_folds(ds)
    if t is None:
        return pd.DataFrame()
    rows = []
    for (proto, model), g in t.groupby(['protocol', 'model']):
        for key, meth in (('nat', 'TabPFN-native'), ('gp', 'GP-native'), ('cvn', 'CV+naive'), ('cvm', 'CV+matched')):
            for lvl in (90, 68):
                c, w = f'cov_{key}{lvl}', f'w_{key}{lvl}'
                if c not in g or g[c].isna().all():
                    continue
                rows.append(dict(dataset=ds, protocol=proto, model=model, method=f'{meth}{lvl}', nominal=lvl / 100,
                                 coverage=np.average(g[c], weights=g.n), coverage_task_min=g[c].min(),
                                 coverage_task_sd=g[c].std(ddof=0), mean_width=np.average(g[w], weights=g.n),
                                 median_width=np.nan, winkler=np.nan, spearman_width_abs_err=np.nan))
    return pd.DataFrame(rows)


def ad_corr(ds, d):
    out = []
    for (proto, model), g in d.groupby(['protocol', 'model']):
        if 'ad_knn' not in g or g.ad_knn.isna().any() or model == 'Mean':
            continue
        err = np.abs(g.yhat - g.y)
        out.append(dict(dataset=ds, protocol=proto, model=model, n=len(g),
                        spearman_ad_abs_err=spearmanr(g.ad_knn, err).correlation,
                        median_ad=g.ad_knn.median()))
    return pd.DataFrame(out)


def main(datasets=('HV', 'YS')):
    P, F, U, UA, A = [], [], [], [], []
    for ds in datasets:
        if not os.path.exists(f"{RES}/{ds}_pred.csv"):
            continue
        d = load_all(ds)
        # attach AD distance to TabPFN rows from the classical runs (same splits -> same distances)
        adm = d[d.model == 'XGB'][['label', 'rid', 'ad_knn']].drop_duplicates(['label', 'rid'])
        if 'TabPFN' in set(d.model):
            t = d[d.model == 'TabPFN'].drop(columns=['ad_knn', 'ad_w', 'ad_raw'], errors='ignore').merge(adm, on=['label', 'rid'], how='left')
            d = pd.concat([d[d.model != 'TabPFN'], t], ignore_index=True)
        tf = tabpfn_folds(ds)
        p, f = perf_tables(ds, d, tf); P.append(p); F.append(f)
        u, ua = uq_tables(ds, d); U.append(u); UA.append(ua); U.append(tabpfn_uq(ds))
        A.append(ad_corr(ds, d))
    perf = pd.concat(P); folds = pd.concat(F)
    perf.to_csv(f"{TAB}/perf_summary.csv", index=False); folds.to_csv(f"{TAB}/perf_folds.csv", index=False)
    folds[folds.protocol == 'element'].to_csv(f"{TAB}/element_tasks.csv", index=False)
    friedman(folds).to_csv(f"{TAB}/friedman.csv", index=False)
    ranking_stability(perf).to_csv(f"{TAB}/ranking.csv", index=False)
    pd.concat(U).to_csv(f"{TAB}/uq_summary.csv", index=False)
    pd.concat(UA).to_csv(f"{TAB}/uq_by_ad.csv", index=False)
    pd.concat(A).to_csv(f"{TAB}/ad_corr.csv", index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(perf.pivot_table(index=['dataset', 'model'], columns='protocol', values='RMSE').round(1))
    print(perf.pivot_table(index=['dataset', 'model'], columns='protocol', values='R2').round(3))


if __name__ == '__main__':
    main(tuple(sys.argv[1].split(',')) if len(sys.argv) > 1 else ('HV', 'YS'))

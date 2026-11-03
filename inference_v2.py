"""Revision-2 statistical inference with alloy systems as the resampling unit (cluster bootstrap).

Point estimates
  P1-P3: RMSE of the pooled out-of-fold predictions of each repeat, averaged over the three repeats
  P4   : mean over held-out elements of the per-element RMSE
  P5,P6: RMSE of the single test set
Cluster bootstrap (B = 2000): alloy systems (element sets) are drawn with replacement; every record of a drawn system
enters with multiplicity equal to the number of draws. The same draws are applied to every model (paired comparisons)
and to every protocol of a dataset (joint uncertainty of rankings). P6 resamples the GE systems independently.
Outputs (results/tables): perf_v2.csv, pairwise_v2.csv, ranking_v2.csv, uq_v2.csv, uq_system_cov_v2.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

F = "/home/claude/hea/results/final"; T = "/home/claude/hea/results/tables"
B = 2000
MODELS = ['Mean', 'kNN', 'Ridge', 'SVR', 'GPR', 'MLP', 'RF', 'XGB', 'TabPFN']
PROTOS = ['random', 'composition', 'system', 'element', 'temporal', 'external']


def system_weights(systems, B, seed):
    """(B, n_systems) bootstrap multiplicities."""
    rng = np.random.default_rng(seed)
    G = len(systems)
    return rng.multinomial(G, np.full(G, 1 / G), size=B).astype(float)


def unit_index(df):
    """Rows grouped into evaluation units: (repeat) for P1-P3, (element task) for P4, single unit otherwise."""
    if df.protocol.iloc[0] in ('random', 'composition', 'system'):
        return df.rep.values
    if df.protocol.iloc[0] == 'element':
        return pd.factorize(df.label)[0]
    return np.zeros(len(df), int)


def weighted_rmse(err2, W, unit):
    """err2: (n,), W: (B, n) record weights, unit: (n,) -> (B,) mean over units of weighted RMSE."""
    out = []
    for u in np.unique(unit):
        s = unit == u
        num = W[:, s] @ err2[s]; den = W[:, s].sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            out.append(np.sqrt(num / den))
    return np.nanmean(np.vstack(out), axis=0)


def point_rmse(err2, unit):
    return np.mean([np.sqrt(err2[unit == u].mean()) for u in np.unique(unit)])


def r2_point(df):
    vals = []
    for u in np.unique(unit_index(df)):
        g = df[unit_index(df) == u]
        vals.append(1 - ((g.yhat - g.y) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum())
    return float(np.mean(vals))


def accuracy(ds, seed=2026):
    acc = pd.read_csv(f"{F}/{ds}_acc.csv")
    rows, pair, boot_store = [], [], {}
    for proto in PROTOS:
        a = acc[acc.protocol == proto]
        if a.empty:
            continue
        systems = np.array(sorted(a.sys_key.unique()))
        Wsys = system_weights(systems, B, seed + (17 if proto == 'external' else 0))
        sidx = {s: i for i, s in enumerate(systems)}
        models = [m for m in MODELS if m in set(a.model)]
        # align: every model must have the same (label, rid) set
        ref = a[a.model == 'XGB'][['label', 'rid', 'sys_key', 'rep', 'protocol']].sort_values(['label', 'rid']).reset_index(drop=True)
        unit = unit_index(ref)
        W = Wsys[:, [sidx[s] for s in ref.sys_key]]
        bs = {}
        for m in models:
            g = a[a.model == m].set_index(['label', 'rid']).loc[list(zip(ref.label, ref.rid))].reset_index()
            e2 = ((g.yhat - g.y) ** 2).values
            bs[m] = weighted_rmse(e2, W, unit)
            rows.append(dict(dataset=ds, protocol=proto, model=m, RMSE=point_rmse(e2, unit), R2=r2_point(g.assign(protocol=proto)),
                             RMSE_lo=np.nanpercentile(bs[m], 2.5), RMSE_hi=np.nanpercentile(bs[m], 97.5),
                             MAE=float(np.mean([np.abs(g.yhat - g.y)[unit == u].mean() for u in np.unique(unit)])),
                             rho=float(np.mean([spearmanr(g.y[unit == u], g.yhat[unit == u]).correlation for u in np.unique(unit)])),
                             bias=float((g.yhat - g.y).mean()), n_records=len(g), n_systems=len(systems), n_units=len(np.unique(unit))))
        pt = {r['model']: r['RMSE'] for r in rows if r['protocol'] == proto and r['dataset'] == ds}
        cand = [m for m in models if m != 'Mean']
        best = min(cand, key=lambda m: pt[m])
        for m in cand:
            d = bs[m] - bs[best]
            pair.append(dict(dataset=ds, protocol=proto, model=m, best=best, diff=pt[m] - pt[best],
                             diff_lo=np.nanpercentile(d, 2.5), diff_hi=np.nanpercentile(d, 97.5),
                             p_best=float(np.mean(np.vstack([bs[c] for c in cand]).argmin(0) == cand.index(m)))))
        boot_store[proto] = (cand, bs)
    return pd.DataFrame(rows), pd.DataFrame(pair), boot_store


def ranking(ds, perf, boot):
    out = []
    if 'random' not in boot:
        return pd.DataFrame()
    base_c, base_bs = boot['random']
    for proto, (cand, bs) in boot.items():
        if proto == 'random':
            continue
        common = [m for m in base_c if m in cand]
        p0 = perf[(perf.dataset == ds) & (perf.protocol == 'random')].set_index('model').RMSE
        p1 = perf[(perf.dataset == ds) & (perf.protocol == proto)].set_index('model').RMSE
        tau = kendalltau(p0[common], p1[common]).correlation
        taus = [kendalltau([base_bs[m][b] for m in common], [bs[m][b] for m in common]).correlation for b in range(B)]
        out.append(dict(dataset=ds, protocol=proto, tau=tau, tau_lo=np.nanpercentile(taus, 2.5),
                        tau_hi=np.nanpercentile(taus, 97.5), n_models=len(common)))
    return pd.DataFrame(out)


def uq(ds, seed=2026):
    it = pd.read_csv(f"{F}/{ds}_int.csv")
    rows, sysrows = [], []
    for (proto, model, method, lvl), g in it.groupby(['protocol', 'model', 'method', 'level']):
        g = g.reset_index(drop=True)
        systems = np.array(sorted(g.sys_key.unique()))
        Wsys = system_weights(systems, B, seed + (17 if proto == 'external' else 0) + 3)
        sidx = {s: i for i, s in enumerate(systems)}
        W = Wsys[:, [sidx[s] for s in g.sys_key]]
        inside = ((g.y >= g.lo) & (g.y <= g.hi)).values.astype(float)
        width = (g.hi - g.lo).values
        a = 1 - lvl / 100
        wink = width + (2 / a) * (np.clip(g.lo - g.y, 0, None) + np.clip(g.y - g.hi, 0, None)).values
        cb = (W @ inside) / W.sum(1)
        finite = np.isfinite(width)
        rows.append(dict(dataset=ds, protocol=proto, model=model, method=method, level=lvl, n=len(g),
                         coverage=inside.mean(), cov_lo=np.percentile(cb, 2.5), cov_hi=np.percentile(cb, 97.5),
                         mean_width=width[finite].mean() if finite.any() else np.inf, frac_infinite=1 - finite.mean(),
                         winkler=wink[finite].mean() if finite.any() else np.inf,
                         sys_cov_median=np.nan))
        # per-system coverage (systems with >= 5 test records in a fold / test set)
        per = g.assign(inside=inside).groupby(['label', 'sys_key']).inside.agg(['mean', 'size']).reset_index()
        per = per[per['size'] >= 5]
        if len(per):
            rows[-1].update(sys_cov_median=per['mean'].median(), sys_cov_frac_below_nominal_minus_10=float((per['mean'] < lvl / 100 - 0.1).mean()),
                            n_sys_units=len(per))
            sysrows.append(per.assign(dataset=ds, protocol=proto, model=model, method=method, level=lvl))
    return pd.DataFrame(rows), (pd.concat(sysrows) if sysrows else pd.DataFrame())


if __name__ == '__main__':
    import sys
    perfs, pairs, ranks, uqs, syscov = [], [], [], [], []
    for ds in sys.argv[1].split(','):
        p, pr, bs = accuracy(ds); perfs.append(p); pairs.append(pr)
        ranks.append(ranking(ds, p, bs))
        u, sc = uq(ds); uqs.append(u); syscov.append(sc)
    pd.concat(perfs).to_csv(f"{T}/perf_v2.csv", index=False)
    pd.concat(pairs).to_csv(f"{T}/pairwise_v2.csv", index=False)
    pd.concat(ranks).to_csv(f"{T}/ranking_v2.csv", index=False)
    pd.concat(uqs).to_csv(f"{T}/uq_v2.csv", index=False)
    pd.concat(syscov).to_csv(f"{T}/uq_system_cov_v2.csv", index=False)
    print(pd.concat(perfs).round(3).to_string())

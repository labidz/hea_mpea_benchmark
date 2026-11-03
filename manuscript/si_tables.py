"""Supplementary tables built directly from result CSVs."""
import numpy as np
import pandas as pd
from tables import MODELS, PROTOS, PNAME
from si_static import PHYS_DEFS, HP_SPACES


def df_table(df, caption, widths=None, notes=None, fmt=None):
    fmt = fmt or {}
    rows = []
    for _, r in df.iterrows():
        row = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                f = fmt.get(c, '{:.3g}')
                row.append('–' if np.isnan(v) else f.format(v))
            else:
                row.append(str(v))
        rows.append(row)
    return dict(caption=caption, header=[str(c) for c in df.columns], rows=rows,
                widths=widths or [1.0] * len(df.columns), notes=notes)


def s_features():
    return dict(caption="**Table S1.** Alloy-theory (PHYS) descriptors computed with matminer's WenAlloys featurizer.",
                header=['Descriptor', 'Definition'], rows=[[a, b] for a, b in PHYS_DEFS], widths=[2.2, 5.0],
                notes="*c*_{i}: atomic fraction; *r*_{i}: atomic radius; *T*_{m}: rule-of-mixtures melting temperature. "
                      "The Magpie (MAG) block comprises the minimum, maximum, range, mean, average deviation and mode of 22 "
                      "elemental properties [@ward2016].")


def s_hparams():
    return dict(caption="**Table S2.** Hyper-parameter search spaces (nested random search, three inner folds matched to the outer protocol).",
                header=['Model', 'Search space / settings', 'Configurations sampled'],
                rows=[list(r) for r in HP_SPACES], widths=[1.1, 5.0, 1.4])


def s_perf(perf, ds, n):
    g = perf[perf.dataset == ds]
    rows = []
    for p in PROTOS:
        for m in MODELS:
            s = g[(g.protocol == p) & (g.model == m)]
            if s.empty:
                continue
            r = s.iloc[0]
            sd = lambda k: f" ± {r[k + '_sd']:.2f}" if (k + '_sd') in r and not np.isnan(r[k + '_sd']) else ''
            sdr = lambda k: f" ± {r[k + '_sd']:.0f}" if (k + '_sd') in r and not np.isnan(r[k + '_sd']) else ''
            ci = f" [{r.RMSE_lo:.0f}, {r.RMSE_hi:.0f}]" if 'RMSE_lo' in r and not np.isnan(r.get('RMSE_lo', np.nan)) else ''
            rows.append([PNAME[p], m, f"{r.MAE:.0f}{sdr('MAE')}", f"{r.RMSE:.0f}{sdr('RMSE')}{ci}", f"{r.R2:.2f}{sd('R2')}",
                         f"{r.rho:.2f}{sd('rho')}", f"{r.bias:+.0f}"])
    unit = 'HV' if ds == 'HV' else 'MPa'
    return dict(caption=f"**Table S{n}.** Complete accuracy metrics for {'hardness' if ds == 'HV' else 'yield strength'} ({unit}).",
                header=['Protocol', 'Model', 'MAE', 'RMSE', '*R*^{2}', 'Spearman *ρ*', 'Bias'], rows=rows,
                widths=[1.4, 0.9, 1.0, 1.7, 1.0, 1.0, 0.8],
                notes="Mean ± SD over folds (P1–P3, 15 folds) or held-out elements (P4); temporal and external: single test set "
                      "with bootstrap 95% CI of the RMSE in brackets. Bias = mean(predicted − measured).")


def s_element(folds, ds, n):
    g = folds[(folds.dataset == ds) & (folds.protocol == 'element')].copy()
    g['element'] = g.label.str.replace('element-', '')
    P = g.pivot_table(index='element', columns='model', values='RMSE')
    N = g.groupby('element').n.first()
    cols = [m for m in MODELS if m in P.columns]
    P = P.loc[N.sort_values(ascending=False).index, cols]
    rows = [[f"{e} ({int(N[e])})"] + [f"{P.loc[e, m]:.0f}" for m in cols] for e in P.index]
    return dict(caption=f"**Table S{n}.** Leave-one-element-out RMSE for {'hardness (HV)' if ds == 'HV' else 'yield strength (MPa)'}; "
                        "number of held-out test records in parentheses.",
                header=['Held-out element'] + cols, rows=rows, widths=[1.4] + [0.8] * len(cols))


def s_friedman(fr, n):
    rows = [[r.dataset, PNAME[r.protocol], int(r.n_tasks), f"{r.friedman_chi2:.1f}", f"{r.friedman_p:.1e}", r.best,
             r.mean_ranks.replace('; ', ', ')] for r in fr.itertuples()]
    return dict(caption=f"**Table S{n}.** Friedman tests on fold-wise RMSE and mean ranks (1 = best).",
                header=['Data', 'Protocol', 'Folds', '*χ*^{2}', '*p*', 'Best', 'Mean rank'], rows=rows,
                widths=[0.6, 1.3, 0.6, 0.7, 0.9, 0.7, 4.2])

"""Publication figures from results/tables (+ prediction files)."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import (setup, save, panel_label, SINGLE, DOUBLE, MM, INK, INK2, MUTED, GRID, CAT, PROTO_ORDER,
                      PROTO_LABEL, PROTO_COL, MODEL_ORDER, DIVERGING, SEQ)

RES = "/home/claude/hea/results"
TAB = f"{RES}/tables_v2"   # revision 2
UNITS = {'HV': 'HV', 'YS': 'MPa'}
TITLE = {'HV': 'Vickers hardness', 'YS': 'Yield strength'}


def fig_performance(perf):
    """Dot plot: RMSE per model under each evaluation protocol (ordinal colour = extrapolation severity)."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 72 * MM), sharey=True)
    models = [m for m in MODEL_ORDER if m in set(perf.model) and m != 'Mean']
    for k, (ax, ds) in enumerate(zip(axes, ['HV', 'YS'])):
        g = perf[perf.dataset == ds]
        mr = g[(g.model == 'Mean') & (g.protocol == 'random')]
        if len(mr):
            ax.axvline(mr.RMSE.iloc[0], color=INK2, ls='--', lw=0.8, zorder=1)
            ax.text(mr.RMSE.iloc[0], -0.62, ' mean predictor', fontsize=5.8, color=INK2, va='bottom', ha='right')
        protos = [p for p in PROTO_ORDER if p in set(g.protocol)]
        offs = np.linspace(-0.27, 0.27, len(protos))
        for j, pr in enumerate(protos):
            gp = g[g.protocol == pr].set_index('model')
            for i, m in enumerate(models):
                if m not in gp.index:
                    continue
                r = gp.loc[m]
                x = r.RMSE
                if not np.isnan(r.get('RMSE_lo', np.nan)):   # cluster-bootstrap 95% CI (alloy systems resampled)
                    ax.errorbar(x, i + offs[j], xerr=[[x - r.RMSE_lo], [r.RMSE_hi - x]], fmt='none', ecolor=PROTO_COL[pr],
                                elinewidth=0.8, capsize=0)
                ax.plot(x, i + offs[j], 'o', ms=4.2, mfc=PROTO_COL[pr], mec='white', mew=0.5, zorder=3)
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_xscale('log')
        from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
        tk = [80, 100, 150, 200, 300, 400] if ds == 'HV' else [250, 300, 400, 500, 600]
        ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.set_xlabel(f"RMSE ({UNITS[ds]}, log scale)")
        ax.set_title(TITLE[ds], loc='left', fontweight='bold')
        for i in range(len(models)):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color='#f6f5f2', zorder=0, lw=0)
        ax.grid(axis='y', visible=False)
        panel_label(ax, 'ab'[k], x=-0.12 if k == 0 else -0.04)
    handles = [Line2D([], [], marker='o', ls='none', mfc=PROTO_COL[p], mec='white', ms=5, label=PROTO_LABEL[p])
               for p in PROTO_ORDER if p in set(perf.protocol)]
    fig.legend(handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.06), handletextpad=0.2,
               columnspacing=1.0)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'Fig2_performance_protocols')


def fig_element(el):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 80 * MM), gridspec_kw=dict(width_ratios=[11, 16], wspace=0.08))
    for k, (ax, ds) in enumerate(zip(axes, ['HV', 'YS'])):
        g = el[(el.dataset == ds) & (el.model != 'Mean')].copy()
        g['element'] = g.label.str.replace('element-', '')
        models = [m for m in MODEL_ORDER if m in set(g.model)]
        cols = g.groupby('element').n.first().sort_values(ascending=False).index.tolist()
        P = g.pivot_table(index='model', columns='element', values='R2').reindex(models)[cols]
        im = ax.imshow(P.values.clip(-1, 1), cmap=DIVERGING, vmin=-1, vmax=1, aspect='auto')
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                v = P.values[i, j]
                ax.text(j, i, f"{v:.2f}".replace('-', '−'), ha='center', va='center', fontsize=4.6,
                        color='white' if (v > 0.6 or v < -0.6) else INK)
        ns = g.groupby('element').n.first()
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{c}\n{ns[c]}" for c in cols], fontsize=5.6)
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models if k == 0 else [])
        ax.tick_params(length=0)
        ax.grid(False)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.set_title('Hardness' if ds == 'HV' else 'Yield strength', loc='left', fontweight='bold')
        ax.set_xlabel('Held-out element (number of test records)', fontsize=6.2)
        panel_label(ax, 'ab'[k], x=-0.12 if k == 0 else -0.03, y=1.03)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label('$R^2$ (colour clipped to [−1, 1])', fontsize=6)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5)
    save(fig, 'Fig3_leave_one_element_out')


# ---------------------------------------------------------------- parity / external
def fig_parity(datasets_preds):
    """datasets_preds: list of (title, df with y,yhat,ad_knn,(lo,hi)), model name)."""
    n = len(datasets_preds)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE, 66 * MM))
    for k, (ax, (title, d, model, unit)) in enumerate(zip(axes, datasets_preds)):
        lo_, hi_ = min(d.y.min(), d.yhat.min()), max(d.y.max(), d.yhat.max())
        pad = 0.05 * (hi_ - lo_); lim = (lo_ - pad, hi_ + pad)
        if 'lo' in d and d['lo'].notna().all():
            ax.vlines(d.y, d['lo'], d['hi'], color='#c9c8c3', lw=0.5, zorder=1)
        sc = ax.scatter(d.y, d.yhat, c=d.ad_knn, cmap=SEQ, s=9, edgecolor='white', linewidth=0.25, zorder=3,
                        vmin=np.percentile(d.ad_knn, 2), vmax=np.percentile(d.ad_knn, 98))
        ax.plot(lim, lim, color=INK2, lw=0.8, ls='--', zorder=2)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        e = d.yhat - d.y; r2 = 1 - (e ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()
        txt = f"{model}\nRMSE {np.sqrt((e**2).mean()):.0f} {unit}, $R^2$ {r2:.2f}\nbias {e.mean():+.0f} {unit}"
        if 'lo' in d and d['lo'].notna().all():
            cov = ((d.y >= d['lo']) & (d.y <= d['hi'])).mean()
            txt += f"\n90% CV+ coverage {cov:.2f}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left', fontsize=5.6,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85), zorder=5)
        ax.set_xlabel(f"Measured ({unit})"); ax.set_ylabel(f"Predicted ({unit})")
        ax.set_title(title, loc='left', fontweight='bold')
        panel_label(ax, 'abc'[k], x=-0.27)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label('AD distance $d_5$', fontsize=6); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'Fig4_parity_temporal_external')


# ---------------------------------------------------------------- uncertainty
UQ_METHODS = [('XGB', 'CV+naive', 'XGB, CV+ (random folds)', 'o'), ('XGB', 'CV+matched', 'XGB, CV+ (deployment-matched)', 's'),
              ('XGB', 'nCV+matched', 'XGB, normalized CV+ (matched)', 'D'), ('XGB', 'gwCV+matched', 'XGB, group-weighted CV+ (matched)', 'X'),
              ('GPR', 'GP-native', 'GPR, native', '^'),
              ('TabPFN', 'TabPFN-native', 'TabPFN, native quantiles', 'v'), ('TabPFN', 'CV+matched', 'TabPFN, CV+ (matched)', 'P')]


def fig_uq(uq, nominal=90):
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 130 * MM))
    for r, ds in enumerate(['HV', 'YS']):
        g = uq[(uq.dataset == ds)]
        protos = [p for p in PROTO_ORDER if p in set(g.protocol)]
        x = np.arange(len(protos)); offs = np.linspace(-0.3, 0.3, len(UQ_METHODS))
        for j, (model, meth, lab, mk) in enumerate(UQ_METHODS):
            cov, wid, clo, chi = [], [], [], []
            for p in protos:
                s = g[(g.protocol == p) & (g.model == model) & (g.method == f"{meth}{nominal}")]
                if s.empty and meth.endswith('matched') and p == 'random':
                    s = g[(g.protocol == p) & (g.model == model) & (g.method == f"{meth.replace('matched', 'naive')}{nominal}")]
                cov.append(s.coverage.iloc[0] if len(s) else np.nan); wid.append(s.mean_width.iloc[0] if len(s) else np.nan)
                clo.append(s.cov_lo.iloc[0] if len(s) and 'cov_lo' in s and not pd.isna(s.cov_lo.iloc[0]) else np.nan)
                chi.append(s.cov_hi.iloc[0] if len(s) and 'cov_hi' in s and not pd.isna(s.cov_hi.iloc[0]) else np.nan)
            sd = SD[ds]
            cov, clo, chi = map(np.array, (cov, clo, chi)); ok = ~np.isnan(clo)
            if ok.any():
                axes[r, 0].errorbar((x + offs[j])[ok], cov[ok], yerr=[cov[ok] - clo[ok], chi[ok] - cov[ok]], fmt='none',
                                    ecolor=CAT[j], elinewidth=0.7, capsize=0, alpha=0.8)
            axes[r, 0].plot(x + offs[j], cov, mk, ms=4, color=CAT[j], mec='white', mew=0.4, label=lab, ls='none')
            axes[r, 1].plot(x + offs[j], np.array(wid) / sd, mk, ms=4, color=CAT[j], mec='white', mew=0.4, ls='none')
        axes[r, 0].axhline(nominal / 100, color=INK2, lw=0.8, ls='--')
        axes[r, 0].text(len(protos) - 0.55, nominal / 100 + 0.01, f'nominal {nominal}%', fontsize=5.8, color=INK2, ha='right', va='bottom')
        axes[r, 0].set_ylabel(f"{TITLE[ds]}\nempirical coverage")
        axes[r, 1].set_ylabel("mean interval width / SD(y)")
        axes[r, 0].set_ylim(0.2, 1.02)
        for c in range(2):
            axes[r, c].set_xticks(x); axes[r, c].set_xticklabels([PROTO_LABEL[p].replace(' (', '\n(').replace('-grouped', '-\ngrouped').replace('Leave-one-element-out', 'Leave-one-\nelement-out') for p in protos], fontsize=5.8)
            axes[r, c].grid(axis='x', visible=False)
            panel_label(axes[r, c], 'abcd'[2 * r + c], x=-0.17)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(h_pad=1.2, w_pad=1.5, rect=(0, 0.1, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), handletextpad=0.2, columnspacing=1.0)
    save(fig, f'Fig5_uncertainty_coverage{nominal}')


SD = {}


def fig_ad(byad, adc, preds):
    """(a) |error| vs AD-distance decile; (b) conditional coverage by AD tertile; (c) error-retention curves."""
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 62 * MM))
    ax = axes[0]
    for k, ds in enumerate(['HV', 'YS']):
        d = preds[ds]
        d = d[(d.model == 'XGB') & (d.protocol != 'random')]
        q = pd.qcut(d.ad_knn, 10, labels=False, duplicates='drop')
        ae = (d.yhat - d.y).abs() / SD[ds]
        med = ae.groupby(q).median(); q1 = ae.groupby(q).quantile(0.25); q3 = ae.groupby(q).quantile(0.75)
        xm = d.ad_knn.groupby(q).median()
        ax.fill_between(xm, q1, q3, color=CAT[k], alpha=0.18, lw=0)
        ax.plot(xm, med, '-o', ms=3, color=CAT[k], label=f"{TITLE[ds]} (ρ = {adc[ds]:.2f})")
    ax.set_xlabel('AD distance $d_5$ (decile medians)'); ax.set_ylabel('|error| / SD(y)')
    ax.legend(loc='upper left', fontsize=6)
    panel_label(ax, 'a', x=-0.2)
    ax = axes[1]
    meths = [('CV+naive90', 'CV+ (random folds)', 'o'), ('CV+matched90', 'CV+ (matched)', 's'), ('nCV+matched90', 'normalized CV+ (matched)', 'D')]
    for k, ds in enumerate(['HV', 'YS']):
        for j, (m, lab, mk) in enumerate(meths):
            s = byad[(byad.dataset == ds) & (byad.model == 'XGB') & (byad.method == m) & (byad.protocol != 'random')]
            c = s.groupby('ad_tertile').apply(lambda t: np.average(t.coverage, weights=t.n))
            ax.plot(np.arange(3) + (k - 0.5) * 0.18, c.values, marker=mk, ls='-' if k == 0 else ':', color=CAT[j], ms=3.5,
                    mec='white', mew=0.3, label=f"{lab}" if k == 0 else None)
    ax.axhline(0.9, color=INK2, lw=0.8, ls='--')
    ax.set_xticks(range(3)); ax.set_xticklabels(['near', 'middle', 'far'])
    ax.set_xlabel('AD tertile (test records)'); ax.set_ylabel('empirical coverage (nominal 90%)')
    from matplotlib.lines import Line2D as _L
    h, l = ax.get_legend_handles_labels()
    h += [_L([], [], color=INK2, ls='-', lw=1), _L([], [], color=INK2, ls=':', lw=1)]; l += ['hardness', 'yield strength']
    ax.legend(h, l, loc='lower left', fontsize=5.4, ncol=1, handlelength=2.2)
    ax.set_ylim(0.6, 0.97)
    ax.grid(axis='x', visible=False)
    panel_label(ax, 'b', x=-0.2)
    ax = axes[2]
    fr = np.linspace(0.1, 1.0, 19)
    for k, ds in enumerate(['HV', 'YS']):
        d = preds[ds]; d = d[(d.model == 'XGB') & (d.protocol.isin(['system', 'element', 'temporal']))]
        e2 = ((d.yhat - d.y) / SD[ds]) ** 2
        for score, ls, lab in [('ad_knn', '-', 'AD distance'), ('width', '--', 'normalized CV+ width'), ('oracle', ':', 'oracle |error|')]:
            if score == 'width':
                s = d.ncp_matched_hw90
            elif score == 'oracle':
                s = e2
            else:
                s = d.ad_knn
            order = np.argsort(s.values)
            rm = [np.sqrt(e2.values[order[:max(1, int(f * len(order)))]].mean()) for f in fr]
            ax.plot(fr, rm, ls=ls, color=CAT[k], lw=1.1)
    ax.set_xlabel('fraction of predictions retained'); ax.set_ylabel('RMSE / SD(y) of retained')
    from matplotlib.lines import Line2D as _L
    hs = [_L([], [], color=CAT[0], lw=1.2), _L([], [], color=CAT[1], lw=1.2),
          _L([], [], color=INK2, ls='-', lw=1), _L([], [], color=INK2, ls='--', lw=1), _L([], [], color=INK2, ls=':', lw=1)]
    ax.legend(hs, ['hardness', 'yield strength', 'discard by AD distance', 'discard by interval width', 'oracle (true error)'],
              loc='center left', bbox_to_anchor=(0.02, 0.42), fontsize=5.3, handlelength=2.2)
    panel_label(ax, 'c', x=-0.2)
    fig.tight_layout(w_pad=1.3)
    save(fig, 'Fig6_applicability_domain')


# ---------------------------------------------------------------- SHAP
PRETTY = {'PHYS:VEC mean': 'VEC', 'PHYS:Yang delta': 'δ (size mismatch)', 'PHYS:Mixing enthalpy': 'ΔH$_{mix}$',
          'PHYS:Yang omega': 'Ω', 'PHYS:Configuration entropy': 'ΔS$_{conf}$', 'PHYS:Electronegativity delta': 'Δχ',
          'PHYS:Shear modulus mean': 'mean G', 'PHYS:Shear modulus delta': 'δG', 'PHYS:Shear modulus strength model': 'G-strength model',
          'PHYS:Mean cohesive energy': 'mean E$_{coh}$', 'PHYS:Atomic weight mean': 'mean atomic weight', 'PHYS:APE mean': 'APE',
          'PHYS:Radii local mismatch': 'radius local mismatch', 'PHYS:Radii gamma': 'γ (radii)', 'PHYS:Lambda entropy': 'Λ',
          'TEST:T_C': 'test temperature', 'TEST:tension': 'tension test', 'PHYS:Interant d electrons': 'interant d e$^-$',
          'PHYS:Electronegativity local mismatch': 'χ local mismatch', 'PHYS:Shear modulus local mismatch': 'G local mismatch'}


def pretty(f):
    if f in PRETTY:
        return PRETTY[f]
    b, n = f.split(':', 1)
    if b == 'FRAC':
        return f"x$_{{{n}}}$"
    if b == 'PROC':
        return f"process: {n.lower()}"
    if b == 'MAG':
        return n.replace('mean ', 'mean ').replace('avg_dev ', 'dev ').replace('NdValence', 'd-valence').replace('MeltingT', 'T$_m$').replace('CovalentRadius', 'cov. radius').replace('GSvolume_pa', 'V$_{GS}$').replace('NValence', 'valence e$^-$').replace('Electronegativity', 'χ')
    return n


def fig_shap(top=12):
    import json
    fig = plt.figure(figsize=(DOUBLE, 112 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.55, wspace=0.55)
    for r, ds in enumerate(['HV', 'YS']):
        st = pd.read_csv(f"{TAB}/shap_stability_{ds}.csv").head(top)
        js = json.load(open(f"{TAB}/shap_stability_{ds}.json"))
        ax = fig.add_subplot(gs[r, 0])
        y = np.arange(len(st))
        ax.barh(y, st.mean_abs_shap_full, color=CAT[r], height=0.62, edgecolor='white', linewidth=0.3)
        ax.errorbar(st.fold_mean, y, xerr=st.fold_sd, fmt='|', color=INK2, ms=4, elinewidth=0.7, capsize=0)
        xmax = max(st.mean_abs_shap_full.max(), (st.fold_mean + st.fold_sd).max()) * 1.08
        for i, fq in enumerate(st.top10_frequency):
            ax.text(xmax * 1.02, i, f"{fq:.0%}", va='center', ha='left', fontsize=5.6, color=INK2)
        ax.text(xmax * 1.02, -1.0, 'top-10\nfreq.', va='bottom', ha='left', fontsize=5.2, color=INK2)
        ax.set_xlim(0, xmax)
        ax.set_yticks(y); ax.set_yticklabels([pretty(f) for f in st.feature], fontsize=6)
        ax.set_ylim(len(st) - 0.5, -0.5)
        ax.set_xlabel(f"mean |SHAP| ({UNITS[ds]})")
        ax.set_title(f"{'Hardness' if ds == 'HV' else 'Yield strength'}  (fold stability ρ̄ = {js['pairwise_spearman_mean']:.2f})", loc='left', fontweight='bold', fontsize=7)
        ax.grid(axis='y', visible=False)
        panel_label(ax, 'ac'[r], x=-0.45)
        # dependence plot of the top non-trivial physical feature
        sv = np.load(f"{RES}/shap_values_{ds}.npy"); X = pd.read_csv(f"{RES}/shap_X_{ds}.csv")
        imp = pd.read_csv(f"{TAB}/shap_importance_{ds}.csv", index_col=0)
        feat = 'TEST:T_C' if ds == 'YS' else 'PHYS:VEC mean'
        col2 = 'PHYS:VEC mean' if ds == 'YS' else 'PHYS:Radii local mismatch'
        j = list(X.columns).index(feat)
        ax = fig.add_subplot(gs[r, 1])
        sc = ax.scatter(X[feat], sv[:, j], c=X[col2] if col2 in X else None, cmap=SEQ, s=6, lw=0)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xlabel(pretty(feat) + (' (°C)' if feat == 'TEST:T_C' else ''))
        ax.set_ylabel(f"SHAP value ({UNITS[ds]})")
        if col2 in X:
            cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.set_label(pretty(col2), fontsize=6)
            cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5)
        panel_label(ax, 'bd'[r], x=-0.25)
    save(fig, 'Fig7_shap')


def _preds():
    from analysis_v2 import load_all
    out = {}
    for ds in ('HV', 'YS'):
        try:
            out[ds] = load_all(ds)
        except FileNotFoundError:
            pass
    return out


if __name__ == '__main__':
    setup()
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    perf = pd.read_csv(f"{TAB}/perf_summary.csv")
    for ds, fn in (('HV', 'hv_records.csv'), ('YS', 'ys_records.csv')):
        SD[ds] = pd.read_csv(f"/home/claude/hea/data/{fn}").target.std()
    if which in ('all', 'perf'):
        fig_performance(perf)
    if which in ('all', 'element'):
        fig_element(pd.read_csv(f"{TAB}/element_tasks.csv"))
    if which in ('all', 'uq'):
        fig_uq(pd.read_csv(f"{TAB}/uq_summary.csv"), 90)
        fig_uq(pd.read_csv(f"{TAB}/uq_summary.csv"), 68)
    if which in ('all', 'ad', 'parity'):
        P = _preds()
    if which in ('all', 'ad'):
        adc = pd.read_csv(f"{TAB}/ad_corr.csv")
        rho = {}
        for ds in P:
            d = P[ds]; d = d[(d.model == 'XGB') & (d.protocol != 'random')]
            from scipy.stats import spearmanr
            rho[ds] = spearmanr(d.ad_knn, (d.yhat - d.y).abs()).correlation
        fig_ad(pd.read_csv(f"{TAB}/uq_by_ad.csv"), rho, P)
    if which in ('all', 'parity'):
        panels = []
        for ds, lab, title in (('HV', 'temporal-2019', 'Hardness, 2019 literature'), ('HV', 'external-GE', 'Hardness, external GE alloys'),
                               ('YS', 'temporal-2019', 'Yield strength, 2019 literature')):
            d = P[ds]; d = d[d.label == lab]
            best = 'XGBoost'
            g = d[d.model == 'XGB'].copy()
            g['lo'], g['hi'] = g.get('cvp_matched_lo90'), g.get('cvp_matched_hi90')
            if g['ad_knn'].isna().any():
                adm = d[d.model == 'XGB'].set_index('rid').ad_knn
                g['ad_knn'] = g.rid.map(adm)
            panels.append((title, g, best, UNITS[ds]))
        fig_parity(panels)
    if which in ('all', 'shap'):
        fig_shap()

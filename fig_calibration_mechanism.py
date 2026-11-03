"""Fig. 8: why protocol-matched calibration works. (a) schematic of calibration folds; (b, c) the interval
half-width needed for 90% coverage (90th percentile of |test error|) versus the half-widths supplied by naive,
protocol-matched and distance-normalized CV+ (XGBoost)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from analysis_v2 import load_all
from figstyle import setup, save, panel_label, MM, INK, INK2, MUTED, CAT, PROTO_ORDER, PROTO_COL

R = "/home/claude/hea/results"; T = f"{R}/tables_v2"
SHORT = {'random': 'P1', 'composition': 'P2', 'system': 'P3', 'element': 'P4', 'temporal': 'P5', 'external': 'P6'}


def needed_vs_supplied(ds):
    P = load_all(ds)   # same merge as the reported analysis (winsorized normalized CV+ for XGBoost)
    P = P[P.model == 'XGB'].copy()
    rows = []
    for p in PROTO_ORDER:
        g = P[P.protocol == p]
        if g.empty:
            continue
        err = (g.y - g.yhat).abs()
        rows.append(dict(protocol=p, needed=np.quantile(err, 0.9),
                         naive=((g.cvp_naive_hi90 - g.cvp_naive_lo90) / 2).mean(),
                         matched=((g.cvp_matched_hi90 - g.cvp_matched_lo90) / 2).mean() if p != 'random' else np.nan,
                         normalized=g.ncp_matched_hw90.mean() if p != 'random' else np.nan,
                         cov_naive=((g.y >= g.cvp_naive_lo90) & (g.y <= g.cvp_naive_hi90)).mean(),
                         cov_matched=((g.y >= g.cvp_matched_lo90) & (g.y <= g.cvp_matched_hi90)).mean()))
    return pd.DataFrame(rows)


def schematic(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    rng = np.random.default_rng(3)
    centers = [(13, 0), (31, 2.5), (49, -2), (67, 2), (85, -1)]
    pts = [(cx + rng.normal(0, 2.4, 7), cy + rng.normal(0, 1.8, 7)) for cx, cy in centers]
    for row, (y0, title, sub) in enumerate([(74, 'Naive CV+: random calibration folds',
                                             'held-out alloys have near-copies in the remaining\ndata → small residuals → narrow intervals'),
                                            (24, 'Matched CV+ (P3): whole systems held out',
                                             'held-out system has no relatives in the remaining\ndata → residuals of a new system → wider intervals')]):
        ax.add_patch(FancyBboxPatch((1, y0 - 22), 98, 44, boxstyle="round,pad=0,rounding_size=2", fc='#f4f6f9', ec='#c9d3e0', lw=0.6))
        ax.text(4, y0 + 17.5, title, fontsize=6.4, fontweight='bold', va='center')
        held = []
        for k, (xs, ys) in enumerate(pts):
            for i, (x, y) in enumerate(zip(xs, ys)):
                h = (rng.random() < 0.22) if row == 0 else (k == 2)
                held.append(h)
                ax.plot(x, y0 + 5 + y, 'o', ms=3.4, mfc=CAT[1] if h else 'white', mec=CAT[1] if h else MUTED, mew=0.7)
            ax.text(cx_ := centers[k][0], y0 - 5.5, f"system {chr(65 + k)}", fontsize=5.2, color=INK2, ha='center')
        ax.text(4, y0 - 14.5, sub, fontsize=6.0, color=INK2, va='center', linespacing=1.25)
    ax.plot([], [], 'o', ms=3.4, mfc=CAT[1], mec=CAT[1], label='calibration (held-out) alloy')
    ax.plot([], [], 'o', ms=3.4, mfc='white', mec=MUTED, label='alloy used for fitting')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=6.0, handletextpad=0.2, columnspacing=1.0)


def main():
    setup()
    fig = plt.figure(figsize=(190 * MM, 84 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], left=0.005, right=0.99, top=0.9, bottom=0.2, wspace=0.3)
    ax = fig.add_subplot(gs[0]); schematic(ax); panel_label(ax, 'a', x=0.0, y=1.0)
    out = []
    for k, ds in enumerate(('HV', 'YS')):
        d = needed_vs_supplied(ds); d['dataset'] = ds; out.append(d)
        ax = fig.add_subplot(gs[k + 1])
        x = np.arange(len(d))
        ax.bar(x, d.needed, 0.62, color=[PROTO_COL[p] for p in d.protocol], label='needed for 90% coverage', zorder=1)
        ax.plot(x, d.naive, 'o-', color=CAT[1], ms=3.6, lw=1.1, label='naive CV+', zorder=3)
        ax.plot(x, d.matched, 's-', color=INK, ms=3.4, lw=1.1, label='matched CV+', zorder=3)
        ax.plot(x, d.normalized, '^--', color=CAT[2], ms=3.6, lw=1.0, label='normalized CV+', zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[p] for p in d.protocol], fontsize=6.5); ax.set_xlabel('Protocol (increasing extrapolation →)')
        unit = 'HV' if ds == 'HV' else 'MPa'
        ax.set_ylabel(f"90% interval half-width ({unit})")
        ax.set_title('Hardness (XGBoost)' if ds == 'HV' else 'Yield strength (XGBoost)', loc='left')
        ax.grid(axis='x', visible=False)
        ax.set_ylim(0, max(d.needed.max(), np.nanmax(d.normalized.values)) * 1.15)
        panel_label(ax, 'b' if k == 0 else 'c', x=-0.2)
        if k == 0:
            h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.66, 0.0), fontsize=6, columnspacing=1.2)
    pd.concat(out).to_csv(f"{T}/needed_vs_supplied_halfwidth.csv", index=False)
    print(pd.concat(out).round(3).to_string())
    save(fig, 'Fig0c_calibration_mechanism')


if __name__ == '__main__':
    main()

"""Graphical abstract (>= 1328 px wide, readable at 13 cm)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from figstyle import setup, MM, INK, INK2, CAT, PROTO_ORDER, PROTO_COL

T = "/home/claude/hea/results/tables_v2"
LAB = {'random': 'Random', 'composition': 'New composition', 'system': 'New alloy system', 'element': 'New element',
       'temporal': 'Next year of literature', 'external': 'Independent laboratory'}


def main():
    setup()
    perf = pd.read_csv(f"{T}/perf_summary.csv"); uq = pd.read_csv(f"{T}/uq_summary.csv")
    sd = pd.read_csv("/home/claude/hea/data/hv_records.csv").target.std()
    fig = plt.figure(figsize=(160 * MM, 62 * MM))
    fig.text(0.01, 0.97, 'Machine learning for high-entropy alloy hardness: interpolation ≠ discovery', fontsize=9,
             fontweight='bold', va='top')
    ax = fig.add_axes([0.20, 0.14, 0.33, 0.68])
    g = perf[(perf.dataset == 'HV') & (perf.model == 'XGB')].set_index('protocol')
    protos = [p for p in PROTO_ORDER if p in g.index]
    y = np.arange(len(protos))
    vals = [g.loc[p, 'RMSE'] for p in protos]
    ax.barh(y, vals, color=[PROTO_COL[p] for p in protos], height=0.62, edgecolor='white')
    for i, v in enumerate(vals):
        ax.text(v + 4, i, f"{v:.0f} HV", va='center', fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([LAB[p] for p in protos], fontsize=6.8)
    ax.set_ylim(len(protos) - 0.5, -0.5); ax.set_xlim(0, max(vals) * 1.3)
    ax.set_xlabel('Prediction error (RMSE, XGBoost)', fontsize=6.8)
    ax.grid(axis='y', visible=False)
    ax.set_title('Error grows with the novelty of the alloy', fontsize=7.2, loc='left')
    ax2 = fig.add_axes([0.66, 0.14, 0.31, 0.68])
    for j, (meth, lab, mk, c) in enumerate([('CV+naive90', 'random-fold calibration', 'o', CAT[1]),
                                            ('CV+matched90', 'protocol-matched calibration', 's', CAT[0])]):
        cov = []
        for p in protos:
            s = uq[(uq.dataset == 'HV') & (uq.model == 'XGB') & (uq.protocol == p) & (uq.method == meth)]
            if s.empty:
                s = uq[(uq.dataset == 'HV') & (uq.model == 'XGB') & (uq.protocol == p) & (uq.method == 'CV+naive90')]
            cov.append(s.coverage.iloc[0])
        ax2.plot(cov, y + (j - 0.5) * 0.22, mk, color=c, ms=5, mec='white', mew=0.5, label=lab, ls='none')
    ax2.axvline(0.9, color=INK2, ls='--', lw=0.8)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_ylim(len(protos) - 0.5, -0.5); ax2.set_xlim(0.3, 1.0)
    ax2.set_xlabel('Coverage of 90% conformal intervals', fontsize=6.8)
    ax2.set_title('Matched calibration helps, but is not a guarantee', fontsize=7.2, loc='left')
    ax2.legend(loc='upper left', fontsize=6, handletextpad=0.2)
    ax2.grid(axis='y', visible=False)
    fig.savefig('/home/claude/hea/figures/Graphical_abstract.png', dpi=400)
    fig.savefig('/home/claude/hea/figures/Graphical_abstract.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()

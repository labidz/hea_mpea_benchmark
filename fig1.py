"""Figure 1: data overview and evaluation protocols."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from figstyle import setup, save, panel_label, DOUBLE, MM, INK, INK2, MUTED, CAT, PROTO_COL, PROTO_LABEL

D = "/home/claude/hea/data"


def main():
    setup()
    hv = pd.read_csv(f"{D}/hv_records.csv"); ys = pd.read_csv(f"{D}/ys_records.csv"); ge = pd.read_csv(f"{D}/ext_ge_records.csv")
    sp = json.load(open(f"{D}/splits.json"))
    fig = plt.figure(figsize=(DOUBLE, 120 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], width_ratios=[1, 1.15, 1.0], hspace=0.62, wspace=0.42)

    # (a) records per publication year
    ax = fig.add_subplot(gs[0, 0])
    yrs = np.arange(2004, 2020)
    a = hv.year.value_counts().reindex(yrs, fill_value=0); b = ys.year.value_counts().reindex(yrs, fill_value=0)
    w = 0.4
    ax.bar(yrs - w / 2, a.values, w, color=CAT[0], label=f'Hardness (n={len(hv)})', edgecolor='white', linewidth=0.3)
    ax.bar(yrs + w / 2, b.values, w, color=CAT[1], label=f'Yield strength (n={len(ys)})', edgecolor='white', linewidth=0.3)
    ax.axvline(2018.5, color=INK2, lw=0.8, ls='--')
    ax.text(2018.2, 230, 'temporal split', ha='right', va='center', fontsize=5.8, color=INK2, rotation=90)
    ax.set_xlabel('Publication year'); ax.set_ylabel('Records')
    ax.set_xticks([2004, 2008, 2012, 2016, 2019])
    ax.set_ylim(0, 560)
    ax.legend(loc='upper left', fontsize=6, borderaxespad=0.1)
    ax.grid(axis='x', visible=False)
    panel_label(ax, 'a', x=-0.3)

    # (b) element coverage (fraction of records containing element)
    ax = fig.add_subplot(gs[0, 1])
    def frac(df):
        return pd.Series([e for s in df.sys_key for e in s.split('-')]).value_counts() / len(df)
    fh, fy, fg = frac(hv), frac(ys), frac(ge)
    tot = fh.add(fy, fill_value=0).add(fg * 0.001, fill_value=0).sort_values(ascending=False)
    els = [e for e in tot.index if (fh.get(e, 0) >= 0.02 or fy.get(e, 0) >= 0.02 or fg.get(e, 0) > 0)]
    x = np.arange(len(els))
    ax.plot(x, [fh.get(e, 0) for e in els], 'o', ms=3.2, color=CAT[0], label='Hardness (MPEA)')
    ax.plot(x, [fy.get(e, 0) for e in els], 's', ms=3.0, color=CAT[1], label='Yield strength (MPEA)')
    ax.plot(x, [fg.get(e, 0) for e in els], 'D', ms=3.0, mfc='none', mec=CAT[2], mew=0.9, label='External GE set')
    ax.set_xticks(x); ax.set_xticklabels(els, fontsize=5.6, rotation=90)
    ax.set_ylabel('Fraction of records\ncontaining element')
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc='upper right', fontsize=6, handletextpad=0.1, borderaxespad=0.1)
    ax.grid(axis='x', visible=False)
    panel_label(ax, 'b', x=-0.2)

    # (c) target distributions
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, 1350, 28)
    ax.hist(hv.target, bins=bins, color=CAT[0], alpha=0.85, label='MPEA (n=501)', edgecolor='white', linewidth=0.3)
    ax.hist(ge.target, bins=bins, histtype='step', color=CAT[2], lw=1.2, label='External GE (n=86)')
    ax.set_xlabel('Vickers hardness (HV)'); ax.set_ylabel('Records')
    ax.set_ylim(0, 78)
    ax.legend(loc='upper right', fontsize=6, borderaxespad=0.1)
    ax.grid(axis='x', visible=False)
    panel_label(ax, 'c', x=-0.28)

    # (d) protocol schematic on the real hardness data: records ordered by alloy system
    ax = fig.add_subplot(gs[1, :])
    order = hv.sort_values(['sys_key', 'comp_key', 'year']).rid.values
    pos = {r: i for i, r in enumerate(order)}
    rows = [('random', 'random-r0f0'), ('composition', 'composition-r0f0'), ('system', 'system-r0f0'),
            ('element', 'element-Ti'), ('temporal', 'temporal-2019')]
    lab = {s['label']: set(s['test']) for s in sp['HV']}
    M = np.zeros((len(rows), len(order)))
    for i, (pr, l) in enumerate(rows):
        for r in lab[l]:
            M[i, pos[r]] = 1
    for i, (pr, l) in enumerate(rows):
        cmap = ListedColormap(['#ecebe7', PROTO_COL[pr] if pr != 'random' else '#86b4ea'])
        ax.imshow(M[i:i + 1], aspect='auto', cmap=cmap, extent=(0, len(order), i + 0.42, i - 0.42),
                  interpolation='nearest', vmin=0, vmax=1)
    # system boundaries (alternating ticks along the bottom)
    sysk = hv.set_index('rid').loc[order].sys_key.values
    bounds = [0] + [i for i in range(1, len(order)) if sysk[i] != sysk[i - 1]] + [len(order)]
    for k in range(len(bounds) - 1):
        ax.plot([bounds[k], bounds[k + 1]], [len(rows) - 0.35] * 2, color=INK2 if k % 2 == 0 else MUTED,
                lw=2.2, solid_capstyle='butt')
    ax.set_ylim(len(rows) - 0.1, -0.6); ax.set_xlim(0, len(order))
    names = {'random': 'Random K-fold', 'composition': 'Composition-grouped', 'system': 'System-grouped',
             'element': 'Leave-one-element-out (Ti)', 'temporal': 'Temporal (train ≤2018, test 2019)'}
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([names[p] for p, _ in rows])
    ax.set_xlabel(f'{len(order)} hardness records, ordered by alloy system (bar segments = {len(bounds) - 1} systems) and composition')
    ax.set_xticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(len(order), -0.75, 'coloured = test records of one split; grey = training', ha='right', va='bottom',
            fontsize=6, color=INK2)
    panel_label(ax, 'd', x=-0.165, y=1.0)
    save(fig, 'Fig1_data_protocols')


if __name__ == '__main__':
    main()

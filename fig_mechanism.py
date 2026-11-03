"""Fig. 3: why random splits are optimistic - near-duplicates in the MPEA data."""
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import setup, save, panel_label, MM, INK, INK2, MUTED, CAT, PROTO_ORDER, PROTO_COL, PROTO_LABEL

D = "/home/claude/hea/data"; T = "/home/claude/hea/results/tables"
SYS = 'Al-Co-Cr-Fe-Ni'


def al_frac(k):
    return float(re.search(r'Al([0-9.]+)', k).group(1)) * 100


def main():
    setup()
    hv = pd.read_csv(f"{D}/hv_records.csv"); ys = pd.read_csv(f"{D}/ys_records.csv")
    sp = json.load(open(f"{D}/splits.json"))
    nn = pd.read_csv(f"{T}/nn_comp_distance.csv")
    su = pd.read_csv(f"{T}/random_seen_unseen.csv")
    perf = pd.read_csv(f"{T}/perf_summary.csv")

    fig = plt.figure(figsize=(190 * MM, 112 * MM))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, top=0.95, bottom=0.14, wspace=0.28, hspace=0.55)

    # (a) one alloy family under a random split and a system-grouped split
    ax = fig.add_subplot(gs[0, 0])
    g = hv[hv.sys_key == SYS].copy(); g['al'] = g.comp_key.map(al_frac)
    test = set(next(s for s in sp['HV'] if s['label'] == 'random-r0f0')['test'])
    tr, te = g[~g.rid.isin(test)], g[g.rid.isin(test)]
    jit = np.random.default_rng(0).uniform(-0.25, 0.25, len(g))
    j = dict(zip(g.rid, jit))
    ax.scatter(tr.al + tr.rid.map(j), tr.target, s=16, facecolor='white', edgecolor=MUTED, lw=0.8, label='training (random fold)', zorder=2)
    ax.scatter(te.al + te.rid.map(j), te.target, s=18, color=CAT[0], edgecolor='white', lw=0.4, label='test (random fold)', zorder=3)
    d_te = nn[(nn.label == 'random-r0f0') & nn.rid.isin(te.rid)].dnn
    ax.set_xlabel('Al content (at.%)'); ax.set_ylabel('Vickers hardness (HV)')
    ax.set_title(f"Al–Co–Cr–Fe–Ni family: {len(g)} records, {g.comp_key.nunique()} compositions", loc='left')
    ax.legend(loc='upper left', handletextpad=0.2, borderaxespad=0.2)
    ax.set_ylim(0, 960)
    ax.text(0.98, 0.03, f"Random fold: {len(te)} test records; nearest training\n"
                        f"alloy a median {d_te.median():.1f} at.% away.\n"
                        f"System-grouped: all {len(g)} records held out",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color=INK2, linespacing=1.3)
    panel_label(ax, 'a', x=-0.15)

    # (b, c) distance from each test record to its nearest training alloy
    for k, ds in enumerate(('HV', 'YS')):
        ax = fig.add_subplot(gs[0, 1]) if k == 0 else fig.add_subplot(gs[1, 0])
        for p in PROTO_ORDER:
            v = np.sort(nn[(nn.dataset == ds) & (nn.protocol == p)].dnn.values)
            if len(v) == 0:
                continue
            ax.step(np.r_[0, v], np.r_[0, np.arange(1, len(v) + 1) / len(v)], where='post', color=PROTO_COL[p],
                    lw=1.4, label=PROTO_LABEL[p])
            ax.plot(np.median(v), 0.5, 'o', ms=3.6, color=PROTO_COL[p], mec='white', mew=0.5, zorder=4)
        ax.set_xscale('symlog', linthresh=1); ax.set_xlim(0, 80)
        ax.set_xticks([0, 1, 2, 5, 10, 20, 50]); ax.set_xticklabels(['0', '1', '2', '5', '10', '20', '50'])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('Composition change to nearest training alloy (at.%)')
        ax.set_ylabel('Cumulative fraction of test set')
        ax.set_title(f"{'Hardness' if ds == 'HV' else 'Yield strength'}: novelty of test alloys", loc='left')
        ax.axhline(0.5, color=MUTED, lw=0.5, ls=':')
        ax.text(0.03, 0.52, 'dots: medians', transform=ax.transAxes, fontsize=6.0, color=INK2)
        if k == 0:
            handles, labels = ax.get_legend_handles_labels()
        panel_label(ax, 'b' if k == 0 else 'c', x=-0.15)

    # (d) random-split error for seen vs unseen compositions, versus grouped protocols (XGBoost)
    ax = fig.add_subplot(gs[1, 1])
    sd = {'HV': hv.target.std(), 'YS': ys.target.std()}
    cats = ['P1: composition\nin training', 'P1: composition\nnot in training', 'P2: composition-\ngrouped', 'P3: system-\ngrouped']
    cols = [PROTO_COL['random'], PROTO_COL['random'], PROTO_COL['composition'], PROTO_COL['system']]
    x = np.arange(len(cats)); w = 0.36
    for k, ds in enumerate(('HV', 'YS')):
        s = su[(su.dataset == ds) & (su.model == 'XGB')].set_index('seen').RMSE
        pr = perf[(perf.dataset == ds) & (perf.model == 'XGB')].set_index('protocol').RMSE
        vals = [s[True], s[False], pr['composition'], pr['system']]
        unit = 'HV' if ds == 'HV' else 'MPa'
        bars = ax.bar(x + (k - 0.5) * w, np.array(vals) / sd[ds], w * 0.92, color=cols, edgecolor=INK2 if k else 'white',
                      lw=0.5, hatch='////' if k else None, label='hardness' if ds == 'HV' else 'yield strength')
        for xi, v in zip(x + (k - 0.5) * w, vals):
            ax.text(xi, v / sd[ds] + 0.01, f"{v:.0f}", ha='center', va='bottom', fontsize=6.0, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=6)
    ax.set_ylabel('XGBoost RMSE / target SD'); ax.set_ylim(0, 0.75)
    ax.grid(axis='x', visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#9db8d9', edgecolor='white', label='hardness (labels in HV)'),
                       Patch(facecolor='#9db8d9', edgecolor=INK2, hatch='////', label='yield strength (labels in MPa)')],
              loc='upper left', fontsize=6.0, borderaxespad=0.2)
    ax.set_title('Random-split accuracy rests on near-copies', loc='left')
    panel_label(ax, 'd', x=-0.15)
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=6, handlelength=1.6, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, 'Fig0b_near_duplicates')


if __name__ == '__main__':
    main()

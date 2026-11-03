"""Fig. 1: methodology workflow schematic (drawn in millimetre coordinates on a 190 x 112 mm canvas)."""
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from figstyle import setup, save, MM, INK, INK2, MUTED, PROTO_COL, PROTO_ORDER

W, H = 190, 112
HEAD = '#1b4f93'
FILL = '#f4f6f9'
EDGE = '#c9d3e0'


def box(ax, x, y, w, h, num, title, fill=FILL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fill, ec=EDGE, lw=0.7))
    ax.add_patch(FancyBboxPatch((x, y + h - 6.2), w, 6.2, boxstyle="round,pad=0,rounding_size=1.6", fc=HEAD, ec=HEAD, lw=0.7))
    ax.add_patch(Rectangle((x, y + h - 6.2), w, 3, fc=HEAD, ec='none'))
    ax.text(x + 2.2, y + h - 3.1, f"{num}", color='white', fontsize=7.5, fontweight='bold', va='center')
    ax.text(x + 6.4, y + h - 3.1, title, color='white', fontsize=7, fontweight='bold', va='center')


def txt(ax, x, y, s, size=6.2, **kw):
    ax.text(x, y, s, fontsize=size, va='top', ha=kw.pop('ha', 'left'), color=kw.pop('color', INK), linespacing=1.35, **kw)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=8, lw=1.0, color=INK2,
                                 shrinkA=0, shrinkB=0))


def chip(ax, x, y, w, h, color, label, desc, dark):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=color, ec='none'))
    c = 'white' if dark else INK
    ax.text(x + 1.8, y + h / 2, label, fontsize=6.6, fontweight='bold', color=c, va='center')
    ax.text(x + 7.6, y + h / 2, desc, fontsize=6.0, color=c, va='center')


def main():
    setup()
    hv = pd.read_csv('/home/claude/hea/data/hv_records.csv'); ys = pd.read_csv('/home/claude/hea/data/ys_records.csv')
    ge = pd.read_csv('/home/claude/hea/data/ext_ge_records.csv')
    sp = json.load(open('/home/claude/hea/data/splits.json'))
    n_el = {ds: len({s['label'] for s in sp[ds] if s['protocol'] == 'element'}) for ds in ('HV', 'YS')}
    n_tmp = {ds: len([r for s in sp[ds] if s['protocol'] == 'temporal' for r in s['test']]) for ds in ('HV', 'YS')}

    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    # ---- column A: data and features
    xa, wa = 1, 41
    box(ax, xa, 60, wa, 51, '1', 'Open data')
    txt(ax, xa + 2.5, 102.5, "MPEA database (Borg et al.)\n1,545 records, 2004–2019", size=6.2)
    ax.add_patch(FancyBboxPatch((xa + 2.5, 80.5), wa - 5, 10.2, boxstyle="round,pad=0,rounding_size=1", fc='white', ec=EDGE, lw=0.6))
    txt(ax, xa + 4, 89.6, f"Hardness: {len(hv)} records\n{hv.sys_key.nunique()} systems, room temp.", size=6.0)
    ax.add_patch(FancyBboxPatch((xa + 2.5, 69.3), wa - 5, 10.2, boxstyle="round,pad=0,rounding_size=1", fc='white', ec=EDGE, lw=0.6))
    txt(ax, xa + 4, 78.4, f"Yield strength: {len(ys):,} records\n{ys.sys_key.nunique()} systems, −269 to 1600 °C", size=6.0)
    txt(ax, xa + 2.5, 67.2, f"GE refractory set (Detor et al.)\n{len(ge)} alloys, independent lab", size=6.0, color=INK2)

    box(ax, xa, 1, wa, 56, '2', 'Composition features')
    rows = [('FRAC', '29', 'element fractions'), ('PHYS', '22', 'alloy theory: VEC, δ, Ω…'),
            ('MAG', '132', 'Magpie statistics'), ('PROC', '6', 'processing route'),
            ('TEST', '2', 'test T and type (YS)')]
    for i, (k, n, d) in enumerate(rows):
        y = 45.5 - i * 6.6
        ax.add_patch(FancyBboxPatch((xa + 2.5, y - 2.4), 10.5, 4.8, boxstyle="round,pad=0,rounding_size=0.8", fc='#dbe8f7', ec='none'))
        ax.text(xa + 7.75, y, k, fontsize=6.0, fontweight='bold', ha='center', va='center', color=HEAD)
        ax.text(xa + 14.5, y + 1.1, n, fontsize=6.0, fontweight='bold', va='center')
        ax.text(xa + 14.5, y - 1.4, d, fontsize=6.0, va='center', color=INK2)
    txt(ax, xa + 2.5, 11.8, "189 (HV) / 191 (YS) inputs, filtered,\nscaled and clipped per training fold",
        size=6.0, color=INK2)

    # ---- column B: protocols
    xb, wb = 46, 54
    box(ax, xb, 1, wb, 110, '3', 'Evaluation protocols')
    txt(ax, xb + 2.5, 102.5, "Identical frozen splits for every model; P1–P3:\n5-fold CV × 3 repeats (test sets in Fig. 2d)", size=6.0, color=INK2)
    desc = [('P1', 'Random CV', 'near-copies in train and test'),
            ('P2', 'Composition-grouped CV', 'test compositions unseen'),
            ('P3', 'System-grouped CV', 'test alloy systems unseen'),
            ('P4', 'Leave-one-element-out', f"{n_el['HV']} HV / {n_el['YS']} YS elements"),
            ('P5', 'Temporal: ≤ 2018 → 2019', f"{n_tmp['HV']} HV / {n_tmp['YS']} YS test records"),
            ('P6', 'External laboratory', 'GE alloys (hardness only)')]
    for i, ((lab, d1, d2), p) in enumerate(zip(desc, PROTO_ORDER)):
        y = 83 - i * 13.4
        ax.add_patch(FancyBboxPatch((xb + 9, y), wb - 11.5, 11, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc=PROTO_COL[p], ec='none'))
        c = 'white' if i >= 2 else INK
        ax.text(xb + 10.8, y + 5.5, lab, fontsize=7, fontweight='bold', color=c, va='center')
        ax.text(xb + 17.5, y + 7.4, d1, fontsize=6.0, fontweight='bold', color=c, va='center')
        ax.text(xb + 17.5, y + 3.4, d2, fontsize=6.0, color=c, va='center')
    # severity arrow
    ax.add_patch(FancyArrowPatch((xb + 5.2, 93), (xb + 5.2, 4), arrowstyle='-|>', mutation_scale=9, lw=1.2, color=HEAD))
    ax.text(xb + 3.2, 50, 'increasing extrapolation', rotation=90, fontsize=6.0, color=HEAD, va='center', ha='center')

    # ---- column C: models and uncertainty
    xc, wc = 104, 40
    box(ax, xc, 52, wc, 59, '4', 'Models and tuning')
    models = ['Mean baseline', 'kNN', 'Ridge', 'SVR', 'GPR', 'MLP', 'Random forest', 'XGBoost', 'TabPFN v2']
    for i, m in enumerate(models):
        col, row = i % 2, i // 2
        x = xc + 2.5 + col * 18.0; y = 99.5 - row * 6.4
        fc = '#fde3d3' if m == 'TabPFN v2' else 'white'
        ax.add_patch(FancyBboxPatch((x, y - 2.5), 17.0, 5.0, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=EDGE, lw=0.6))
        ax.text(x + 8.5, y, m, fontsize=6.0, ha='center', va='center')
    txt(ax, xc + 2.5, 67.2, "Nested random search; inner\nfolds mirror the outer protocol\n(TabPFN: pre-trained, untuned)",
        size=6.0, color=INK2)

    box(ax, xc, 1, wc, 48, '5', 'Prediction intervals')
    uq = [('CV+ naive', 'random calibration folds'), ('CV+ matched', 'folds grouped like deployment'),
          ('Normalized CV+', 'residuals scaled by d₅ + β'), ('Native', 'GP posterior, TabPFN quantiles')]
    for i, (k, d) in enumerate(uq):
        y = 36.2 - i * 8.6
        ax.add_patch(FancyBboxPatch((xc + 2.5, y - 3.6), wc - 5, 7.2, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc='white', ec=EDGE, lw=0.6))
        ax.text(xc + 4.2, y + 1.3, k, fontsize=6.0, fontweight='bold', va='center')
        ax.text(xc + 4.2, y - 1.6, d, fontsize=6.0, color=INK2, va='center')
    txt(ax, xc + 2.5, 5.6, "Nominal 68% and 90% levels", size=6.0, color=INK2)

    # ---- column D: analyses
    xd, wd = 148, 41
    box(ax, xd, 1, wd, 110, '6', 'Analyses and outputs')
    an = [('Accuracy and ranking', 'RMSE, R², ρ; Friedman,\nHolm; Kendall τ', 'Figs. 4–6'),
          ('Calibration', 'coverage, width,\nWinkler score', 'Figs. 7–8'),
          ('Applicability domain', 'd₅ distance; error–\nretention curves', 'Fig. 9'),
          ('Screening', 'within-system ρ;\ntop-20% recall', 'Sec. 3.6'),
          ('Interpretation', 'TreeSHAP; stability over\n15 grouped folds', 'Fig. 10'),
          ('Ablation', 'feature blocks, processing,\nobserved phase', 'Table 3')]
    for i, (k, d, ref) in enumerate(an):
        y = 98 - i * 14.2
        ax.add_patch(FancyBboxPatch((xd + 2.5, y - 6.0), wd - 5, 12.0, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc='white', ec=EDGE, lw=0.6))
        ax.text(xd + 4.2, y + 3.4, k, fontsize=6.1, fontweight='bold', va='center')
        ax.text(xd + wd - 4.0, y - 4.4, ref, fontsize=6.0, color=HEAD, va='center', ha='right')
        ax.text(xd + 4.2, y - 1.5, d, fontsize=6.0, color=INK2, va='center', linespacing=1.3)
    ax.add_patch(FancyBboxPatch((xd + 2.5, 3.5), wd - 5, 9.5, boxstyle="round,pad=0,rounding_size=0.8", fc=HEAD, ec='none'))
    ax.text(xd + wd / 2, 8.25, "Open benchmark: data,\nsplits, predictions, code", fontsize=6.0, color='white',
            ha='center', va='center', fontweight='bold', linespacing=1.3)

    # ---- arrows between stages
    arrow(ax, xa + wa / 2, 60, xa + wa / 2, 57.2)
    arrow(ax, xa + wa, 29, xb, 29)
    arrow(ax, xb + wb, 81, xc, 81)
    arrow(ax, xc + wc / 2, 52, xc + wc / 2, 49.2)
    arrow(ax, xc + wc, 81, xd, 81)
    arrow(ax, xc + wc, 25, xd, 25)
    save(fig, 'Fig0_workflow')


if __name__ == '__main__':
    main()

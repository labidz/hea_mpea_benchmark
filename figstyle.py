"""Shared matplotlib style for publication figures (Elsevier: 90 mm single / 190 mm double column)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MM = 1 / 25.4
SINGLE, DOUBLE = 90 * MM, 190 * MM
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
# ordinal protocol ramp (single hue, light -> dark = increasing extrapolation severity)
PROTO_ORDER = ['random', 'composition', 'system', 'element', 'temporal', 'external']
PROTO_LABEL = {'random': 'Random', 'composition': 'Composition-grouped', 'system': 'System-grouped',
               'element': 'Leave-one-element-out', 'temporal': 'Temporal (2019)', 'external': 'External (GE)'}
PROTO_COL = dict(zip(PROTO_ORDER, ['#b8d4f2', '#86b4ea', '#4f8fdc', '#2a6fc2', '#1b4f93', '#0d2f5e']))
MODEL_ORDER = ['Mean', 'kNN', 'Ridge', 'SVR', 'GPR', 'MLP', 'RF', 'XGB', 'TabPFN']
DIVERGING = LinearSegmentedColormap.from_list('div', ['#c4372f', '#f0b9a8', '#f2f1ee', '#a9c8ee', '#1b4f93'])
SEQ = LinearSegmentedColormap.from_list('seq', ['#dbe8f7', '#86b4ea', '#2a6fc2', '#0d2f5e'])


def setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'text.color': INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
        'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'lines.linewidth': 1.2,
    })


def panel_label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='bottom', ha='left')


def save(fig, name, outdir="/home/claude/hea/figures"):
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/{name}.png")
    fig.savefig(f"{outdir}/{name}.pdf")
    plt.close(fig)

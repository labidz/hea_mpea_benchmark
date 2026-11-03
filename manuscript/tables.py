"""Table builders (data come only from results/tables)."""
import json
import numpy as np
import pandas as pd

T = "/home/claude/hea/results/tables"
MODELS = ['Mean', 'kNN', 'Ridge', 'SVR', 'GPR', 'MLP', 'RF', 'XGB', 'TabPFN']
PROTOS = ['random', 'composition', 'system', 'element', 'temporal', 'external']
PNAME = {'random': 'Random', 'composition': 'Comp.-grouped', 'system': 'System-grouped', 'element': 'Leave-element-out',
         'temporal': 'Temporal', 'external': 'External'}


def table1(R):
    rows = [
        ['Records', R['HV_n'], R['YS_n'], R['GE_n']],
        ['Unique compositions', R['HV_comp'], R['YS_comp'], R['GE_comp']],
        ['Alloy systems (element sets)', R['HV_sys'], R['YS_sys'], R['GE_sys']],
        ['Source publications', R['HV_refs'], R['YS_refs'], 1],
        ['Records of repeated compositions', f"{R['HV_rep_rec']} ({R['HV_rep_comp']} comp.)", f"{R['YS_rep_rec']} ({R['YS_rep_comp']} comp.)", f"{R['GE_rep_rec']} ({R['GE_rep_comp']} comp.)"],
        ['Target range', f"{R['HV_min']:.0f}–{R['HV_max']:.0f} HV", f"{R['YS_min']:.0f}–{R['YS_max']:.0f} MPa", f"{R['GE_min']:.0f}–{R['GE_max']:.0f} HV"],
        ['Target mean ± SD', f"{R['HV_mean']:.0f} ± {R['HV_sd']:.0f} HV", f"{R['YS_mean']:.0f} ± {R['YS_sd']:.0f} MPa", f"{R['GE_mean']:.0f} ± {R['GE_sd']:.0f} HV"],
        ['Repeat scatter (pooled SD)^{a}', f"{R['noise']['HV']['pooled_sd_same_proc']:.0f} HV", f"{R['noise']['YS']['pooled_sd_same_proc']:.0f} MPa", '–'],
        ['Publication years', '2004–2019', '2005–2019', '2022'],
        ['Test conditions', 'room temperature', f"{R['YS_tension']} tension, {R['YS_compression']} compression; {R['YS_Tmin']:.0f} to {R['YS_Tmax']:.0f} °C", 'room temperature, 500 gf'],
        ['Role', 'training / P1–P5', 'training / P1–P5', 'external test (P6)'],
    ]
    return dict(caption="**Table 1.** Datasets used in this study.",
                header=['', 'Hardness (MPEA)', 'Yield strength (MPEA)', 'External hardness (GE)'],
                rows=[[str(c) for c in r] for r in rows], widths=[2.3, 2.0, 2.6, 2.0],
                notes=f"^{{a}}Pooled within-group standard deviation of records sharing composition and processing route "
                      f"(and, for yield strength, test type and temperature); {R['noise']['HV']['groups']} and "
                      f"{R['noise']['YS']['groups']} groups; an approximate noise floor (decomposition and confidence intervals "
                      f"in Table S15). Sources: MPEA database [@borg2020]; GE dataset [@detor2022].")


def table2(perf, pair):
    """RMSE (R2) for all models x protocols; best per column in bold; models whose paired cluster-bootstrap RMSE
    difference to the best model has a 95% CI containing zero are marked with a dagger (all protocols)."""
    rows = []
    for ds, unit in (('HV', 'HV'), ('YS', 'MPa')):
        g = perf[perf.dataset == ds]
        protos = [p for p in PROTOS if p in set(g.protocol)]
        rows.append([f"**{'Hardness' if ds == 'HV' else 'Yield strength'} ({unit})**"] + [''] * len(PROTOS))
        best = {p: g[(g.protocol == p) & (g.model != 'Mean')].sort_values('RMSE').model.iloc[0] for p in protos}
        ns = {}
        for p in protos:
            q = pair[(pair.dataset == ds) & (pair.protocol == p)]
            assert q.best.iloc[0] == best[p]
            ns[p] = {r.model for r in q.itertuples() if r.model != r.best and r.diff_lo <= 0}
        for m in MODELS:
            row = [m]
            for p in PROTOS:
                s_ = g[(g.protocol == p) & (g.model == m)]
                if s_.empty:
                    row.append('–'); continue
                txt = f"{s_.RMSE.iloc[0]:.0f} ({s_.R2.iloc[0]:.2f})"
                if m == best.get(p):
                    txt = f"**{txt}**"
                elif m in ns.get(p, set()):
                    txt += '^{†}'
                row.append(txt)
            rows.append(row)
    return dict(caption="**Table 2.** Test RMSE (and *R*^{2}) of all models under the six evaluation protocols. P1–P3: "
                        "RMSE of the pooled out-of-fold predictions of each repeat, averaged over three repeats of five-fold "
                        "cross-validation; leave-element-out: mean over held-out elements; temporal and external: single "
                        "test set.",
                header=['Model'] + [PNAME[p] for p in PROTOS], rows=rows, widths=[1.3, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45],
                notes="Bold: lowest RMSE in the column. ^{†}Indistinguishable from the bold model: the 95% interval of the "
                      "paired RMSE difference, from a cluster bootstrap that resamples alloy systems (2,000 replicates), "
                      "contains zero (Table S7; confidence intervals of every entry in Tables S3–S4). Mean: training-set mean "
                      "predictor. External: hardness models applied to the GE dataset [@detor2022].")


VNAME = {'FRAC': 'Element fractions only', 'PHYS': 'Alloy-theory descriptors only', 'MAG': 'Magpie statistics only',
         'PHYS+MAG': 'Alloy-theory + Magpie', 'ALL': 'All composition features (default)',
         'ALL-noPROC': 'Default without processing', 'ALL+PHASE': 'Default + observed phase^{a}'}


def table3(A, model='XGB'):
    protos = ['random', 'system', 'element', 'temporal']
    rows = []
    for v in VNAME:
        row = [VNAME[v]]
        for ds in ('HV', 'YS'):
            for p in protos:
                s = A[(A.ds == ds) & (A.variant == v) & (A.model == model) & (A.protocol == p)]
                row.append(f"{s.RMSE.iloc[0]:.0f}" if len(s) else '–')
        rows.append(row)
    # bold the best per column (excluding the leaky phase variant)
    for j in range(1, len(rows[0])):
        vals = [float(r[j]) for r in rows[:-1]]
        i = int(np.argmin(vals)); rows[i][j] = f"**{rows[i][j]}**"
    return dict(caption=f"**Table 3.** Feature-block ablation: test RMSE of {model} (fixed hyper-parameters) for hardness (HV) "
                        "and yield strength (MPa). Processing descriptors are included in all rows except the sixth; test "
                        "temperature and test type are included in every yield-strength model.",
                header=['Feature set', 'HV P1', 'HV P3', 'HV P4', 'HV P5', 'YS P1', 'YS P3', 'YS P4', 'YS P5'],
                rows=rows, widths=[2.6] + [0.85] * 8,
                notes="Bold: lowest RMSE among design-time feature sets. ^{a}Post-synthesis information (FCC/BCC/other), "
                      "unavailable when screening untested compositions. P1, random; P3, system-grouped; P4, leave-one-element-out; "
                      "P5, temporal. P1 and P3 values are means over 15 folds, P4 values means over held-out elements; P5 is a "
                      "single test set. GPR and kNN results are given in Table S9.")

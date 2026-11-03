"""Results and discussion text. Every number comes from result files through Res; qualitative claims are
asserted against the data so that the text cannot silently drift from the results."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TAB = '/home/claude/hea/results/tables_v2'
WORDS = dict(enumerate(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']))

H = "/home/claude/hea"
MODELS8 = ['kNN', 'Ridge', 'SVR', 'GPR', 'MLP', 'RF', 'XGB', 'TabPFN']
NAMES = {'XGB': 'XGBoost', 'RF': 'RF', 'GPR': 'GPR', 'TabPFN': 'TabPFN', 'Ridge': 'ridge regression', 'SVR': 'SVR', 'MLP': 'MLP',
         'kNN': 'kNN'}


def _and(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' or ' + xs[-1]


def _and_list(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else (xs[0] + ' and ' + xs[1] if len(xs) == 2 else ', '.join(xs[:-1]) + ' and ' + xs[-1])


def pct(a, b):
    return 100.0 * (a - b) / b


def ordinal(n):
    return f"{n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'}"


def results(R, X):
    _nn = pd.read_csv(f"{TAB}/nn_comp_distance.csv")
    nn_med = _nn.groupby(['dataset', 'protocol']).dnn.median().to_dict()
    pct_twin_ys = 100 * (_nn[(_nn.dataset == 'YS') & (_nn.protocol == 'random')].dnn < 0.05).mean()
    _su = pd.read_csv(f"{TAB}/random_seen_unseen.csv")
    su = {(r.dataset, bool(r.seen)): r.RMSE for r in _su[_su.model == 'XGB'].itertuples()}
    _hv = pd.read_csv("/home/claude/hea/data/hv_records.csv")
    _fam = _hv[_hv.sys_key == 'Al-Co-Cr-Fe-Ni']
    fam_n, fam_c = len(_fam), _fam.comp_key.nunique()
    _sp = json.load(open("/home/claude/hea/data/splits.json"))
    _t0 = set(next(s for s in _sp['HV'] if s['label'] == 'random-r0f0')['test'])
    fam_d = _nn[(_nn.label == 'random-r0f0') & _nn.rid.isin(set(_fam.rid) & _t0)].dnn.median()
    fam_t = len(set(_fam.rid) & _t0)
    assert abs(pct_twin_ys - 75) < 1 and nn_med[('HV', 'random')] < nn_med[('HV', 'composition')] < nn_med[('HV', 'system')]
    B = []
    hv_r_best, hv_r_rmse = X.best('HV', 'random'); ys_r_best, ys_r_rmse = X.best('YS', 'random')
    hv_c_best, hv_c_rmse = X.best('HV', 'composition'); hv_s_best, hv_s_rmse = X.best('HV', 'system')
    ys_c_best, ys_c_rmse = X.best('YS', 'composition'); ys_s_best, ys_s_rmse = X.best('YS', 'system')
    hv_r2r = X.p('HV', 'random', hv_r_best, 'R2'); hv_r2s = X.p('HV', 'system', hv_s_best, 'R2')
    ys_r2r = X.p('YS', 'random', ys_r_best, 'R2'); ys_r2s = X.p('YS', 'system', ys_s_best, 'R2')
    nf_hv = X.noise2['HV']['same_processing']; nf_ys = X.noise2['YS']['same_processing']
    hv_r_ci = X.ci('HV', 'random', hv_r_best); ys_r_ci = X.ci('YS', 'random', ys_r_best)
    hv_s_ci = X.ci('HV', 'system', hv_s_best); ys_s_ci = X.ci('YS', 'system', ys_s_best)
    assert nf_hv['median_group_sd'] < 0.7 * nf_hv['pooled_sd'] and nf_ys['median_group_sd'] < 0.5 * nf_ys['pooled_sd']
    assert hv_c_rmse > hv_r_rmse and hv_s_rmse > hv_c_rmse and ys_c_rmse > ys_r_rmse and ys_s_rmse > ys_c_rmse
    gap_hv = [pct(X.p('HV', 'system', m), X.p('HV', 'random', m)) for m in MODELS8]
    gap_ys = [pct(X.p('YS', 'system', m), X.p('YS', 'random', m)) for m in MODELS8]
    assert np.mean(gap_ys) > np.mean(gap_hv) and R['YS_random_overlap'] > R['HV_random_overlap']
    B += [
        ('h1', '3. Results and discussion'),
        ('h2', '3.1. Random cross-validation measures interpolation, not discovery'),
        ('p', f"Table 1 summarizes the datasets and Fig. 2 illustrates how the protocols partition them. Under random "
              f"cross-validation (P1) the best hardness model ({hv_r_best}) reached RMSE = {hv_r_rmse:.0f} HV (95% CI "
              f"{hv_r_ci[0]:.0f}–{hv_r_ci[1]:.0f} HV) and *R*^{{2}} = {hv_r2r:.2f}, and the best yield-strength model "
              f"({ys_r_best}) RMSE = {ys_r_rmse:.0f} MPa ({ys_r_ci[0]:.0f}–{ys_r_ci[1]:.0f} MPa) and *R*^{{2}} = {ys_r2r:.2f} "
              f"(Table 2). These values are in the range typically reported for HEA property models. They should be read "
              f"against the scatter of repeated measurements, which is only approximately known (Table S15): hardness records "
              f"that share composition and processing differ with a pooled SD of {nf_hv['pooled_sd']:.0f} HV (95% CI "
              f"{nf_hv['ci'][0]:.0f}–{nf_hv['ci'][1]:.0f} HV; {nf_hv['groups']} groups), but the median within-group SD is "
              f"{nf_hv['median_group_sd']:.0f} HV, so the pooled value is dominated by a few discordant groups; for yield "
              f"strength the pooled SD ({nf_ys['pooled_sd']:.0f} MPa) is as large as the random-split RMSE, while the median "
              f"group SD is {nf_ys['median_group_sd']:.0f} MPa and some groups are re-tabulations of one measurement. Random "
              f"splitting, moreover, leaves the composition of {R['HV_random_overlap']:.0%} of hardness and "
              f"{R['YS_random_overlap']:.0%} of yield-strength test records in the training set, so part of this accuracy "
              f"reflects recall of the same alloy rather than prediction."),
        ('p', f"Fig. 3 shows how random splitting creates this overlap. In the largest alloy family, Al–Co–Cr–Fe–Ni ({fam_n} records of "
              f"{fam_c} compositions), the {WORDS.get(fam_t, fam_t)} test records of one random fold lay a median of {fam_d:.1f} at.% "
              f"in composition from their nearest training alloy (Fig. 3a). Across the hardness dataset, the median composition change "
              f"from a test alloy to its nearest training alloy was {nn_med[('HV', 'random')]:.1f} at.% under random "
              f"splitting, against {nn_med[('HV', 'composition')]:.1f}, {nn_med[('HV', 'system')]:.1f} and "
              f"{nn_med[('HV', 'element')]:.1f} at.% under composition-grouped, system-grouped and leave-one-element-out "
              f"validation, {nn_med[('HV', 'temporal')]:.1f} at.% for the 2019 literature and "
              f"{nn_med[('HV', 'external')]:.1f} at.% for the external alloys (Fig. 3b). For yield strength the random-split "
              f"median was {nn_med[('YS', 'random')]:.0f} at.%, because {pct_twin_ys:.0f}% of test records had an exact "
              f"compositional twin in training (Fig. 3c). Within random splits, XGBoost predicted such twins more accurately "
              f"than the remaining test records ({su[('HV', True)]:.0f} versus {su[('HV', False)]:.0f} HV and "
              f"{su[('YS', True)]:.0f} versus {su[('YS', False)]:.0f} MPa; Fig. 3d): exact repeats inflate random-split "
              f"accuracy strongly for yield strength, whose twins are often the same alloy measured at another temperature "
              f"or after another processing route, but only mildly for hardness, where close relatives within a compositional series matter more, as the "
              f"de-duplication test below confirms."),
        ('p', f"Holding out whole compositions (P2) raised the best RMSE to {hv_c_rmse:.0f} HV and {ys_c_rmse:.0f} MPa (Fig. 4), and "
              f"holding out whole alloy systems (P3) to {hv_s_rmse:.0f} HV (95% CI {hv_s_ci[0]:.0f}–{hv_s_ci[1]:.0f} HV; "
              f"*R*^{{2}} = {hv_r2s:.2f}) and {ys_s_rmse:.0f} MPa ({ys_s_ci[0]:.0f}–{ys_s_ci[1]:.0f} MPa; *R*^{{2}} = {ys_r2s:.2f}). Averaged over the eight non-trivial models, the system-grouped RMSE exceeded the "
              f"random-split RMSE by {np.mean(gap_hv):.0f}% for hardness (range {min(gap_hv):.0f}–{max(gap_hv):.0f}%) and by "
              f"{np.mean(gap_ys):.0f}% for yield strength (range {min(gap_ys):.0f}–{max(gap_ys):.0f}%). The penalty was larger "
              f"for yield strength, whose random folds are more redundant because one alloy is often tested at several "
              f"temperatures. Exact repeats are, however, not the main source of optimism: collapsing repeated hardness "
              f"records to one median value per composition and processing route ({X.dd['n_dedup']} records) changed the "
              f"random-split XGBoost RMSE only from {X.dd['orig_random']:.0f} to {X.dd['dedup_random']:.0f} HV, while the "
              f"system-grouped RMSE remained at {X.dd['dedup_system']:.0f} HV (Table S8). The optimism therefore stems mainly "
              f"from near-duplicates—members of the same compositional series that differ by a few atomic per cent—which "
              f"random folds distribute over training and test sets and which de-duplication cannot remove."),
    ]
    # ---------------------------------------------------------------- 3.2
    hv_e_best, hv_e_rmse = X.best('HV', 'element'); ys_e_best, ys_e_rmse = X.best('YS', 'element')
    hv_t_best, hv_t_rmse = X.best('HV', 'temporal'); ys_t_best, ys_t_rmse = X.best('YS', 'temporal')
    hv_x_best, hv_x_rmse = X.best('HV', 'external')
    x_bias = [X.p('HV', 'external', m, 'bias') for m in MODELS8]
    x_rho = {m: X.p('HV', 'external', m, 'rho') for m in MODELS8}
    hv_xr2 = [X.p('HV', 'external', m, 'R2') for m in MODELS8]
    assert max(hv_xr2) < 0 and max(x_bias) < 0
    et = element_transfer(X)
    hv_t_ci = X.ci('HV', 'temporal', hv_t_best); ys_t_ci = X.ci('YS', 'temporal', ys_t_best)
    to = {r.dataset: r for r in X.temporal_overlap_v2.itertuples()}
    assert abs(to['HV'].frac_comp_seen - R['temporal_HV_comp_seen']) < 0.005 and abs(to['YS'].frac_comp_seen - R['temporal_YS_comp_seen']) < 0.005
    assert to['HV'].test_pubs_also_in_train == 0 and to['YS'].test_pubs_also_in_train == 0
    xs = X.external_subsets_v2; xa = xs[xs.method.isna() & xs.model.isin(MODELS8)]
    xs_n = xa.groupby('subset').n.first().to_dict(); xs_sys = xa.groupby('subset').n_systems.first().astype(int).to_dict()
    xs_rng = lambda sub: (xa[xa.subset == sub].RMSE.min(), xa[xa.subset == sub].RMSE.max())
    def xs_best(sub):
        g = xa[xa.subset == sub].sort_values('RMSE').iloc[0]
        return g.model, g.RMSE, g.R2
    assert xs_rng('previously represented systems')[1] < xs_rng('unseen systems')[0]
    assert xs_rng('Ru-containing')[0] > xs_rng('Ru-free')[1] - 50 and xs_best('unseen systems, Ru-free')[2] < 0
    assert xs_n['all'] == R['GE_n']
    ys_poor = et['YS'][et['YS'].medR2 < 0.1].sort_values('medR2'); ys_good = et['YS'][et['YS'].medR2 >= 0.55]
    hv_poor = et['HV'].sort_values('medR2').head(2)
    rho_parent_ys = spearmanr(et['YS'].parent_in_train, et['YS'].medR2).correlation
    rho_parent_hv = spearmanr(et['HV'].parent_in_train, et['HV'].medR2).correlation
    assert 0.3 < rho_parent_ys < 0.6 and abs(rho_parent_hv) < 0.3
    EY = et['YS'].set_index('el')
    assert 'Co' in set(ys_poor.el) and EY.loc['Ni', 'medR2'] < 0.2 and EY.loc['Fe', 'medR2'] < 0.2
    assert EY.loc['Cu', 'medR2'] < 0.1 and EY.loc['Cu', 'parent_in_train'] > 0.6
    ni_parent = float(EY.loc['Ni', 'parent_in_train'])
    hv_e_ci = X.ci('HV', 'element', hv_e_best); ys_e_ci = X.ci('YS', 'element', ys_e_best)
    B += [
        ('h2', '3.2. Unseen elements, the next year of literature and an independent laboratory'),
        ('p', f"Leave-one-element-out validation (P4) probes transfer to chemistries containing an element absent from "
              f"training. Errors rose further, to a mean RMSE over the held-out elements of {hv_e_rmse:.0f} HV ({NAMES[hv_e_best]}; "
              f"95% CI {hv_e_ci[0]:.0f}–{hv_e_ci[1]:.0f} HV) and {ys_e_rmse:.0f} MPa ({NAMES[ys_e_best]}; {ys_e_ci[0]:.0f}–"
              f"{ys_e_ci[1]:.0f} MPa), with large element-to-element variation (Fig. 5; Tables S5–S6). For yield strength, "
              f"transfer was poorest when " + _and(ys_poor.el.tolist()) + f" was held out (median *R*^{{2}} across models ≤ "
              f"{ys_poor.medR2.max():.2f}) and weak for Ni and Fe ({EY.loc['Ni', 'medR2']:.2f} and "
              f"{EY.loc['Fe', 'medR2']:.2f}). Ni, Fe and Co define the family of 3d-transition-metal HEAs, and removing one of "
              f"them leaves few of the corresponding parent systems in training (e.g. only {ni_parent:.0%} of the Ni-containing "
              f"test records had their Ni-free parent system in the training data). By contrast, models transferred well to "
              f"alloys containing " + _and(ys_good.sort_values('medR2', ascending=False).el.tolist()) +
              f" (median *R*^{{2}} ≥ 0.55). Across elements, the median *R*^{{2}} correlated moderately with the "
              f"fraction of test records whose parent system remained in training (Spearman *ρ* = {rho_parent_ys:.2f}, "
              f"{len(et['YS'])} elements), with Cu as a clear exception ({EY.loc['Cu', 'parent_in_train']:.0%} of parents "
              f"present, median *R*^{{2}} = {EY.loc['Cu', 'medR2']:.2f}). For hardness this association was weak "
              f"(*ρ* = {rho_parent_hv:.2f}); the most difficult elements were {hv_poor.el.iloc[0]} and {hv_poor.el.iloc[1]} "
              f"(median *R*^{{2}} {hv_poor.medR2.iloc[0]:.2f} and {hv_poor.medR2.iloc[1]:.2f})."),
        ('p', f"The temporal split (P5) asks how a model trained on the literature up to 2018 would have fared on the alloys "
              f"reported in 2019. The overlap between the two periods is limited but not negligible (Table S14). None of the "
              f"{to['HV'].test_publications} hardness and {to['YS'].test_publications} yield-strength publications of 2019 "
              f"contributed training records; {to['HV'].frac_comp_seen:.0%} and {to['YS'].frac_comp_seen:.0%} of their records "
              f"had a composition, and {to['HV'].frac_system_seen:.0%} and {to['YS'].frac_system_seen:.0%} an alloy system, "
              f"already present in the earlier literature, whereas {to['HV'].frac_new_element:.0%} and "
              f"{to['YS'].frac_new_element:.0%} contained an element never reported before; the median composition change to "
              f"the nearest earlier alloy was {to['HV'].median_nn_at_pct:.0f} and {to['YS'].median_nn_at_pct:.0f} at.%. The "
              f"practice of the field also moved: the share of powder-metallurgy hardness records rose from "
              f"{R['temporal_HV_POWDER_tr']:.0%} to {R['temporal_HV_POWDER_te']:.0%}, that of wrought yield-strength records "
              f"from {R['temporal_YS_WROUGHT_tr']:.0%} to {R['temporal_YS_WROUGHT_te']:.0%}, the share of hardness records "
              f"without processing information from {to['HV'].frac_unknown_proc_train:.0%} to "
              f"{to['HV'].frac_unknown_proc_test:.0%}, and the mean hardness from {R['temporal_HV_ytr']:.0f} to "
              f"{R['temporal_HV_yte']:.0f} HV. The best models reached RMSE = {hv_t_rmse:.0f} HV ({NAMES[hv_t_best]}; 95% CI "
              f"{hv_t_ci[0]:.0f}–{hv_t_ci[1]:.0f} HV) and {ys_t_rmse:.0f} MPa ({NAMES[ys_t_best]}; {ys_t_ci[0]:.0f}–{ys_t_ci[1]:.0f} "
              f"MPa), {pct(hv_t_rmse, hv_r_rmse):.0f}% and {pct(ys_t_rmse, ys_r_rmse):.0f}% above the random-split estimates "
              f"(Fig. 6a,c). Because the test set comprises a single year from {to['HV'].test_publications} and "
              f"{to['YS'].test_publications} publications, these estimates are uncertain, but they are the closest proxy in "
              f"our data for applying a published model to the next experiments of the field."),
        ('p', f"The independent GE dataset (P6) is the most severe test: hard, arc-melted refractory alloys rich in Re and Ru "
              f"from a single laboratory. All models failed in absolute terms (*R*^{{2}} between {min(hv_xr2):.2f} and "
              f"{max(hv_xr2):.2f}; best RMSE {hv_x_rmse:.0f} HV, {NAMES[hv_x_best]}) and systematically under-predicted hardness "
              f"(mean bias {min(x_bias):.0f} to {max(x_bias):.0f} HV; Fig. 6b). Of the {R['GE_n']} external alloys, "
              f"{R['GE_Ru']} contain Ru, which never occurs in the training data, and Re occurs in only {R['HV_Re_train']} "
              f"training records. Table S13 separates these effects. Excluding the {xs_n['all'] - xs_n['overlap compositions excluded']} "
              f"external alloys whose composition also occurs in the MPEA data left the picture unchanged (best RMSE "
              f"{xs_best('overlap compositions excluded')[1]:.0f} HV). {WORDS[xs_n['previously represented systems']].capitalize()} "
              f"alloys from the {WORDS[xs_sys['previously represented systems']]} GE systems that are represented in the MPEA "
              f"data were predicted with RMSEs of {xs_rng('previously represented systems')[0]:.0f}–"
              f"{xs_rng('previously represented systems')[1]:.0f} HV by the eight non-trivial models, no worse than under "
              f"random splitting, whereas the "
              f"{xs_n['unseen systems']} alloys from unseen systems gave {xs_rng('unseen systems')[0]:.0f}–"
              f"{xs_rng('unseen systems')[1]:.0f} HV. Errors were largest for the {xs_n['Ru-containing']} Ru-containing alloys "
              f"({xs_rng('Ru-containing')[0]:.0f}–{xs_rng('Ru-containing')[1]:.0f} HV) and smaller, but still without "
              f"predictive skill, for the {xs_n['unseen systems, Ru-free']} Ru-free alloys of unseen systems (best RMSE "
              f"{xs_best('unseen systems, Ru-free')[1]:.0f} HV, *R*^{{2}} = {xs_best('unseen systems, Ru-free')[2]:.2f}). The "
              f"external failure is therefore a failure on new chemistry, aggravated by an unseen element, and possibly by an "
              f"inter-laboratory offset that these data cannot separate from it (Section 3.9). Rank information was partially "
              f"retained (Spearman *ρ* up to {max(x_rho.values()):.2f}, {NAMES[max(x_rho, key=x_rho.get)]}), but selecting the hardest "
              f"alloys from this set was little better than chance (Section 3.6). Random cross-validation gives no warning of "
              f"such a failure, whereas the applicability-domain distance did (Section 3.5)."),
    ]
    # ranking, with alloy-system cluster-bootstrap inference
    rk = X.rank.set_index(['dataset', 'protocol'])
    best_counts, combos, unresolved, solo = {}, [], [], []
    for ds in ('HV', 'YS'):
        for p in ('random', 'composition', 'system', 'element', 'temporal', 'external'):
            if p in set(X.perf[X.perf.dataset == ds].protocol):
                b, ind = X.indist(ds, p)
                assert b == X.best(ds, p)[0]
                best_counts[b] = best_counts.get(b, 0) + 1; combos.append((ds, p, b))
                (unresolved if ind else solo).append((ds, p, b))
    n_cases = len(combos)
    tab_wins = best_counts.get('TabPFN', 0)
    assert tab_wins >= 6
    NAME = {'XGB': 'XGBoost', 'RF': 'RF', 'GPR': 'GPR', 'TabPFN': 'TabPFN', 'Ridge': 'ridge regression', 'SVR': 'SVR', 'MLP': 'MLP', 'kNN': 'kNN'}
    PDESC = {'random': 'random splits', 'composition': 'new compositions', 'system': 'new systems', 'element': 'unseen elements',
             'temporal': 'the 2019 literature', 'external': 'the external alloys'}
    PROP = {'HV': 'hardness', 'YS': 'yield strength'}
    exc = [f"{NAME[b]} for {PROP[ds]} on {PDESC[p]}" for ds, p, b in combos if b != 'TabPFN']
    exc_txt = ", ".join(exc[:-1]) + " and " + exc[-1] if len(exc) > 1 else exc[0]
    solo_txt = (f"{PROP[solo[0][0]]} on " + _and_list([PDESC[p] for _, p, _ in solo]).replace(' and ', ' and on ')
                if len({d for d, _, _ in solo}) == 1 else _and_list([f"{PROP[ds]} on {PDESC[p]}" for ds, p, b in solo]))
    assert all(b == 'TabPFN' for _, _, b in solo)
    b_hs, ind_hs = X.indist('HV', 'system')
    _ps = X.perf[(X.perf.dataset == 'HV') & (X.perf.protocol == 'system')].set_index('model').RMSE
    grp = sorted([b_hs] + ind_hs, key=lambda m: _ps[m])
    ts = X.tuning_scheme_v2
    ts_big = ts.loc[ts.change_pct.idxmax()]
    tau = lambda ds, p: (rk.loc[(ds, p), 'kendall_tau_vs_random'], rk.loc[(ds, p), 'tau_lo'], rk.loc[(ds, p), 'tau_hi'])
    t_he, t_hx, t_yc, t_ys = tau('HV', 'element'), tau('HV', 'external'), tau('YS', 'composition'), tau('YS', 'system')
    assert t_hx[2] < t_he[0] and t_yc[1] > 0
    B += [
        ('p', f"Model rankings depended on the property and the protocol (Table 2). TabPFN, used without task-specific "
              f"tuning, had the lowest RMSE in {tab_wins} of the {n_cases} property–protocol combinations; the exceptions were "
              + exc_txt + ". The paired cluster-bootstrap intervals, however, resolved few of these differences: in "
              f"{len(unresolved)} of the {n_cases} combinations at least one other model was indistinguishable from the most "
              f"accurate one (Table 2; Table S7)—for hardness under system-grouped validation, for example, "
              + _and_list([NAME[m] for m in grp]) + f" ({_ps[grp[0]]:.0f}–{_ps[grp[-1]]:.0f} HV)—and TabPFN was "
              f"distinguishable from every other model only for {solo_txt}. For hardness, the random-split ranking was a poor "
              f"guide to the ranking for unseen elements (Kendall's *τ* = {t_he[0]:.2f}, 95% CI {t_he[1]:.2f} to "
              f"{t_he[2]:.2f}) and was not preserved on the external set (*τ* = {t_hx[0]:.2f}, {t_hx[1]:.2f} to {t_hx[2]:.2f}); "
              f"for yield strength it was largely preserved under grouped validation (*τ* = {t_yc[0]:.2f}, {t_yc[1]:.2f} to "
              f"{t_yc[2]:.2f}, and {t_ys[0]:.2f}, {t_ys[1]:.2f} to {t_ys[2]:.2f}). Deployment-specific tuning did not "
              f"systematically improve accuracy: compared with tuning on system-grouped inner folds, it changed the P4–P6 test "
              f"RMSE of the tuned models by {ts.change_pct.min():+.0f}% to {ts.change_pct.max():+.0f}% (median "
              f"{ts.change_pct.median():+.0f}%; lower in {int((ts.change_pct < 0).sum())} of {len(ts)} model–protocol "
              f"combinations; Table S16). The largest change, a {ts_big.change_pct:.0f}% higher error of "
              f"{NAME[ts_big.model]} on {PDESC[ts_big.protocol]}, shows that hyper-parameters selected on a few inner folds "
              f"are themselves noisy; the benefit of deployment-matched folds lies mainly in calibration (Section 3.4). No "
              f"model family escaped the extrapolation penalty, and the distance-weighted nearest-neighbour baseline was among "
              f"the two weakest models under every protocol, confirming that the descriptors carry information beyond simple "
              f"look-up of similar alloys."),
    ]
    for ds in ('HV', 'YS'):
        for p in set(X.perf[X.perf.dataset == ds].protocol):
            assert 'kNN' in sorted(MODELS8, key=lambda m: X.p(ds, p, m))[-2:], ('kNN weakness claim', ds, p)
    return B


def element_transfer(X):
    sp = json.load(open(f"{H}/data/splits.json")); E = X.folds
    out = {}
    for ds, fn in (('HV', 'hv_records.csv'), ('YS', 'ys_records.csv')):
        rec = pd.read_csv(f"{H}/data/{fn}").set_index('rid'); rows = []
        for s in sp[ds]:
            if s['protocol'] != 'element':
                continue
            el = s['label'].split('-')[1]; te = list(s['test']); trsys = set(rec.drop(index=te).sys_key)
            parent = rec.loc[te].sys_key.apply(lambda k: '-'.join(e for e in k.split('-') if e != el))
            g = E[(E.dataset == ds) & (E.label == s['label']) & (E.model != 'Mean')]
            rows.append(dict(el=el, n=len(te), parent_in_train=float(parent.isin(trsys).mean()), medR2=float(g.R2.median())))
        out[ds] = pd.DataFrame(rows)
    return out


def results2(R, X):
    B = []
    A = X.ablation_summary

    def a(ds, var, model, proto):
        return float(A[(A.ds == ds) & (A.variant == var) & (A.model == model) & (A.protocol == proto)].RMSE.iloc[0])
    P4 = ('random', 'system', 'element', 'temporal')
    hv = {v: {p: a('HV', v, 'XGB', p) for p in P4} for v in A.variant.unique()}
    hg = {v: {p: a('HV', v, 'GPR', p) for p in P4} for v in A.variant.unique()}
    ys = {v: {p: a('YS', v, 'XGB', p) for p in P4} for v in A.variant.unique()}
    yg = {v: {p: a('YS', v, 'GPR', p) for p in P4} for v in A.variant.unique()}
    dt = ('FRAC', 'PHYS', 'MAG', 'PHYS+MAG', 'ALL')
    spread_r = max(hv[v]['random'] for v in dt) - min(hv[v]['random'] for v in dt)
    assert hv['FRAC']['element'] > hv['ALL']['element'] and hg['FRAC']['element'] > hg['ALL']['element']
    assert ys['FRAC']['element'] < ys['ALL']['element'] and yg['FRAC']['element'] > yg['ALL']['element']
    assert hv['ALL-noPROC']['temporal'] > hv['ALL']['temporal']
    ph_hv = max(100 * (1 - hv['ALL+PHASE'][p] / hv['ALL'][p]) for p in ('random', 'system', 'element'))
    ph_ys = max(100 * (1 - ys['ALL+PHASE'][p] / ys['ALL'][p]) for p in P4)
    B += [
        ('h2', '3.3. What the representation contributes'),
        ('p', f"The feature ablation (Table 3; GPR and kNN in Table S9) shows that the representation matters little for "
              f"interpolation but more for extrapolation, and that its benefit is model- and property-dependent. Under "
              f"random splitting, the five design-time feature sets gave hardness RMSEs within {spread_r:.0f} HV of each "
              f"other with XGBoost. When an element was held out, element fractions alone—whose column for the held-out "
              f"element is constant in training—were clearly inferior for hardness (XGBoost {hv['FRAC']['element']:.0f} "
              f"versus {hv['ALL']['element']:.0f} HV with all features; GPR {hg['FRAC']['element']:.0f} versus "
              f"{hg['ALL']['element']:.0f} HV), because elemental-property statistics place an unseen element in the "
              f"context of known ones. For yield strength the picture was mixed: the descriptors again helped the GPR "
              f"(fractions only versus all features: {yg['FRAC']['element']:.0f} versus {yg['ALL']['element']:.0f} MPa) but "
              f"not XGBoost ({ys['FRAC']['element']:.0f} versus {ys['ALL']['element']:.0f} MPa), indicating that composition "
              f"descriptors alone do not capture what changes when a new element enters a temperature-dependent property. "
              f"Processing information was decisive for the temporal test: removing the six processing indicators raised "
              f"the hardness RMSE on the 2019 literature from {hv['ALL']['temporal']:.0f} to "
              f"{hv['ALL-noPROC']['temporal']:.0f} HV, consistent with the shift in processing routes noted above. Adding "
              f"the observed phase constitution, which is available only after synthesis, reduced the RMSE by at most "
              f"{ph_hv:.0f}% (hardness) and {ph_ys:.0f}% (yield strength), so models that use it gain little in exchange for "
              f"losing their applicability to untested compositions."),
    ]
    return B


def results3(R, X):
    B = []
    hard = ['system', 'element', 'temporal', 'external']
    cov = lambda ds, p, m, meth: X.u(ds, p, m, meth, 'coverage')
    wid = lambda ds, p, m, meth: X.u(ds, p, m, meth, 'mean_width')
    wk = lambda ds, p, m, meth: X.u(ds, p, m, meth, 'winkler')
    below = lambda ds, p, m, meth: X.u(ds, p, m, meth, 'sys_cov_frac_below_nominal_minus_10')
    ci = lambda ds, p, meth, m='XGB': X.ucov(ds, p, m, meth)
    f2 = lambda t: f"{t[0]:.2f} ({t[1]:.2f}–{t[2]:.2f})"
    hv_naive = {p: cov('HV', p, 'XGB', 'CV+naive90') for p in ['random', 'composition'] + hard}
    hv_match = {p: cov('HV', p, 'XGB', 'CV+matched90') for p in ['composition'] + hard}
    hv_norm = {p: cov('HV', p, 'XGB', 'nCV+matched90') for p in ['composition'] + hard}
    hv_gw = {p: cov('HV', p, 'XGB', 'gwCV+matched90') for p in ['composition'] + hard}
    yp = ['composition', 'system', 'element', 'temporal']
    ys_naive = {p: cov('YS', p, 'XGB', 'CV+naive90') for p in ['random'] + yp}
    ys_match = {p: cov('YS', p, 'XGB', 'CV+matched90') for p in yp}
    ys_norm = {p: cov('YS', p, 'XGB', 'nCV+matched90') for p in yp}
    ys_gw = {p: cov('YS', p, 'XGB', 'gwCV+matched90') for p in yp}
    w_hv = 100 * (wid('HV', 'system', 'XGB', 'CV+matched90') / wid('HV', 'system', 'XGB', 'CV+naive90') - 1)
    w_ys = 100 * (wid('YS', 'system', 'XGB', 'CV+matched90') / wid('YS', 'system', 'XGB', 'CV+naive90') - 1)
    w_hve = 100 * (wid('HV', 'element', 'XGB', 'CV+matched90') / wid('HV', 'element', 'XGB', 'CV+naive90') - 1)
    shifted = [('HV', p) for p in hard] + [('YS', p) for p in ('system', 'element', 'temporal')]
    assert all(wk(d, p, 'XGB', 'CV+matched90') <= 1.01 * wk(d, p, 'XGB', 'CV+naive90') for d, p in shifted)
    strict = [(d, p) for d, p in shifted if p in ('element', 'temporal', 'external')]
    assert all(wk(d, p, 'XGB', 'CV+matched90') < 0.95 * wk(d, p, 'XGB', 'CV+naive90') for d, p in strict)
    wk_hs = (wk('HV', 'system', 'XGB', 'CV+matched90'), wk('HV', 'system', 'XGB', 'CV+naive90'))
    # theoretical CV+ bound for P1 (K = 5, n = outer training-set size)
    nmin = {ds: int(R[f'{ds}_n'] * 0.8) for ds in ('HV', 'YS')}
    bound = {ds: 1 - 0.2 - min(2 * (1 - 1 / 5) / (n / 5 + 1), (1 - 5 / n) / 6) for ds, n in nmin.items()}
    assert hv_naive['random'] > bound['HV'] and ys_naive['random'] > bound['YS']
    # group weighting and per-system coverage
    assert all(hv_gw[p] >= hv_match[p] - 0.005 for p in hard) and all(ys_gw[p] >= ys_match[p] - 0.005 for p in yp)
    pb = {k: below(*k, 'XGB', m) for k, m in ((('HV', 'random'), 'CV+naive90'), (('HV', 'system'), 'CV+naive90'),
                                              (('HV', 'element'), 'CV+naive90'), (('YS', 'element'), 'CV+naive90'))}
    pbm = {k: below(*k, 'XGB', 'CV+matched90') for k in (('HV', 'system'), ('HV', 'element'), ('YS', 'element'), ('HV', 'temporal'))}
    n_ht = int(X.u('HV', 'temporal', 'XGB', 'CV+matched90', 'n_sys_units'))
    assert pbm[('HV', 'system')] < pb[('HV', 'system')] and pbm[('HV', 'element')] < pb[('HV', 'element')]
    # external calibration choices
    ext = {m: cov('HV', 'external', 'XGB', m) for m in ('CV+naive90', 'CV+matched90', 'CV+matched-sys90', 'nCV+naive90',
                                                        'nCV+matched90', 'nCV+matched-sys90', 'gwCV+matched90')}
    sd_hv = X.noise2['HV']['target_sd']
    ext_w = {m: wid('HV', 'external', 'XGB', m) / sd_hv for m in ('CV+matched90', 'nCV+matched90', 'nCV+matched-sys90', 'nCV+naive90')}
    assert ext['CV+matched90'] > ext['CV+matched-sys90'] >= ext['CV+naive90'] and ext['nCV+naive90'] > ext['nCV+matched90']
    xsub = X.external_subsets_v2; xsub = xsub[xsub.method == 'CV+ matched'].set_index('subset').coverage90
    tab_m = {(d, p): cov(d, p, 'TabPFN', 'CV+matched90') for d, p in (('HV', 'system'), ('YS', 'system'), ('HV', 'element'),
                                                                      ('YS', 'element'), ('HV', 'temporal'), ('HV', 'external'))}
    gp = {p: cov('HV', p, 'GPR', 'GP-native90') for p in hard}
    tp = {p: cov('HV', p, 'TabPFN', 'TabPFN-native90') for p in hard}
    gpw_e = wid('HV', 'element', 'GPR', 'GP-native90') / wid('HV', 'element', 'XGB', 'CV+matched90')
    _hw = pd.read_csv(f"{TAB}/needed_vs_supplied_halfwidth.csv")
    nd = {(r.dataset, r.protocol): r.needed for r in _hw.itertuples()}
    nv = {(r.dataset, r.protocol): r.naive for r in _hw.itertuples()}
    mt = {(r.dataset, r.protocol): r.matched for r in _hw.itertuples()}
    nm = {(r.dataset, r.protocol): r.normalized for r in _hw.itertuples()}
    nv_rng = (_hw[_hw.dataset == 'HV'].naive.min(), _hw[_hw.dataset == 'HV'].naive.max())
    assert abs(_hw[(_hw.dataset == 'HV') & (_hw.protocol == 'system')].cov_naive.iloc[0] - hv_naive['system']) < 0.01
    assert abs(_hw[(_hw.dataset == 'HV') & (_hw.protocol == 'external')].cov_matched.iloc[0] - hv_match['external']) < 0.01
    assert nd['HV', 'system'] > nv['HV', 'system'] and mt['HV', 'system'] > nd['HV', 'system']
    B += [
        ('h2', '3.4. Calibration under the deployment shift improves, but does not guarantee, interval coverage'),
        ('p', f"Fig. 7 compares the empirical coverage of nominal 90% prediction intervals, with cluster-bootstrap 95% "
              f"confidence intervals in parentheses (all methods, widths, interval scores and the 68% level in Table S12 and "
              f"Fig. S1). Under random splitting, naive CV+ intervals covered {hv_naive['random']:.2f} (hardness) and "
              f"{ys_naive['random']:.2f} (yield strength) of the XGBoost test values, close to the nominal level and above the "
              f"finite-sample guarantee of Section 2.5 (≈{bound['HV']:.2f} for *K* = 5 and the present *n*). Under "
              f"extrapolation the same intervals became progressively over-confident: for hardness, coverage fell to "
              f"{f2(ci('HV', 'system', 'CV+naive90'))} for new systems, {f2(ci('HV', 'element', 'CV+naive90'))} for unseen "
              f"elements, {f2(ci('HV', 'temporal', 'CV+naive90'))} on the 2019 literature and "
              f"{f2(ci('HV', 'external', 'CV+naive90'))} on the external alloys; for yield strength to "
              f"{ys_naive['system']:.2f}, {ys_naive['element']:.2f} and {ys_naive['temporal']:.2f} for the first three of these. "
              f"Random-fold residuals measure interpolation error, which underestimates extrapolation error. CV+ with the "
              f"deployment-specific calibration folds of Section 2.4 raised the coverage for new systems to "
              f"{f2(ci('HV', 'system', 'CV+matched90'))} and {f2(ci('YS', 'system', 'CV+matched90'))}, at the price of "
              + (f"{w_hv:.0f}% wider intervals for both properties" if round(w_hv) == round(w_ys) else
                 f"{w_hv:.0f}% (hardness) and {w_ys:.0f}% (yield strength) wider intervals") +
              f", and for unseen elements to {f2(ci('HV', 'element', 'CV+matched90'))} and "
              f"{f2(ci('YS', 'element', 'CV+matched90'))}, with {w_hve:.0f}% wider hardness intervals. The same held for "
              f"TabPFN (matched coverage {tab_m[('HV', 'system')]:.2f} and {tab_m[('YS', 'system')]:.2f} for new systems, "
              f"{tab_m[('HV', 'element')]:.2f} and {tab_m[('YS', 'element')]:.2f} for unseen elements). Matched calibration "
              f"improved the interval (Winkler) score of XGBoost by more than 5% for unseen elements, the temporal split and "
              f"the external alloys and left it essentially unchanged for new hardness systems ({wk_hs[0]:.0f} versus "
              f"{wk_hs[1]:.0f} HV), i.e. the wider intervals were warranted by the larger errors. Weighting each calibration "
              f"system equally (group-weighted CV+) raised the coverage slightly further, e.g. to {hv_gw['element']:.2f} and "
              f"{ys_gw['element']:.2f} for unseen elements, because the residuals of small systems, which were larger on average, "
              f"are otherwise outweighed by those of large, well-sampled systems. Distance normalization gave "
              f"{hv_norm['element']:.2f} and {ys_norm['element']:.2f} for unseen elements; combined with naive calibration it "
              f"produced very wide intervals for yield strength (mean width {wid('YS', 'element', 'XGB', 'nCV+naive90'):,.0f} "
              f"MPa for unseen elements, against {wid('YS', 'element', 'XGB', 'CV+matched90'):,.0f} MPa for matched CV+; "
              f"Table S12)."),
        ('p', f"Marginal coverage hides variation between alloy systems. Even under random splitting, "
              f"{pb[('HV', 'random')]:.0%} of the hardness system–fold units with at least five test records had a coverage "
              f"more than 10 percentage points below nominal; with naive calibration this share rose to "
              f"{pb[('HV', 'system')]:.0%} for new systems and {pb[('HV', 'element')]:.0%} (hardness) and "
              f"{pb[('YS', 'element')]:.0%} (yield strength) for unseen elements, and matched calibration reduced it to "
              f"{pbm[('HV', 'system')]:.0%}, {pbm[('HV', 'element')]:.0%} and {pbm[('YS', 'element')]:.0%} (Table S12). "
              f"The temporal and external hardness shifts were not fully corrected by any scheme. On the 2019 literature, matched "
              f"coverage was {f2(ci('HV', 'temporal', 'CV+matched90'))} for XGBoost and {pbm[('HV', 'temporal')]:.0%} of the "
              f"{n_ht} system units fell more than 10 points short; GPR reached {cov('HV', 'temporal', 'GPR', 'CV+matched90'):.2f}"
              f", whereas for yield strength matched calibration came close to nominal ({f2(ci('YS', 'temporal', 'CV+matched90')).replace(' (', '; ').rstrip(')')}). "
              f"For the external alloys, the leave-one-element-out calibration chosen beforehand from label-free information "
              f"(Section 2.4) gave {f2(ci('HV', 'external', 'CV+matched90'))}, system-grouped calibration "
              f"{ext['CV+matched-sys90']:.2f} and naive calibration {ext['CV+naive90']:.2f}. By subset (Table S13), matched "
              f"coverage was {xsub['Ru-free']:.2f} for Ru-free and {xsub['Ru-containing']:.2f} for Ru-containing alloys and "
              f"{xsub['unseen systems']:.2f} for the alloys of unseen systems. Distance-normalized intervals covered more of "
              f"the external alloys ({ext['nCV+matched90']:.2f}, {ext['nCV+matched-sys90']:.2f} and {ext['nCV+naive90']:.2f} "
              f"with element-grouped, system-grouped and naive calibration), but only by becoming {ext_w['nCV+matched90']:.1f}–"
              f"{ext_w['nCV+naive90']:.1f} times as wide as the standard deviation of the hardness data, which makes them "
              f"uninformative; naive calibration produced the widest intervals because its calibration alloys lie close to "
              f"their training data, so the ratio of test to calibration distance is largest. The native GPR intervals "
              f"({gp['system']:.2f}, {gp['element']:.2f}, {gp['temporal']:.2f} and {gp['external']:.2f} for P3–P6) and "
              f"TabPFN quantiles ({tp['system']:.2f}, {tp['element']:.2f}, {tp['temporal']:.2f} and {tp['external']:.2f}) "
              f"degraded less than naive CV+ because their predictive variance grows away from the data; for unseen elements the "
              f"GPR intervals were " + (f"about as wide as matched XGBoost intervals of similar coverage (mean widths "
                                         f"{wid('HV', 'element', 'GPR', 'GP-native90'):.0f} and "
                                         f"{wid('HV', 'element', 'XGB', 'CV+matched90'):.0f} HV)"
                                         if abs(gpw_e - 1) < 0.1 else f"{gpw_e:.1f} times as wide as matched XGBoost intervals of "
                                         f"similar coverage") + "."),
        ('p', f"Fig. 8 shows the mechanism. The half-width that a constant-width interval needs for 90% coverage—the 90th "
              f"percentile of the absolute test error—grew with novelty, from {nd['HV', 'random']:.0f} HV under random "
              f"splitting to {nd['HV', 'system']:.0f} HV for new systems, {nd['HV', 'element']:.0f} HV for unseen elements, "
              f"{nd['HV', 'temporal']:.0f} HV for the 2019 literature and {nd['HV', 'external']:.0f} HV for the external "
              f"alloys. The half-width supplied by naive CV+ stayed between {nv_rng[0]:.0f} and {nv_rng[1]:.0f} HV under every "
              f"protocol, because its random calibration folds always measure the same interpolation error. Matched "
              f"calibration folds hold out whole systems or elements (Fig. 8a), so their residuals are extrapolation errors "
              f"and the supplied half-width rose accordingly ({mt['HV', 'system']:.0f} HV for new systems and "
              f"{mt['HV', 'element']:.0f} HV for unseen elements); for the 2019 literature ({mt['HV', 'temporal']:.0f} HV) and "
              f"the external alloys ({mt['HV', 'external']:.0f} HV) it remained well below what was needed, because "
              f"historical residuals cannot anticipate changes in processing, reporting and measurement practice or the "
              f"hardening by an element that never occurs in training. For yield strength the picture was the same "
              f"({nd['YS', 'element']:.0f} MPa needed for unseen elements, against {nv['YS', 'element']:.0f} MPa from naive, "
              f"{mt['YS', 'element']:.0f} MPa from matched and {nm['YS', 'element']:.0f} MPa from normalized CV+). Taken "
              f"together, an uncertainty estimate is only as trustworthy as the match between its calibration split and the "
              f"intended use; even a matched split yields empirical, not guaranteed, coverage, which should be reported "
              f"under grouped, temporal or external protocols with system-level confidence intervals rather than under random "
              f"splits."),
    ]
    return B


def results4(R, X):
    B = []
    adm, rho = X.ad_median, X.ad_rho
    order = ['random', 'composition', 'temporal', 'system', 'element', 'external']
    hv_ad = {p: adm[('HV', p)] for p in order}
    hv_rmse_x = {p: X.p('HV', p, 'XGB') for p in order}
    r_proto = spearmanr([hv_ad[p] for p in order], [hv_rmse_x[p] for p in order]).correlation
    ret = X.retention
    tm = X.temporal_ad
    assert rho[('HV', 'temporal')] < 0 and tm['near_unknown'] > tm['far_unknown']
    asn = {ds: X.ad_sensitivity_v2[X.ad_sensitivity_v2.dataset == ds] for ds in ('HV', 'YS')}
    dflt = asn['HV'][(asn['HV'].space.str.contains('default')) & (asn['HV'].k == 5)].iloc[0]
    assert abs(dflt.retained50 - ret['HV']['ad']) < 0.005 and abs(dflt.all_P3P5 - ret['HV']['full']) < 0.005
    assert asn['HV'].sort_values('rho_pooled').space.iloc[-1].startswith('element fractions')
    assert asn['HV'].retained50.min() > ret['HV']['oracle'] + 0.2 and asn['YS'].retained50.min() > ret['YS']['oracle'] + 0.2
    B += [
        ('h2', '3.5. The applicability domain flags shifts, but is a weak per-alloy error predictor'),
        ('p', f"The median AD distance *d*_{{5}} of the hardness test alloys increased from {hv_ad['random']:.1f} under "
              f"random splitting to {hv_ad['system']:.1f} for new systems, {hv_ad['element']:.1f} for unseen elements and "
              f"{hv_ad['external']:.1f} for the external alloys, tracking the growth of the error across protocols "
              f"(Spearman *ρ* between median *d*_{{5}} and XGBoost RMSE over the six protocols = {r_proto:.2f}; Fig. 9). The "
              f"exception was the temporal split, whose error was large although its distances were moderate "
              f"({hv_ad['temporal']:.1f}), because that shift lies largely in processing and reporting practice rather than "
              f"chemistry. At the population level, *d*_{{5}} is therefore a useful alarm for chemical novelty: the GE "
              f"alloys would have been flagged before any experiment. At the level of individual alloys the association "
              f"was weak (Spearman *ρ* between *d*_{{5}} and the absolute error: {rho[('HV', 'system')]:.2f} for new systems, "
              f"{rho[('HV', 'element')]:.2f} for unseen elements and {rho[('HV', 'external')]:.2f} for the external set; "
              f"{rho[('YS', 'system')]:.2f}, {rho[('YS', 'element')]:.2f} and {rho[('YS', 'temporal')]:.2f} for yield "
              f"strength), and for the 2019 hardness records it was even negative ({rho[('HV', 'temporal')]:.2f}): "
              f"{tm['near_unknown']:.0%} of the alloys in the nearest distance tertile were reported without processing "
              f"information, and these records were under-predicted by {abs(tm['unknown_bias']):.0f} HV on average. "
              f"Accordingly, discarding the half of the shifted-protocol predictions (P3–P5) with the largest *d*_{{5}} lowered "
              f"the normalized RMSE of the retained half only from {ret['HV']['full']:.2f} to {ret['HV']['ad']:.2f} for "
              f"hardness and from {ret['YS']['full']:.2f} to {ret['YS']['ad']:.2f} for yield strength, against "
              f"{ret['HV']['oracle']:.2f} and {ret['YS']['oracle']:.2f} for an oracle (Fig. 9c). Distance in descriptor space "
              f"captures novelty of the inputs, not the size of the response to that novelty. These conclusions did not "
              f"depend on the neighbourhood size or the representation (Table S11): over *k* = 1–20 and six descriptor "
              f"spaces, the pooled record-level *ρ* ranged from {asn['HV'].rho_pooled.min():.2f} to "
              f"{asn['HV'].rho_pooled.max():.2f} for hardness and from {asn['YS'].rho_pooled.min():.2f} to "
              f"{asn['YS'].rho_pooled.max():.2f} for yield strength, the protocol-level *ρ* from "
              f"{asn['HV'].rho_population.min():.2f} to {asn['HV'].rho_population.max():.2f} (hardness), and the retained "
              f"half of the P3–P5 predictions had a normalized RMSE of {asn['HV'].retained50.min():.2f}–"
              f"{asn['HV'].retained50.max():.2f} (hardness) and {asn['YS'].retained50.min():.2f}–"
              f"{asn['YS'].retained50.max():.2f} (yield strength). Distances based on element fractions alone ranked "
              f"individual hardness errors best (*ρ* up to {asn['HV'].rho_pooled.max():.2f}), consistent with element "
              f"novelty being the dominant shift, but no choice approached the oracle."),
    ]
    return B


def results5(R, X):
    W = X.within_system_rho; K = X.topk_recall
    ws = W.groupby(['dataset', 'model']).rho.median()
    hv_ws = {m: float(ws[('HV', m)]) for m in ('GPR', 'RF', 'XGB')}
    ys_ws = {m: float(ws[('YS', m)]) for m in ('GPR', 'RF', 'XGB')}
    assert min(hv_ws.values()) > max(ys_ws.values()) - 0.2 and np.mean(list(hv_ws.values())) > np.mean(list(ys_ws.values()))
    nsys_hv = int(W[(W.dataset == 'HV') & (W.model == 'XGB')].groupby('rep').size().mean())
    nsys_ys = int(W[(W.dataset == 'YS') & (W.model == 'XGB')].groupby('rep').size().mean())

    def rec(ds, lab, m, sc='mean'):
        s = K[(K.dataset == ds) & (K.label == lab) & (K.model == m) & (K.score == sc)]
        return float(s.recall.iloc[0])
    t_hv = {m: rec('HV', 'temporal-2019', m) for m in ('GPR', 'XGB', 'TabPFN')}
    t_ucb = rec('HV', 'temporal-2019', 'XGB', 'UCB')
    x_all = {m: rec('HV', 'external-GE', m) for m in MODELS8}
    t_ys = [rec('YS', 'temporal-2019', m) for m in MODELS8]
    B = [
        ('h2', '3.6. Ranking candidates is easier than predicting their properties'),
        ('p', f"For design, the ability to rank candidates often matters more than absolute accuracy. Within held-out alloy "
              f"systems (P3; systems with at least five test records, {nsys_hv} hardness and {nsys_ys} yield-strength "
              f"systems per repeat), the median within-system Spearman correlation between measured and predicted hardness "
              f"was " + ", ".join(f"{v:.2f} ({NAMES[m]})" for m, v in hv_ws.items()) + "; for room-temperature compressive yield "
              f"strength it was lower, " + ", ".join(f"{v:.2f} ({NAMES[m]})" for m, v in ys_ws.items()) + " (Table S10). "
              f"Composition trends inside a system thus transfer to new systems even when the absolute level does not, "
              f"more reliably for hardness than for yield strength, whose values also depend on microstructural state that "
              f"the processing classes capture only coarsely. On the 2019 literature, ranking by the point prediction "
              f"recovered {t_hv['GPR']:.0%} (GPR), {t_hv['XGB']:.0%} (XGBoost) and {t_hv['TabPFN']:.0%} (TabPFN) of the "
              f"truly hardest quintile against 20% expected by chance, and ranking by the upper bound of the matched 68% "
              f"interval raised the XGBoost recall to {t_ucb:.0%}; for yield strength the recall was "
              f"{min(t_ys):.0%}–{max(t_ys):.0%}. On the external GE alloys, however, recall was only "
              f"{min(x_all.values()):.0%}–{max(x_all.values()):.0%} across models (median {np.median(list(x_all.values())):.0%}), "
              f"so model-guided selection was not reliably better than chance. Models "
              f"can therefore support prioritization within and near the explored composition space, but not the selection "
              f"of record-breaking alloys from chemically new families."),
    ]
    sh = {ds: json.load(open(f"{H}/results/tables/shap_stability_{ds}.json")) for ds in ('HV', 'YS')}
    st = {ds: pd.read_csv(f"{H}/results/tables/shap_stability_{ds}.csv") for ds in ('HV', 'YS')}
    Xs = pd.read_csv(f"{H}/results/shap_X_HV.csv"); sv = np.load(f"{H}/results/shap_values_HV.npy")
    j = list(Xs.columns).index('PHYS:VEC mean'); v = Xs['PHYS:VEC mean'].values
    lowv, highv = sv[v < 7.5, j].mean(), sv[v >= 8.0, j].mean()
    Ys = pd.read_csv(f"{H}/results/shap_X_YS.csv"); svy = np.load(f"{H}/results/shap_values_YS.npy")
    jt = list(Ys.columns).index('TEST:T_C'); T = Ys['TEST:T_C'].values
    rt, hot = svy[(T >= 20) & (T <= 30), jt].mean(), svy[T >= 1000, jt].mean()
    imp_ys = pd.read_csv(f"{H}/results/tables/shap_importance_YS.csv", index_col=0).iloc[:, 0]
    rank_T = list(imp_ys.index).index('TEST:T_C') + 1
    top_hv = list(st['HV'].feature.head(6)); top_ys = list(st['YS'].feature.head(7))
    assert {'PHYS:Shear modulus strength model', 'PHYS:Radii local mismatch', 'PHYS:VEC mean'} <= set(top_hv)
    assert {'PHYS:Radii local mismatch', 'MAG:avg_dev GSvolume_pa', 'MAG:avg_dev MeltingT'} <= set(top_ys)
    f6_hv = (st['HV'].top10_frequency.head(6) >= 0.8).sum(); f6_ys = (st['YS'].top10_frequency.head(6) >= 0.8).sum()
    B += [
        ('h2', '3.7. Physical interpretation and its stability'),
        ('p', f"SHAP analysis (Fig. 10) shows that the hardness model relies mainly on descriptors of solid-solution "
              f"strengthening—the shear-modulus-based strengthening descriptor, local atomic-size and shear-modulus "
              f"mismatch and the spread of elemental atomic volumes—and on the valence-electron concentration. This is "
              f"consistent with misfit-based strengthening theories [@toda2015,varvenne2016,maresca2020] and with the "
              f"established link between VEC and phase constitution [@guo2011]: alloys with VEC < 7.5, which tend to form "
              f"BCC or BCC-containing structures, received an average SHAP contribution of {lowv:+.0f} HV, whereas alloys "
              f"with VEC ≥ 8, typically single-phase FCC, received {highv:+.0f} HV (Fig. 10b). For yield strength, the test "
              f"temperature ranked {ordinal(rank_T)} in importance and contributed on average {rt:+.0f} MPa at room "
              f"temperature and {hot:+.0f} MPa at ≥ 1000 °C (Fig. 10d), reflecting thermally activated softening; the most "
              f"important composition descriptors were again size-misfit measures (local radius mismatch, spread of atomic "
              f"volumes) together with the mean atomic volume and the spread of melting temperatures, and the test type "
              f"(tension versus compression) also ranked among the top features. The attributions were stable: the mean "
              f"pairwise Spearman correlation between the importance vectors of the 15 models trained on system-grouped "
              f"folds was {sh['HV']['pairwise_spearman_mean']:.2f} for hardness and {sh['YS']['pairwise_spearman_mean']:.2f} "
              f"for yield strength, and " + (f"all six top-ranked features of both models were" if f6_hv == 6 and f6_ys == 6 else f"{f6_hv} (hardness) and {f6_ys} (yield strength) of the six top-ranked features were") + f" "
              f"among the ten most important in at least 80% of the fold models. Stable, physically sensible attributions "
              f"are, however, compatible with poor extrapolation: they describe what the model learned within the training "
              f"domain and cannot certify that the same relations hold for unseen chemistries such as the Ru-containing "
              f"external alloys."),
    ]
    return B

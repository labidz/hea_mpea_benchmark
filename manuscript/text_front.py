TITLE = ("Extrapolation-aware evaluation and calibrated uncertainty for machine-learning models of high-entropy "
         "alloy hardness and yield strength")

KEYWORDS = "High-entropy alloys; Machine learning; Out-of-distribution generalization; Conformal prediction; Uncertainty quantification; Hardness; Yield strength"


def _tab_stats(X):
    combos = [(ds, p) for ds in ('HV', 'YS') for p in ('random', 'composition', 'system', 'element', 'temporal', 'external')
              if p in set(X.perf[X.perf.dataset == ds].protocol)]
    wins = sum(X.best(ds, p)[0] == 'TabPFN' for ds, p in combos)
    resolved = sum(X.best(ds, p)[0] == 'TabPFN' and not X.indist(ds, p)[1] for ds, p in combos)
    near = sum(X.best(ds, p)[0] == 'TabPFN' or 'TabPFN' in X.indist(ds, p)[1] for ds, p in combos)
    return wins, resolved, near, len(combos)


def ABSTRACT(R, X):
    hv_r = X.best('HV', 'random'); hv_s = X.best('HV', 'system'); hv_t = X.best('HV', 'temporal')
    ys_r = X.best('YS', 'random'); ys_s = X.best('YS', 'system'); ys_t = X.best('YS', 'temporal')
    models = ['kNN', 'Ridge', 'SVR', 'GPR', 'MLP', 'RF', 'XGB', 'TabPFN']
    xr2 = max(X.p('HV', 'external', m, 'R2') for m in models)
    n_s = X.u('HV', 'system', 'XGB', 'CV+naive90'); m_s = X.u('HV', 'system', 'XGB', 'CV+matched90')
    n_e = X.u('HV', 'element', 'XGB', 'CV+naive90'); m_e = X.u('HV', 'element', 'XGB', 'CV+matched90')
    n_x = X.u('HV', 'external', 'XGB', 'CV+naive90'); m_x = X.u('HV', 'external', 'XGB', 'CV+matched90')
    m_t = X.u('HV', 'temporal', 'XGB', 'CV+matched90')
    wins, resolved, near, n_c = _tab_stats(X)
    assert resolved < wins and m_t < 0.75 and m_x < 0.75
    return (f"Machine-learning models of high-entropy alloy (HEA) properties are usually validated by random "
            f"cross-validation, which rewards interpolation among closely related alloys. We quantified this optimism on "
            f"{R['HV_n']} hardness and {R['YS_n']:,} yield-strength records from an open multi-principal element alloy "
            f"database and on {R['GE_n']} independent refractory alloys. Nine regressors, including the tabular foundation "
            f"model TabPFN, were evaluated under six protocols of increasing novelty, with inner validation folds that "
            f"reproduce each deployment scenario and confidence intervals from a bootstrap over alloy systems. The best "
            f"hardness root-mean-square error rose from {hv_r[1]:.0f} HV (random splits) to {hv_s[1]:.0f} HV for unseen "
            f"alloy systems and {hv_t[1]:.0f} HV for alloys reported a year later; yield-strength errors rose from "
            f"{ys_r[1]:.0f} to {ys_s[1]:.0f} and {ys_t[1]:.0f} MPa. On the independent alloys, which contain elements absent "
            f"from training, every model failed (coefficient of determination ≤ {xr2:.2f}). TabPFN had the lowest error in "
            f"{wins} of {n_c} property–protocol combinations, but most differences from tree ensembles and Gaussian "
            f"processes were within statistical uncertainty. Conformal 90% intervals calibrated with random folds covered "
            f"{n_s:.0%} of unseen-system, {n_e:.0%} of unseen-element and {n_x:.0%} of external hardness values. "
            f"Calibration folds that mimic the deployment raised coverage to {m_s:.0%} and {m_e:.0%} for unseen systems and "
            f"elements, but only to {m_t:.0%} and {m_x:.0%} for the temporal and external shifts; under such shifts coverage "
            f"is empirical, not guaranteed. A nearest-neighbour applicability-domain distance flagged the external shift but "
            f"was a weak predictor of individual errors. The curated data, frozen splits, predictions and code are released "
            f"as a benchmark.")


def HIGHLIGHTS(R, X):
    return [
        "Random cross-validation overstates the accuracy of HEA models for new alloys",
        "Six frozen protocols probe new compositions, systems, elements, years and labs",
        "A bootstrap over alloy systems leaves many model differences unresolved",
        "Deployment-matched calibration improves interval coverage but cannot guarantee it",
        f"Open benchmark: curated data, frozen splits, predictions and code for {R['HV_n'] + R['YS_n']:,} records",
    ]

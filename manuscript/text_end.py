def discussion(R, X, limitations):
    return [
        ('h2', '3.8. Implications for model-guided alloy design'),
        ('p', "Taken together, the results suggest a hierarchy of evidence for HEA property models. Random cross-validation "
              "is a useful check that a model has learned the composition–property map within the sampled region, but it "
              "should not be reported as the expected accuracy for proposed alloys; when near-duplicate alloys are as common "
              "as in MPEA data, it mainly measures interpolation within compositional series. Composition- and "
              "system-grouped cross-validation give realistic estimates for, respectively, re-optimizing a known system and "
              "entering a new one; leave-one-element-out, temporal and external tests probe genuinely new chemistry and "
              "changing practice. We therefore recommend that ML studies of MPEAs (i) report at least one grouped protocol "
              "alongside random cross-validation, together with the fraction of test compositions present in the training "
              "data; (ii) tune hyper-parameters and calibrate intervals with inner folds that reproduce the intended "
              "deployment—grouped, leave-element-out or forward-chaining as appropriate; (iii) report the empirical coverage "
              "of the intervals under that protocol, with system-level confidence intervals and per-system coverage, and "
              "state that no distribution-free guarantee holds under the shift; (iv) base statistical comparisons on "
              "resampling of alloy systems rather than of records or correlated folds; (v) report the applicability-domain "
              "distance of every proposed candidate relative to the distances seen in validation; and (vi) compare against "
              "the experimental repeat scatter and simple baselines (training mean, nearest neighbour). Each of these steps "
              "is inexpensive, and the released benchmark makes them straightforward to adopt."),
        ('p', "The results also clarify what kind of design decision current models can support. Within a known or "
              "neighbouring alloy system the hardness models ranked candidates well even when their absolute errors were "
              "large (Section 3.6), so they are useful for prioritizing compositions within a family and for active-learning "
              "loops in which each experiment corrects the model. Far outside the training domain—new elements, new "
              "processing routes, unfamiliar laboratories—neither the predictions nor random-fold uncertainty estimates "
              "should be taken at face value, and model-guided proposals there should be treated as hypotheses to be tested "
              "with wide, honestly calibrated error bars. The accuracy of an untuned tabular foundation model suggests that, "
              "for datasets of this size, pre-trained priors are a competitive default, although its advantage over tree "
              "ensembles and Gaussian processes was mostly within the statistical uncertainty of these data; the uncertainty "
              "estimates of such models need the same deployment-matched calibration as those of any other model."),
        ('h2', '3.9. Limitations'),
        ('p', limitations),
    ]


def conclusions(R, X):
    from text_front import _tab_stats
    hv_r = X.best('HV', 'random'); hv_s = X.best('HV', 'system'); hv_t = X.best('HV', 'temporal')
    ys_r = X.best('YS', 'random'); ys_s = X.best('YS', 'system')
    wins, resolved, near, n_c = _tab_stats(X)
    m_s = X.u('HV', 'system', 'XGB', 'CV+matched90'); m_e = X.u('HV', 'element', 'XGB', 'CV+matched90')
    m_t = X.u('HV', 'temporal', 'XGB', 'CV+matched90'); m_x = X.u('HV', 'external', 'XGB', 'CV+matched90')
    ts = X.tuning_scheme_v2
    return [
        ('h1', '4. Conclusions'),
        ('p', f"An extrapolation-aware benchmark of eight machine-learning regressors and a mean baseline on {R['HV_n']} "
              f"hardness and {R['YS_n']:,} yield-strength records of multi-principal element alloys, complemented by an "
              f"independent external test set, leads to four conclusions. (1) Random cross-validation substantially "
              f"overstates the accuracy relevant to alloy discovery: the best RMSE grew from {hv_r[1]:.0f} to {hv_s[1]:.0f} HV "
              f"and from {ys_r[1]:.0f} to {ys_s[1]:.0f} MPa when whole alloy systems were held out, reached {hv_t[1]:.0f} HV on "
              f"the following year's literature, and no model was predictive for an external set dominated by alloys of "
              f"unseen systems and an unseen element. (2) The pre-trained tabular foundation model TabPFN had the lowest error "
              f"in {wins} of {n_c} property–protocol combinations without any tuning and was best or statistically "
              f"indistinguishable from the best in {near}, but a bootstrap over alloy systems separated it from all other "
              f"models in only {resolved}; no model family escaped the extrapolation penalty, and for hardness the ranking "
              f"under random splits did not transfer to unseen elements or the external alloys. Tuning with inner folds that "
              f"reproduce the deployment did not systematically improve accuracy (median change {ts.change_pct.median():+.0f}%). "
              f"(3) Conformal prediction intervals calibrated on random folds become over-confident under extrapolation. "
              f"Calibration folds that mimic the deployment shift raised the empirical coverage of 90% intervals to "
              f"{m_s:.2f} for new hardness systems and {m_e:.2f} for unseen elements in these data, but this is an empirical "
              f"property without a distribution-free guarantee: coverage varied between alloy systems, and the temporal "
              f"({m_t:.2f}) and inter-laboratory ({m_x:.2f}) hardness shifts remained under-covered. (4) A nearest-neighbour "
              f"applicability-domain distance detects population-level chemical shift but ranks individual errors poorly, "
              f"irrespective of the neighbourhood size and descriptor space. Grouped, temporal and external evaluation with "
              f"deployment-matched calibration and system-level inference should therefore accompany ML models proposed for "
              f"HEA design; the curated data, frozen splits, predictions and code are provided to make this routine."),
    ]

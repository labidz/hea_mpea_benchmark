"""Methods section text. Markup: *italic*, **bold**, ^{sup}, _{sub}; citations [@key1,key2]; cross refs plain text.
Blocks: ('h2', text) | ('h3', text) | ('p', text) | ('eq', name) | ('table', id) | ('figure', id)."""


def methods(R):
    return [
        ('h1', '2. Materials and methods'),
        ('p', "Fig. 1 summarizes the study design. Two property datasets and an independent external set were curated from "
              "open sources (Section 2.1) and described by composition-derived features (Section 2.2). Every model was "
              "evaluated on the same frozen splits of six protocols of increasing extrapolation severity (Section 2.3), with "
              "hyper-parameters tuned on inner folds that reproduce the deployment scenario of each protocol (Section 2.4). "
              "Prediction intervals were calibrated with random or deployment-matched folds (Section 2.5), and the results were "
              "analysed for accuracy, calibration, applicability domain, screening value and interpretability (Sections 2.6–2.7), "
              "with alloy systems as the resampling unit for all statistical inference (Section 2.8)."),
        ('h2', '2.1. Data sources and curation'),
        ('p', "The primary data source was the open multi-principal element alloy (MPEA) mechanical-property database of "
              "Borg et al. [@borg2020], which consolidates two earlier compilations [@gorsse2018,couzinie2018] with data extracted "
              f"from articles published in 2019. The public release (GitHub commit {R['mpea_commit']}) contains {R['mpea_records']} "
              "composition–property records, each annotated with the nominal composition, processing route, observed phases, "
              "test type and temperature, and the source DOI and publication year. Two regression targets were extracted. "
              f"(i) **Hardness**: all records with a reported Vickers hardness measured at room temperature (test temperature 25 °C "
              f"or not reported), giving {R['HV_n']} records of {R['HV_comp']} unique compositions from {R['HV_sys']} alloy systems "
              f"and {R['HV_refs']} publications ({R['HV_min']:.0f}–{R['HV_max']:.0f} HV); {R['HV_excl_T']} hardness values "
              f"measured at {R['HV_excl_Tmin']:.0f}–{R['HV_excl_Tmax']:.0f} °C were excluded. (ii) **Yield strength**: all records with a reported yield strength, test type "
              f"(tension or compression) and test temperature, giving {R['YS_n']} records of {R['YS_comp']} unique compositions from "
              f"{R['YS_sys']} systems and {R['YS_refs']} publications ({R['YS_min']:.0f}–{R['YS_max']:.0f} MPa; test temperatures "
              f"{R['YS_Tmin']:.0f} to {R['YS_Tmax']:.0f} °C)."),
        ('p', "Nominal compositions were parsed into atomic fractions and normalized to unity. Each record was assigned a "
              "*composition key* (fractions rounded to 0.001) and a *system key* (the sorted set of constituent elements, e.g. "
              "Al–Co–Cr–Fe–Ni). Repeated measurements are intrinsic to MPEA literature: the same nominal alloy is often re-reported "
              "after a different processing route, by another group, or—for yield strength—at several test temperatures. In the "
              f"hardness set, {R['HV_rep_comp']} compositions occur more than once and account for {R['HV_rep_rec']} records; in the "
              f"yield-strength set, {R['YS_rep_comp']} compositions account for {R['YS_rep_rec']} records. These repeats were retained "
              "as genuine observations, because averaging would hide the processing and test-condition dependence, and were instead "
              "handled at the level of the evaluation protocol (Section 2.3). The observed-phase label (FCC, BCC or other) was "
              "retained only for an ablation, because it is known only after an alloy has been made and characterized."),
        ('p', "An independent external hardness set was taken from the GE Research refractory-alloy screening study of Detor et "
              f"al. [@detor2022] (figshare, CC BY 4.0; file MD5 {R['ge_md5']}). It contains {R['GE_n']} records of {R['GE_comp']} "
              "arc-melted, as-cast alloys built from Hf, Mo, Nb, Re, Ru, Ta, Ti, W and Zr, with Vickers microhardness (500 gf) "
              "reported in GPa and converted to HV (1 HV = 9.80665 MPa). The set was produced in a single laboratory after the "
              f"training data were compiled, shares only {R['GE_comp_overlap']} compositions and {R['GE_sys_overlap']} alloy systems "
              f"with the hardness training set, and {R['GE_Ru']} of its records contain Ru, an element absent from the training "
              "data. It therefore represents a deliberately hard, out-of-domain test of the hardness models."),
        ('h2', '2.2. Featurization'),
        ('p', "Every alloy was described by composition-derived features computed with matminer 0.10 [@ward2018] and pymatgen "
              "[@ong2013]: (i) the atomic fractions of the 29 elements occurring in any of the datasets (FRAC); (ii) 22 alloy-theory "
              "descriptors (PHYS) implemented following Wen et al. [@wen2019], including the valence-electron concentration (VEC) "
              "[@guo2011], the atomic-size mismatch *δ* and the parameter *Ω* [@yang2012], the Miedema-type mixing enthalpy "
              "[@takeuchi2005], the configurational entropy, electronegativity and shear-modulus mismatches, the mean cohesive energy "
              "and a solid-solution strengthening descriptor; and (iii) 132 Magpie descriptors (MAG), i.e. six statistics (minimum, "
              "maximum, range, fraction-weighted mean, average deviation, mode) of 22 elemental properties [@ward2016]. The "
              "processing route was one-hot encoded into six classes (cast, annealed, wrought, powder metallurgy, other, unknown; "
              "PROC). For yield strength, the test temperature and a tension/compression indicator were added (TEST). The default "
              f"input therefore comprised {R['HV_nfeat']} (hardness) or {R['YS_nfeat']} (yield strength) features. Within every "
              "training fold, constant features and one member of every pair with |Pearson *r*| > 0.98 were removed; for models that "
              "require it, features were standardized and test-set features were clipped to the range observed in training, which "
              "prevents unbounded extrapolation of linear and neural models without affecting tree ensembles."),
        ('h2', '2.3. Evaluation protocols'),
        ('p', "Six protocols, ordered by the degree of extrapolation they demand, were applied to identical data (Fig. 2d). "
              "(P1) **Random** five-fold cross-validation at the record level, repeated three times—the protocol used in most "
              "published HEA models. Because repeats are scattered across folds, "
              f"{R['HV_random_overlap']:.0%} (hardness) and {R['YS_random_overlap']:.0%} (yield strength) of test records have the "
              "identical composition in the corresponding training set. (P2) **Composition-grouped** five-fold cross-validation: all "
              "records of a composition fall into the same fold, so the model must predict alloys it has never seen. (P3) "
              "**System-grouped** five-fold cross-validation: all records of an element set (e.g. every Al_{x}CoCrFeNi variant) fall into "
              "the same fold, emulating the exploration of a new alloy system [@meredig2018,roberts2017]. Groups were assigned to "
              "folds at random while balancing fold sizes, and P1–P3 were repeated with three seeds. (P4) **Leave-one-element-out**: "
              "for each element present in at least 40 records and in at most 60% of the dataset, all records containing that element "
              f"were held out ({R['HV_loeo']} tasks for hardness, {R['YS_loeo']} for yield strength), testing transfer to chemistries "
              "in which an element has never been observed [@li2025]. (P5) **Temporal**: models were trained on literature published "
              f"up to 2018 and tested on the 2019 literature ({R['HV_temporal_test']} hardness and {R['YS_temporal_test']} "
              "yield-strength records), mimicking prospective use of a model on the next year of experiments. (P6) **External** "
              "(hardness only): models trained on all MPEA records were applied to the independent GE dataset. All splits were "
              "frozen in a JSON file and used identically by every model."),
        ('h2', '2.4. Models and deployment-specific hyper-parameter selection'),
        ('p', "Nine regressors spanning the families used in the HEA literature were compared: a training-mean baseline; "
              "distance-weighted *k*-nearest neighbours (kNN), the extrapolation reference recommended by Meredig et al. "
              "[@meredig2018]; ridge regression; support-vector regression (SVR, RBF kernel); Gaussian-process regression (GPR, "
              "Matérn-5/2 plus white-noise kernel with hyper-parameters fitted by marginal-likelihood maximization "
              "[@rasmussen2006]); a multilayer perceptron (MLP); random forest (RF) [@breiman2001]; gradient-boosted trees "
              "(XGBoost) [@chen2016]; and TabPFN v2, a transformer pre-trained on synthetic tabular tasks that performs in-context "
              "Bayesian inference without task-specific training [@hollmann2025]. Hyper-parameters of kNN, ridge regression, SVR, "
              "RF, XGBoost and the MLP were selected by nested random search over the spaces of Table S2 (5–13 configurations), "
              "using only the outer training data; TabPFN was used without tuning, with the configuration listed in Table S2b "
              "(open v2 regression checkpoint, eight ensemble members, default inference settings)."),
        ('p', "The inner validation folds were constructed to reproduce the deployment scenario of each outer protocol: random "
              "folds for P1; composition-grouped folds for P2; system-grouped folds for P3; leave-one-element-out folds for P4, "
              "in which each inner fold holds out the not-yet-assigned training records that contain one element (elements "
              "occurring in 5–60% of the training records, taken in order of decreasing frequency) and the corresponding inner "
              "model is trained only on records without that element; and forward-chaining folds for P5, in which each of the "
              "latest publication years of the training literature is predicted by a model trained on all earlier years. Three "
              "inner folds were used for tuning and five for conformal calibration (Section 2.5); records that no inner fold holds "
              "out (possible for the element and forward-chaining schemes) are used for fitting only. For P6 no measurement from "
              "the external laboratory may be used, so its inner scheme was chosen from label-free information: the external "
              "alloys are characterized by elements that are absent from or rare in the training data (Fig. 2b), and "
              "leave-one-element-out folds were therefore used for tuning and calibration, with system-grouped calibration as a "
              "sensitivity analysis. Neither scheme can anticipate a systematic offset between laboratories (Section 3.9). To "
              "assess the effect of the inner scheme on accuracy, the tuned models were also run under P4–P6 with "
              "system-grouped inner folds (Table S16)."),
        ('p', "Classical models were implemented in scikit-learn 1.9 [@pedregosa2011] and XGBoost 3.4 and run on a two-core "
              "workstation; TabPFN (tabpfn 9.2.0) was run on an NVIDIA T4 GPU in a Kaggle notebook. All environments read the "
              "same frozen splits and features. Accuracy was summarized by the root-mean-square error (RMSE), mean absolute "
              "error (MAE), coefficient of determination (*R*^{2}) and Spearman rank correlation (*ρ*). For P1–P3, metrics were "
              "computed on the pooled out-of-fold predictions of each repeat and averaged over the three repeats; for P4, they "
              "were computed per held-out element and averaged; for P5 and P6, they refer to the single test set."),
        ('h2', '2.5. Uncertainty quantification'),
        ('p', "**Cross-conformal intervals (CV+).** Let an inner scheme divide the training records into folds, and let "
              "*μ̂*_{−k} be the model fitted on the training records available to fold *k* (its complement, or for the element and "
              "forward-chaining schemes the records without the held-out element or published before the held-out year). Each "
              "calibration record *i* in fold *k*(*i*) has the out-of-fold residual *R*_{i} = |*y*_{i} − *μ̂*_{−k(i)}(**x**_{i})|, "
              "and *n* denotes the number of calibration records. Following Barber et al. [@barber2021], the interval at a test "
              "point **x** is"),
        ('eq', 'cvplus'),
        ('p', "where *q̂*^{−}_{α}{*v*_{i}} is the ⌊*α*(*n*+1)⌋-th smallest and *q̂*^{+}_{α}{*v*_{i}} the ⌈(1−*α*)(*n*+1)⌉-th "
              "smallest of the *n* values *v*_{i} (−∞ and +∞, respectively, when the index falls outside 1, …, *n*). Each "
              "calibration record thus contributes the test prediction of the same fold model that produced its residual; the "
              "prediction of the model refitted on all training data is used as the point prediction only. If training and test "
              "records are exchangeable and the folds are random, *K*-fold CV+ guarantees a coverage of at least 1 − 2*α* − "
              "min{2(1 − 1/*K*)/(*n*/*K* + 1), (1 − *K*/*n*)/(*K* + 1)} (Theorem 4 of [@barber2021]); empirically, coverage is "
              "usually close to 1 − *α*. We compared **naive CV+**, with *K* = 5 random folds as in standard practice, and "
              "**protocol-matched CV+**, which uses the deployment-specific inner folds of Section 2.4. The rationale is that "
              "residuals obtained under random folds measure interpolation error, whereas residuals obtained under matched folds "
              "measure the error of the kind of extrapolation being evaluated. No finite-sample guarantee applies to the matched "
              "variant, or to any interval under P2–P6, because calibration and test records are not exchangeable by "
              "construction; matched calibration is a heuristic whose coverage we assess empirically."),
        ('p', "**Normalized CV+.** To let interval width grow with distance from the training data, CV+ was also applied to the "
              "scaled score *R*_{i}/*s*_{i}:"),
        ('eq', 'ncvp'),
        ('p', "with *s*_{i} = *d*_{5}(**x**_{i}; *D*_{−k(i)}) + *β*_{−k(i)} and *s*_{−k}(**x**) = *d*_{5}(**x**; *D*_{−k}) + "
              "*β*_{−k}, where *d*_{5} is the applicability-domain distance of Section 2.6 computed in a descriptor space "
              "standardized on the fold training set *D*_{−k}, and *β*_{−k}, the median leave-self-out *d*_{5} within *D*_{−k}, "
              "prevents division by near-zero distances. Every quantity is a fold-specific fit, so Eq. (2) is CV+ with a "
              "different nonconformity score and carries the same exchangeability-based bound."),
        ('p', "**Group-weighted CV+.** In grouped data the exchangeable units are alloy systems rather than records, and large "
              "systems would otherwise dominate the calibration quantiles. Following the CDF-pooling construction of Dunn et al. "
              "[@dunn2023], each calibration record was given the weight 1/*n*_{g}, where *n*_{g} is the number of calibration "
              "records of its alloy system *g*, so that every system contributes unit weight; the upper end of Eq. (1) was replaced "
              "by the smallest value whose cumulative weight reaches (1 − *α*)(*G* + 1), with *G* the number of calibration "
              "systems and unit weight for the test point (and symmetrically for the lower end). This construction is "
              "asymptotically valid as the number of systems grows under two-layer hierarchical exchangeability, but offers no "
              "finite-sample guarantee."),
        ('p', "**Native intervals and evaluation.** The GPR Gaussian predictive interval (*μ̂* ± *z*_{1−α/2}*σ̂*) and the "
              "predicted quantiles of TabPFN were also evaluated. All CV+ variants were computed for XGBoost, CV+ for GPR (both "
              "properties), RF (hardness; yield strength under P1–P3) and TabPFN. Intervals were evaluated by their empirical "
              "coverage at nominal levels of "
              "90% and 68%, their mean width and the interval (Winkler) score [@tran2020], *W* = (*u* − *l*) + (2/*α*)[(*l* − "
              "*y*)_{+} + (*y* − *u*)_{+}], which rewards narrow intervals and penalizes misses in proportion to their size. "
              "Because marginal coverage can hide failures on individual systems, coverage was also computed per alloy system "
              "(systems with at least five test records in a fold or test set). Coverage confidence intervals use the cluster "
              "bootstrap of Section 2.8; TabPFN intervals, evaluated in the GPU environment, are summarized by their pooled "
              "coverage only."),
        ('h2', '2.6. Applicability domain and screening analysis'),
        ('p', "As a model-agnostic applicability-domain (AD) score [@sutton2020,janet2019], we used the mean Euclidean distance "
              "from a test alloy to its *k* = 5 nearest training alloys in the composition-feature space (FRAC, PHYS and MAG "
              "blocks), standardized on the training fold and winsorized at |*z*| ≤ 5 so that near-constant descriptors (e.g. the "
              "fraction of an element present in only one or two training alloys) cannot dominate the distance:"),
        ('eq', 'ad'),
        ('p', "The association between *d*_{5} and the absolute error was quantified by Spearman's *ρ*, by coverage within AD "
              "tertiles and by error–retention curves, which report the RMSE of the subset of predictions retained when the most "
              "doubtful predictions (largest *d*_{5} or widest interval) are discarded first [@hirschfeld2020]. The sensitivity of "
              "these results to the neighbourhood size (*k* = 1, 3, 5, 10, 20) and to the representation (unwinsorized "
              "descriptors, alloy-theory or Magpie descriptors alone, element fractions with the L1 distance, and the first ten "
              "principal components) was evaluated in a full factorial design (Table S11). To assess the value of the models for "
              "alloy selection rather than for absolute prediction, we computed (i) the within-system Spearman *ρ* between "
              "measured and predicted values for every held-out alloy system with at least five records in P3, and (ii) the "
              "recall of the true top 20% of alloys in the temporal and external test sets when candidates were ranked by the "
              "point prediction, by the upper bound (optimistic, exploration-oriented) or by the lower bound (conservative) of "
              "the protocol-matched 68% CV+ interval."),
        ('h2', '2.7. Feature ablation and interpretation'),
        ('p', "The contribution of each feature block was assessed on the frozen splits with XGBoost (fixed hyper-parameters: "
              "500 trees, depth 5, learning rate 0.05, subsample 0.8, column fraction 0.5), GPR and kNN (*k* = 5), comparing "
              "fractions only, PHYS only, MAG only, PHYS+MAG, all composition blocks with and without processing, and the default "
              "inputs augmented with the observed phase. Feature attributions were computed with TreeSHAP [@lundberg2020] for an "
              "XGBoost model trained on all data with the hyper-parameters most frequently selected under P3. Their robustness was "
              "measured by the mean pairwise Spearman correlation of the mean |SHAP| vectors of the 15 models trained on the P3 "
              "training folds, and by the fraction of those models in which a feature ranked among the ten most important."),
        ('h2', '2.8. Statistical inference under dependence'),
        ('p', "Records of the same alloy system are strongly correlated, and the folds of repeated cross-validation share "
              "training data, so neither records nor folds are independent observations. All uncertainty statements therefore "
              "use a cluster bootstrap with alloy systems as the resampling units [@field2007]: in each of 2,000 replicates the "
              "systems of the test records were drawn with replacement, and every record of a drawn system entered with a "
              "multiplicity equal to its number of draws. The same draws were applied to all models (paired comparisons) and to "
              "all protocols of a dataset; for P1–P3 they re-weight the three repeats simultaneously, for P4 all element tasks, "
              "and for P6 the external systems were resampled. We report percentile 95% intervals for the RMSE, for the "
              "difference in RMSE between each model and the most accurate model (two models are called indistinguishable when "
              "this interval contains zero), for Kendall's *τ* between model rankings and for interval coverage. Fold-level "
              "significance tests that treat correlated folds as independent observations were not used."),
        ('h2', '2.9. Reproducibility'),
        ('p', "All analyses were scripted in Python 3.13 with fixed random seeds for every split, inner fold and model, and "
              "per-record predictions of every model were stored. TabPFN predictions were regenerated in a second GPU run with "
              "the same seeds, which reproduced the per-fold RMSEs of the first run (Supplementary Section S7). Curated datasets, "
              "features, frozen splits, per-record predictions and intervals, notebooks and code are provided as Supplementary "
              "Material and in the public repository listed under Data availability."),
    ]

"""Cover letter (.docx) for the Next Materials special issue submission."""
from assemble import build
from text_front import TITLE


def main(out="/home/claude/hea/manuscript/out/cover_letter.docx"):
    from res import Res
    X = Res()
    hv_r = X.best('HV', 'random')[1]; hv_s = X.best('HV', 'system')[1]; hv_t = X.best('HV', 'temporal')[1]
    bests = [X.best(ds, p)[0] for ds in ('HV', 'YS') for p in ('random', 'composition', 'system', 'element', 'temporal', 'external')
             if p in set(X.perf[X.perf.dataset == ds].protocol)]
    tab_w, n_c = bests.count('TabPFN'), len(bests)
    n_t = X.u('HV', 'temporal', 'XGB', 'CV+naive90'); m_s = X.u('HV', 'system', 'XGB', 'CV+matched90')
    B = [
        ('plain', "[Date]", {'align': 'left'}),
        ('plain', "The Editors, *Next Materials*", {'align': 'left', 'after': 0}),
        ('plain', "Special Issue: Artificial Intelligence in Materials Engineering", {'align': 'left', 'after': 240}),
        ('plain', "Dear Editors,", {'align': 'left'}),
        ('p', f"We are pleased to submit our Research Article entitled \"{TITLE}\" for consideration in the special issue on "
              "Artificial Intelligence in Materials Engineering (article type: VSI: A.I. in materials)."),
        ('p', "Machine-learning models are increasingly used to propose new high-entropy alloys, but their accuracy is almost always "
              "reported with random cross-validation, which mainly measures interpolation among closely related literature alloys. "
              "Our manuscript quantifies, on open data, how much this overstates the accuracy that matters for alloy design. We "
              "evaluated nine regressors—including a pre-trained tabular foundation model—on 501 hardness and 1,067 yield-strength "
              "records of multi-principal element alloys under six protocols of increasing novelty (new composition, new alloy "
              "system, unseen element, the following year of literature and an independent laboratory), and we studied whether "
              "conformal prediction intervals and applicability-domain scores remain trustworthy under these shifts."),
        ('p', f"The main findings are: (i) errors grow steadily with novelty—the best hardness RMSE rises from {hv_r:.0f} HV "
              f"under random splitting to {hv_s:.0f} HV for unseen alloy systems and {hv_t:.0f} HV for the following year of "
              f"literature—and every model fails on an external set rich in elements absent from training; (ii) a pre-trained "
              f"tabular foundation model (TabPFN) had the lowest error in {tab_w} of {n_c} property–protocol combinations "
              f"without any tuning, although most differences from tree ensembles and Gaussian processes were within "
              f"statistical uncertainty, and it did not escape the extrapolation penalty; (iii) conformal intervals "
              f"calibrated on random folds under-covered under shift (nominal 90% XGBoost intervals covered {n_t:.0%} of the "
              f"next year's hardness data), and calibration folds that reproduce the deployment scenario—grouped, "
              f"leave-one-element-out or forward-chaining—raised empirical coverage to {m_s:.0%} for new systems, although no "
              f"guarantee holds under such shifts and the inter-laboratory shift remained under-covered; and (iv) a "
              f"nearest-neighbour applicability-domain distance flags population-level shift but is a weak predictor of "
              f"individual errors. All comparisons use confidence intervals from a cluster bootstrap over alloy systems. From "
              f"these results we derive concrete reporting recommendations for ML-guided alloy design, and we release the "
              f"curated data, frozen splits, all predictions and code as a reusable benchmark "
              f"(https://github.com/labidz/hea_mpea_benchmark)."),
        ('p', "We believe the work fits the special issue's focus on AI methodologies, predictive modelling and design of metallic "
              "and engineering materials, and that it will interest the broad readership of *Next Materials*, because the "
              "evaluation and uncertainty-calibration practices it establishes apply to data-driven materials design in general."),
        ('p', "This manuscript is original, has not been published previously and is not under consideration elsewhere. All "
              "authors have approved the submission and declare no competing interests. The use of generative AI tools in "
              "preparing the manuscript is disclosed in the dedicated declaration section, in line with Elsevier policy."),
        ('plain', "Sincerely,", {'align': 'left'}),
        ('plain', "Rohul Amin (corresponding author), on behalf of all authors", {'align': 'left', 'after': 0}),
        ('plain', "Department of Computer Science and Engineering, Bangladesh Army University of Science and Technology, Khulna, Bangladesh", {'align': 'left', 'after': 0}),
        ('plain', "E-mail: [corresponding author e-mail]", {'align': 'left'}),
    ]
    build(B, {'creator': 'Rohul Amin', 'title': 'Cover letter', 'description': 'Cover letter'}, out.replace('.docx', '.json'), out)


if __name__ == '__main__':
    main()

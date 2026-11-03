"""Build the manuscript .docx (and the SI) from text modules, result tables and figures."""
import sys
import json
import pandas as pd
from PIL import Image
from assemble import build
from collect import data_stats
from res import Res
from intro_text import intro
from methods_text import methods
import results_text as RT
from tables import table1, table2, table3
from backmatter import backmatter, acknowledgements
from discussion_static import LIMITATIONS, WHATS_NEXT

FIG = "/home/claude/hea/figures"
AUTHORS = "Rohul Amin^{a,}∗, Md. Mehedi Rahman Rana^{a}, Sumya Akter^{a}"
AFFIL = "^{a}Department of Computer Science and Engineering, Bangladesh Army University of Science and Technology, Khulna, Bangladesh"


def fig(path, caption, width_mm=170, alt=''):
    im = Image.open(path)
    return ('figure', dict(path=path, width_mm=width_mm, aspect=im.size[1] / im.size[0], caption=caption, alt=alt or caption[:120]))


def front(title, abstract, keywords, highlights=None):
    B = [('title', title),
         ('plain', AUTHORS, {'align': 'left', 'after': 60}),
         ('plain', AFFIL, {'align': 'left', 'size': 20, 'italic': True, 'after': 60}),
         ('plain', "∗Corresponding author. E-mail address: [corresponding author e-mail]", {'align': 'left', 'size': 20, 'after': 240})]
    if highlights:
        B.append(('h1', 'Highlights'))
        B += [('bullet', h) for h in highlights]
    B += [('h1', 'Abstract'), ('p', abstract), ('plain', f"**Keywords:** {keywords}", {'align': 'left', 'after': 240})]
    return B


def main(out="/home/claude/hea/manuscript/out/manuscript.docx", draft=False):
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    R = data_stats(); X = Res()
    from text_front import TITLE, ABSTRACT, KEYWORDS, HIGHLIGHTS
    from text_end import discussion, conclusions
    abstract = ABSTRACT(R, X)
    B = front(TITLE, abstract, KEYWORDS, HIGHLIGHTS(R, X))
    B += intro(R)
    M = methods(R)
    B += M[:2] + [fig(f"{FIG}/Fig0_workflow.png", FIGCAP['1'])] + M[2:]
    B += [('table', table1(R))]
    B += [fig(f"{FIG}/Fig1_data_protocols.png", FIGCAP['2'])]
    Rb = RT.results(R, X)
    k = next(i for i, b in enumerate(Rb) if b[0] == 'p' and b[1].startswith('Fig. 3 shows how'))
    B += Rb[:k + 1] + [fig(f"{FIG}/Fig0b_near_duplicates.png", FIGCAP['3'])] + Rb[k + 1:]
    B += [('table', table2(X.perf, X.pair))]
    B += [fig(f"{FIG}/Fig2_performance_protocols.png", FIGCAP['4'])]
    B += [fig(f"{FIG}/Fig3_leave_one_element_out.png", FIGCAP['5'])]
    B += [fig(f"{FIG}/Fig4_parity_temporal_external.png", FIGCAP['6'])]
    B += RT.results2(R, X)
    B += [('table', table3(X.ablation_summary))]
    B += RT.results3(R, X)
    B += [fig(f"{FIG}/Fig5_uncertainty_coverage90.png", FIGCAP['7'])]
    B += [fig(f"{FIG}/Fig0c_calibration_mechanism.png", FIGCAP['8'])]
    B += RT.results4(R, X)
    B += [fig(f"{FIG}/Fig6_applicability_domain.png", FIGCAP['9'])]
    B += RT.results5(R, X)
    B += [fig(f"{FIG}/Fig7_shap.png", FIGCAP['10'])]
    B += discussion(R, X, LIMITATIONS)
    B += conclusions(R, X)
    B += [('h1', "What's Next"), ('p', WHATS_NEXT)]
    B += backmatter() + acknowledgements()
    B += [('references',)]
    order = build(B, {'creator': 'Rohul Amin', 'title': TITLE, 'description': 'Manuscript submitted to Next Materials'},
                  out.replace('.docx', '.json'), out)
    print('references cited:', len(order))
    return order


FIGCAP = {
    '1': "**Fig. 1.** Study design. (1) Hardness and yield-strength records were curated from the open MPEA database and an "
         "independent refractory-alloy hardness set from a different laboratory. (2) Alloys were described by element "
         "fractions, alloy-theory descriptors, Magpie statistics, processing class and, for yield strength, test conditions. "
         "(3) Six protocols of increasing extrapolation severity were applied with identical frozen splits. (4) Nine "
         "regressors were tuned with inner folds that reproduce each deployment scenario (random, composition- or "
         "system-grouped, leave-one-element-out, forward-chaining). (5) Prediction intervals were obtained by naive, "
         "deployment-matched, distance-normalized and group-weighted CV+ and from model-native uncertainty. (6) Accuracy, "
         "calibration, applicability domain, screening value and feature attributions were analysed, with alloy systems "
         "as the resampling unit for all confidence intervals.",
    '3': "**Fig. 3.** Why random splitting is optimistic. (a) Hardness of the Al–Co–Cr–Fe–Ni family versus Al content; filled "
         "symbols are the test records of one random fold, open symbols its training records (small horizontal jitter for "
         "visibility). (b,c) Cumulative distribution of the composition change (half the L1 distance between atomic-fraction "
         "vectors, in at.%) from each test record to its nearest training record, for every protocol; dots mark the medians. "
         "(d) XGBoost RMSE under random splitting for test records whose exact composition is or is not present in the "
         "training fold, compared with composition- and system-grouped validation; bar labels give the RMSE in HV or MPa.",
    '8': "**Fig. 8.** Why deployment-matched calibration changes conformal intervals. (a) Schematic of the calibration folds "
         "used by naive CV+ (random alloys held out) and matched CV+ (here for P3: whole alloy systems held out). (b,c) Bars: "
         "interval half-width needed for 90% coverage, i.e. the 90th percentile of the absolute XGBoost test error pooled over "
         "folds; lines: mean half-width supplied by naive, deployment-matched and distance-normalized CV+. Where a line lies "
         "below its bar, the intervals under-cover. P1–P6 as in Fig. 1.",
    '2': "**Fig. 2.** Data and evaluation protocols. (a) Records per publication year for the two MPEA datasets; the dashed "
         "line marks the temporal split. (b) Fraction of records containing each element in the training datasets and in the "
         "external GE set, which is dominated by Re, Ru and refractory elements. (c) Hardness distributions of the MPEA and "
         "external datasets. (d) Test records (coloured) of one split of each protocol, with the 501 hardness records ordered by "
         "alloy system; the bar under the strips marks the 134 systems. Random folds scatter near-copies of every alloy across "
         "training and test, whereas grouped, element and temporal protocols remove whole compositions, systems or chemistries.",
    '4': "**Fig. 4.** Test RMSE (logarithmic axis) of each model under the evaluation protocols for (a) hardness and (b) "
         "yield strength; colour darkens with increasing extrapolation severity. Error bars: 95% confidence intervals from "
         "a cluster bootstrap over alloy systems (2,000 replicates). The dashed line marks the RMSE of the training-mean predictor under random splitting.",
    '5': "**Fig. 5.** Leave-one-element-out *R*^{2} of every model for (a) hardness and (b) yield strength. Columns are "
         "held-out elements with the number of test records; cell labels give the actual *R*^{2}, and the colour scale is "
         "clipped to [−1, 1].",
    '6': "**Fig. 6.** Measured versus predicted values of XGBoost for prospective-style tests: (a) hardness of alloys "
         "published in 2019, (b) hardness of the independent GE refractory alloys and (c) yield strength of alloys published "
         "in 2019. Grey bars: deployment-matched 90% CV+ intervals (forward-chaining calibration for the 2019 literature, "
         "leave-one-element-out calibration for the external alloys); colour: applicability-domain distance *d*_{5}; "
         "dashed line: parity.",
    '7': "**Fig. 7.** Empirical calibration of 90% prediction intervals under the evaluation protocols. (a,c) Empirical "
         "coverage with cluster-bootstrap 95% confidence intervals (no intervals for TabPFN, which is summarized from "
         "per-fold results); the dashed line is the nominal level. (b,d) Mean interval width relative to the standard "
         "deviation of the target. Matched: calibration folds that reproduce the deployment scenario (Section 2.4).",
    '9': "**Fig. 9.** Applicability domain (XGBoost). (a) Absolute error normalized by the target SD versus AD distance "
         "*d*_{5}, pooled over all non-random protocols (lines: decile medians; bands: inter-quartile range; *ρ*: pooled "
         "Spearman correlation). (b) Coverage of 90% intervals within AD tertiles, pooled over non-random protocols. (c) "
         "Error–retention curves for the shifted protocols P3–P5 when predictions are discarded in order of decreasing *d*_{5}, "
         "decreasing normalized-interval width, or decreasing true error (oracle).",
    '10': "**Fig. 10.** SHAP interpretation of the XGBoost models. (a,c) Mean |SHAP| of the twelve most important features "
         "(bars: model trained on all data; error bars: mean ± SD over the 15 system-grouped training folds; percentages: "
         "fraction of fold models in which the feature ranks among the top ten). (b) Dependence of the hardness prediction on "
         "the most important alloy-theory descriptor; (d) dependence of the yield-strength prediction on test temperature.",
}

if __name__ == '__main__':
    main()

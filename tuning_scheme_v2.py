"""Revision 2: effect of the inner-validation scheme on test accuracy under P4-P6.
Compares the test RMSE of every tuned model when its hyper-parameters were selected with
  (a) deployment-specific inner folds (leave-element-out for P4/P6, forward-chaining for P5; results/final), and
  (b) system-grouped inner folds (the P3 scheme; first-run predictions results/{DS}_pred.csv, results/HV_external_pred.csv,
      and for the yield-strength GPR/RF/MLP the per-split metrics results/ext_metrics.csv).
Ridge is excluded because its first-run tuning (leave-one-out RidgeCV over three penalties) differs in the search space as
well; GPR is excluded because its kernel hyper-parameters are fitted by marginal likelihood under both schemes.
Metric definitions as in the paper: P4 mean over held-out elements of the per-element RMSE; P5/P6 RMSE of the test set.
Output: results/tables/tuning_scheme_v2.csv
"""
import numpy as np
import pandas as pd

R = "/home/claude/hea/results"
MODELS = ['kNN', 'SVR', 'MLP', 'RF', 'XGB']


def rmse_by(d):
    out = {}
    for (p, m), g in d.groupby(['protocol', 'model']):
        if p == 'element':
            out[(p, m)] = np.mean([np.sqrt(((h.yhat - h.y) ** 2).mean()) for _, h in g.groupby('label')])
        elif p in ('temporal', 'external'):
            out[(p, m)] = np.sqrt(((g.yhat - g.y) ** 2).mean())
    return out


def main():
    rows = []
    ext = pd.read_csv(f"{R}/ext_metrics.csv")
    for ds in ('HV', 'YS'):
        v1 = pd.read_csv(f"{R}/{ds}_pred.csv", usecols=['protocol', 'label', 'rid', 'model', 'y', 'yhat'])
        if ds == 'HV':
            e = pd.read_csv(f"{R}/HV_external_pred.csv", usecols=['protocol', 'label', 'rid', 'model', 'y', 'yhat'])
            e['protocol'] = 'external'
            v1 = pd.concat([v1, e])
        a = rmse_by(v1[v1.model.isin(MODELS)])
        # yield-strength models of the first run that exist only as per-split metrics
        x = ext[(ext.dataset == ds) & ext.model.isin(MODELS) & ext.protocol.isin(['element', 'temporal'])]
        for (p, m), g in x.groupby(['protocol', 'model']):
            a.setdefault((p, m), float(g.RMSE.mean()))
        v2 = pd.read_csv(f"{R}/final/{ds}_acc.csv", usecols=['protocol', 'label', 'rid', 'model', 'y', 'yhat'])
        b = rmse_by(v2[v2.model.isin(MODELS)])
        for k in sorted(set(a) & set(b)):
            rows.append(dict(dataset=ds, protocol=k[0], model=k[1], rmse_system_grouped=a[k], rmse_deployment=b[k],
                             change_pct=100 * (b[k] / a[k] - 1)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{R}/tables/tuning_scheme_v2.csv", index=False)
    print(df.round(1).to_string())
    print('change range', df.change_pct.min().round(1), df.change_pct.max().round(1), 'median', df.change_pct.median().round(1),
          'n improved', int((df.change_pct < 0).sum()), 'of', len(df))


if __name__ == '__main__':
    main()

"""Accessors for result tables used by the results text (every number in the paper comes through here)."""
import json
import pandas as pd

T = "/home/claude/hea/results/tables_v2"


class Res:
    def __init__(self):
        self.perf = pd.read_csv(f"{T}/perf_summary.csv")
        self.folds = pd.read_csv(f"{T}/perf_folds.csv")
        self.uq = pd.read_csv(f"{T}/uq_summary.csv")
        self.uqad = pd.read_csv(f"{T}/uq_by_ad.csv")
        self.adc = pd.read_csv(f"{T}/ad_corr.csv")
        self.pair = pd.read_csv(f"{T}/pairwise.csv")
        self.rank = pd.read_csv(f"{T}/ranking.csv")
        for name in ('ad_variants', 'within_system_rho', 'topk_recall', 'ablation_summary', 'dedup_sensitivity',
                     'external_subsets_v2', 'ad_sensitivity_v2', 'temporal_overlap_v2', 'uq_system_cov', 'tuning_scheme_v2',
                     'nn_comp_distance_summary', 'random_seen_unseen'):
            try:
                setattr(self, name, pd.read_csv(f"{T}/{name}.csv"))
            except FileNotFoundError:
                setattr(self, name, None)
        self.noise = json.load(open(f"{T}/noise_floor.json"))
        self.noise2 = json.load(open(f"{T}/noise_floor_v2.json"))
        self.tabcfg = json.load(open(f"{T}/tabpfn_config.json"))
        dd = self.dedup_sensitivity
        g = dd.groupby(['data', 'protocol', 'model']).RMSE.mean()
        self.dd = dict(n_dedup=int(dd[dd.data == 'deduplicated'].n.iloc[0]), orig_random=g[('original', 'random', 'XGB')],
                       dedup_random=g[('deduplicated', 'random', 'XGB')], orig_system=g[('original', 'system', 'XGB')],
                       dedup_system=g[('deduplicated', 'system', 'XGB')])
        self._derived()
        x = self.adc[self.adc.model == 'XGB']
        self.ad_median = {(r.dataset, r.protocol): float(r.median_ad) for r in x.itertuples()}
        self.ad_rho = {(r.dataset, r.protocol): float(r.spearman_ad_abs_err) for r in x.itertuples()}

    def p(self, ds, proto, model, metric='RMSE'):
        s = self.perf[(self.perf.dataset == ds) & (self.perf.protocol == proto) & (self.perf.model == model)]
        if s.empty:
            raise KeyError((ds, proto, model))
        return float(s[metric].iloc[0])

    def best(self, ds, proto, metric='RMSE', exclude=('Mean',)):
        s = self.perf[(self.perf.dataset == ds) & (self.perf.protocol == proto) & (~self.perf.model.isin(exclude))]
        s = s.sort_values(metric, ascending=(metric in ('RMSE', 'MAE')))
        return s.model.iloc[0], float(s[metric].iloc[0])

    def u(self, ds, proto, model, method, metric='coverage'):
        s = self.uq[(self.uq.dataset == ds) & (self.uq.protocol == proto) & (self.uq.model == model) & (self.uq.method == method)]
        if s.empty:
            raise KeyError((ds, proto, model, method))
        return float(s[metric].iloc[0])

    def ci(self, ds, proto, model):
        s = self.perf[(self.perf.dataset == ds) & (self.perf.protocol == proto) & (self.perf.model == model)].iloc[0]
        return float(s.RMSE_lo), float(s.RMSE_hi)

    def indist(self, ds, proto):
        """Models whose paired cluster-bootstrap RMSE difference to the most accurate model has a CI containing 0."""
        g = self.pair[(self.pair.dataset == ds) & (self.pair.protocol == proto)]
        best = g.best.iloc[0]
        return best, [m for m, lo in zip(g.model, g.diff_lo) if m != best and lo <= 0]

    def ucov(self, ds, proto, model, method):
        s = self.uq[(self.uq.dataset == ds) & (self.uq.protocol == proto) & (self.uq.model == model) & (self.uq.method == method)].iloc[0]
        return float(s.coverage), float(s.get('cov_lo', float('nan'))), float(s.get('cov_hi', float('nan')))

    def models(self, ds, proto):
        return list(self.perf[(self.perf.dataset == ds) & (self.perf.protocol == proto)].model)

    def _derived(self):
        """Retention-curve points and temporal AD diagnostics (XGB), computed from the prediction files."""
        import sys, numpy as np
        sys.path.insert(0, '/home/claude/hea')
        from analysis_v2 import load_all
        self.retention, self.temporal_ad = {}, {}
        for ds in ('HV', 'YS'):
            d = load_all(ds); x = d[d.model == 'XGB']
            h = x[x.protocol.isin(['system', 'element', 'temporal'])]
            sd = pd.read_csv(f"/home/claude/hea/data/{'hv' if ds == 'HV' else 'ys'}_records.csv").target.std()
            e2 = (((h.yhat - h.y) / sd) ** 2).values
            def keep(score, f=0.5):
                o = np.argsort(np.asarray(score)); k = int(f * len(o)); return float(np.sqrt(e2[o[:k]].mean()))
            self.retention[ds] = dict(full=float(np.sqrt(e2.mean())), ad=keep(h.ad_knn), width=keep(h.ncp_matched_hw90), oracle=keep(e2))
            if ds == 'HV':
                rec = pd.read_csv('/home/claude/hea/data/hv_records.csv').set_index('rid')
                t = x[x.protocol == 'temporal'].copy(); t['proc'] = t.rid.map(rec.proc)
                t['tert'] = pd.qcut(t.ad_knn, 3, labels=[0, 1, 2])
                u = t[t.proc == 'UNKNOWN']
                self.temporal_ad = dict(near_unknown=float((t[t.tert == 0].proc == 'UNKNOWN').mean()),
                                        far_unknown=float((t[t.tert == 2].proc == 'UNKNOWN').mean()),
                                        unknown_bias=float((u.yhat - u.y).mean()), unknown_n=len(u))

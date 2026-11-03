"""Runner for Kaggle CPU sessions (executed inside the notebook after common.py and models.py cells).
Runs `MODELS` on yield strength over the frozen splits with the same tuning/CV+ code as bench.py and prints
compact per-fold metrics (M|), within-system Spearman (W|) and top-quintile recall on the temporal split (K|)."""
import json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def run_kaggle(MODELS, UQ, DATA, ds='YS'):
    rec, Xdf = load(ds, DATA)
    cols = feature_columns(Xdf, DEFAULT_BLOCKS)
    X = Xdf[cols].values; y = rec.target.values.astype(float)
    rid2i = {r: i for i, r in enumerate(rec.rid.values)}
    splits = json.load(open(f"{DATA}/splits.json"))[ds]
    comp_cols = feature_columns(Xdf, ('FRAC', 'PHYS', 'MAG'))
    Xc = Xdf[comp_cols].values
    allrows = []
    for sp in splits:
        te = np.array([rid2i[r] for r in sp['test']]); mask = np.ones(len(y), bool); mask[te] = False; tr = np.where(mask)[0]
        rtr = rec.iloc[tr]; grp = inner_groups(rtr, sp['protocol'])
        seed = (sum(map(ord, sp['label'])) * 7919) % (2 ** 31)
        pc = Prep(clip=False).fit(Xc[tr]); Ztr, Zte = pc.transform(Xc[tr]), pc.transform(Xc[te])
        for name in MODELS:
            t0 = time.time()
            p, _ = tune(name, X[tr], y[tr], grp, seed)
            m = build(name, p, seed).fit(X[tr], y[tr])
            out = pd.DataFrame({'label': sp['label'], 'protocol': sp['protocol'], 'rep': sp['rep'], 'rid': rec.rid.values[te],
                                'model': name, 'y': y[te]})
            if name == 'GPR':
                out['yhat'], out['gp_sd'] = m.predict(X[te], return_std=True)
            else:
                out['yhat'] = m.predict(X[te])
            if name in UQ:
                variants = [('naive', None)] + ([('matched', grp)] if sp['protocol'] != 'random' else [])
                for vn, g in variants:
                    iv, _ = cv_plus(name, p, X[tr], y[tr], X[te], g, seed, alphas=(0.1, 0.32), Zall=(Ztr, Zte))
                    for a in (0.1, 0.32):
                        tag = int(round((1 - a) * 100)); out[f'cvp_{vn}_lo{tag}'], out[f'cvp_{vn}_hi{tag}'] = iv[a]
            allrows.append(out)
            e = out.yhat - out.y
            print(f"{sp['label']:<22} {name:<5} RMSE={np.sqrt((e**2).mean()):7.1f} t={time.time()-t0:5.1f}s params={p}", flush=True)
    D = pd.concat(allrows, ignore_index=True)
    D.to_csv(f"/kaggle/working/{ds}_{'_'.join(MODELS)}_pred.csv", index=False)
    cov = lambda d, lo, hi: float(((d.y >= d[lo]) & (d.y <= d[hi])).mean())
    lines = ["M|model|label|protocol|n|MAE|RMSE|R2|rho|bias|cov_gp90|w_gp90|cov_gp68|w_gp68|cov_cvn90|w_cvn90|cov_cvn68|w_cvn68|cov_cvm90|w_cvm90|cov_cvm68|w_cvm68"]
    for (model, lab), d in D.groupby(['model', 'label'], sort=False):
        e = d.yhat - d.y; ss = ((d.y - d.y.mean()) ** 2).sum()
        v = [len(d), e.abs().mean(), np.sqrt((e ** 2).mean()), 1 - (e ** 2).sum() / ss, spearmanr(d.y, d.yhat).correlation, e.mean()]
        if model == 'GPR':
            for t, z in ((90, 1.6449), (68, 0.9945)):
                lo, hi = d.yhat - z * d.gp_sd, d.yhat + z * d.gp_sd
                v += [float(((d.y >= lo) & (d.y <= hi)).mean()), float((hi - lo).mean())]
        else:
            v += [np.nan] * 4
        for vn in ('naive', 'matched'):
            for t in (90, 68):
                lo, hi = f'cvp_{vn}_lo{t}', f'cvp_{vn}_hi{t}'
                if lo in d and d[lo].notna().all():
                    v += [cov(d, lo, hi), float((d[hi] - d[lo]).mean())]
                else:
                    v += [np.nan, np.nan]
        lines.append("M|" + "|".join([model, lab, d.protocol.iloc[0]] + [f"{x:.4f}" if isinstance(x, float) else str(x) for x in v]))
    # within-system Spearman (system protocol; room-temperature compression records only)
    cond = rec.set_index('rid')
    S = D[D.protocol == 'system'].copy(); S['sys'] = S.rid.map(cond.sys_key)
    S = S[(S.rid.map(cond.test_T) == 25) & (S.rid.map(cond.test) == 'C')]
    for (model, rep, sk), s in S.groupby(['model', 'rep', 'sys']):
        if len(s) >= 5 and s.y.std() > 0:
            lines.append(f"W|{model}|{rep}|{sk}|{len(s)}|{spearmanr(s.y, s.yhat).correlation:.4f}")
    # top-quintile recall on the temporal split
    for model, g in D[D.label == 'temporal-2019'].groupby('model'):
        k = max(1, int(round(0.2 * len(g)))); top = set(g.nlargest(k, 'y').rid)
        scores = {'mean': g.yhat}
        if 'cvp_matched_lo68' in g and g.cvp_matched_lo68.notna().all():
            scores['UCB'] = g.cvp_matched_hi68; scores['LCB'] = g.cvp_matched_lo68
        for sn, s in scores.items():
            sel = set(g.assign(s=s.values).nlargest(k, 's').rid)
            lines.append(f"K|{model}|temporal-2019|{sn}|{k}|{len(g)}|{len(sel & top) / k:.4f}")
    print('=====RES_BEGIN=====')
    print("\n".join(lines))
    print('=====RES_END=====')

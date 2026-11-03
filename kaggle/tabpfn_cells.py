CELLS = [
r'''# TabPFN v2 benchmark on frozen splits (GPU). Data: Kaggle dataset with curated MPEA records,
# composition features (matminer, computed offline) and frozen outer splits (splits.json).
import subprocess, sys, os, json, time, glob
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn==9.2.0"], capture_output=True, text=True)
print("PIP rc", r.returncode, r.stderr[-300:])
import numpy as np, pandas as pd, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
import tabpfn; print("tabpfn", tabpfn.__version__)
cands = glob.glob("/kaggle/input/**/splits.json", recursive=True)
DATA = os.path.dirname(cands[0]); print("DATA", DATA, sorted(os.listdir(DATA)))
''',
r'''PROC_LEVELS = ['CAST', 'ANNEAL', 'WROUGHT', 'POWDER', 'OTHER', 'UNKNOWN']
DROP_PHYS = ['PHYS:Weight Fraction', 'PHYS:Atomic Fraction', 'PHYS:Total weight']
def load(ds):
    fname = {'HV': 'hv_records.csv', 'YS': 'ys_records.csv', 'GE': 'ext_ge_records.csv'}[ds]
    rec = pd.read_csv(f"{DATA}/{fname}")
    F = pd.read_csv(f"{DATA}/composition_features.csv")
    F = F.drop(columns=[c for c in DROP_PHYS + ['PHYS_FAIL'] if c in F.columns])
    X = rec[['comp_key']].merge(F, on='comp_key', how='left').drop(columns=['comp_key'])
    assert len(X) == len(rec) and X.notna().all().all()
    for p in PROC_LEVELS:
        X[f"PROC:{p}"] = (rec.proc == p).astype(float).values
    if ds == 'YS':
        X['TEST:T_C'] = rec.test_T.values
        X['TEST:tension'] = (rec.test == 'T').astype(float).values
    X.index = rec.rid.values
    return rec, X
def feature_columns(X, blocks=('FRAC', 'PHYS', 'MAG', 'PROC', 'TEST')):
    return [c for c in X.columns if c.split(':')[0] in blocks]
def balanced_group_folds(groups, n_splits, rng):
    groups = np.asarray(groups); ug, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(ug)); order = order[np.argsort(-counts[order], kind='stable')]
    load_ = np.zeros(n_splits); gfold = {}
    for gi in order:
        f = int(np.argmin(load_ + rng.random(n_splits) * 1e-6)); gfold[ug[gi]] = f; load_[f] += counts[gi]
    return np.array([gfold[g] for g in groups])
def inner_folds(n, groups, k, seed):
    rng = np.random.default_rng(seed)
    if groups is None:
        f = np.empty(n, int); f[rng.permutation(n)] = np.arange(n) % k; return f
    return balanced_group_folds(groups, k, rng)
def inner_groups(rtr, protocol):
    if protocol == 'random': return None
    if protocol == 'composition': return rtr.comp_key.values
    return rtr.sys_key.values
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
def make_model(seed):
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", random_state=int(seed) % 10000)
QS = [0.05, 0.16, 0.84, 0.95]
def fit_predict(Xtr, ytr, Xte, seed, quant=False):
    keep = Xtr.std(0) > 1e-12
    m = make_model(seed); m.fit(Xtr[:, keep], ytr)
    mu = m.predict(Xte[:, keep])
    if quant:
        q = np.asarray(m.predict(Xte[:, keep], output_type="quantiles", quantiles=QS))
        return mu, q
    return mu
def cv_plus(Xtr, ytr, Xte, groups, seed, alphas=(0.1, 0.32), k=5):
    fid = inner_folds(len(ytr), groups, k, seed + 11)
    mu_te = np.zeros((len(ytr), len(Xte))); R = np.zeros(len(ytr))
    for f in range(k):
        tr, va = fid != f, fid == f
        pr = fit_predict(Xtr[tr], ytr[tr], np.vstack([Xtr[va], Xte]), seed)
        R[va] = np.abs(pr[:va.sum()] - ytr[va]); mu_te[va] = pr[va.sum():][None, :]
    out = {}; n = len(ytr)
    for a in alphas:
        lo_k = max(int(np.floor(a * (n + 1))), 1); hi_k = min(int(np.ceil((1 - a) * (n + 1))), n)
        out[a] = (np.sort(mu_te - R[:, None], 0)[lo_k - 1], np.sort(mu_te + R[:, None], 0)[hi_k - 1])
    return out
print("ok")
''',
r'''from scipy.stats import spearmanr
splits = json.load(open(f"{DATA}/splits.json"))
os.makedirs("/kaggle/working", exist_ok=True)
def run_dataset(ds):
    rec, Xdf = load(ds); cols = feature_columns(Xdf); X = Xdf[cols].values.astype(np.float32); y = rec.target.values.astype(float)
    rid2i = {r: i for i, r in enumerate(rec.rid.values)}
    jobs = list(splits[ds])
    if ds == 'HV':
        jobs.append(dict(label='external-GE', protocol='external', rep=0, fold=0, test=None))
    rows = []
    for sp in jobs:
        t0 = time.time()
        if sp['protocol'] == 'external':
            rge, Xge = load('GE'); Xte = Xge[cols].values.astype(np.float32); yte = rge.target.values.astype(float)
            tr = np.arange(len(y)); rid_te = rge.rid.values; proto_for_groups = 'system'
        else:
            te = np.array([rid2i[r] for r in sp['test']]); mask = np.ones(len(y), bool); mask[te] = False; tr = np.where(mask)[0]
            Xte, yte, rid_te = X[te], y[te], rec.rid.values[te]; proto_for_groups = sp['protocol']
        seed = (sum(map(ord, sp['label'])) * 7919) % (2 ** 31)
        mu, q = fit_predict(X[tr], y[tr], Xte, seed, quant=True)
        df = pd.DataFrame({'label': sp['label'], 'protocol': sp['protocol'], 'rid': rid_te, 'y': yte, 'yhat': mu,
                           'q05': q[0], 'q16': q[1], 'q84': q[2], 'q95': q[3]})
        variants = [('naive', None)]
        if proto_for_groups != 'random':
            variants.append(('matched', inner_groups(rec.iloc[tr], proto_for_groups)))
        for vn, g in variants:
            iv = cv_plus(X[tr], y[tr], Xte, g, seed)
            for a in (0.1, 0.32):
                tag = int(round((1 - a) * 100)); df[f'cvp_{vn}_lo{tag}'], df[f'cvp_{vn}_hi{tag}'] = iv[a]
        rows.append(df)
        e = df.yhat - df.y
        print(f"{ds} {sp['label']:<22} n={len(df):4d} RMSE={np.sqrt((e**2).mean()):7.1f} t={time.time()-t0:5.1f}s", flush=True)
    out = pd.concat(rows); out.insert(0, 'dataset', ds)
    out.to_csv(f"/kaggle/working/tabpfn_pred_{ds}.csv", index=False)
    return out
T0 = time.time()
res = {ds: run_dataset(ds) for ds in ['HV', 'YS']}
print("TOTAL_MIN", round((time.time() - T0) / 60, 1))
''',
r'''# Compact per-split metrics (printed for transfer) + selected per-record predictions
def cov(d, lo, hi): return float(((d.y >= d[lo]) & (d.y <= d[hi])).mean())
lines = ["M|dataset|label|protocol|n|MAE|RMSE|R2|rho|bias|cov_nat90|w_nat90|cov_nat68|w_nat68|cov_cvn90|w_cvn90|cov_cvn68|w_cvn68|cov_cvm90|w_cvm90|cov_cvm68|w_cvm68"]
for ds, d0 in res.items():
    for lab, d in d0.groupby('label', sort=False):
        e = d.yhat - d.y; ss = ((d.y - d.y.mean()) ** 2).sum()
        v = [len(d), e.abs().mean(), np.sqrt((e ** 2).mean()), 1 - (e ** 2).sum() / ss, spearmanr(d.y, d.yhat).correlation, e.mean(),
             cov(d, 'q05', 'q95'), (d.q95 - d.q05).mean(), cov(d, 'q16', 'q84'), (d.q84 - d.q16).mean(),
             cov(d, 'cvp_naive_lo90', 'cvp_naive_hi90'), (d.cvp_naive_hi90 - d.cvp_naive_lo90).mean(),
             cov(d, 'cvp_naive_lo68', 'cvp_naive_hi68'), (d.cvp_naive_hi68 - d.cvp_naive_lo68).mean()]
        if 'cvp_matched_lo90' in d and d.cvp_matched_lo90.notna().all():
            v += [cov(d, 'cvp_matched_lo90', 'cvp_matched_hi90'), (d.cvp_matched_hi90 - d.cvp_matched_lo90).mean(),
                  cov(d, 'cvp_matched_lo68', 'cvp_matched_hi68'), (d.cvp_matched_hi68 - d.cvp_matched_lo68).mean()]
        else:
            v += [np.nan] * 4
        lines.append("M|" + "|".join([ds, lab, d.protocol.iloc[0]] + [f"{x:.4f}" if isinstance(x, float) else str(x) for x in v]))
print("=====METRICS_BEGIN=====")
print("\n".join(lines))
print("=====METRICS_END=====")
print("=====PRED_BEGIN=====")
print("P|dataset|label|rid|yhat|q05|q95|cvm_lo90|cvm_hi90")
for ds, d0 in res.items():
    for lab in ['temporal-2019', 'external-GE']:
        d = d0[d0.label == lab]
        for r in d.itertuples():
            print(f"P|{ds}|{lab}|{r.rid}|{r.yhat:.1f}|{r.q05:.1f}|{r.q95:.1f}|{r.cvp_matched_lo90:.1f}|{r.cvp_matched_hi90:.1f}")
print("=====PRED_END=====")
''',
]

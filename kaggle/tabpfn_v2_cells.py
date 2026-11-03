"""Kaggle GPU notebook (revision 2): TabPFN per-record predictions for every frozen split (for cluster-bootstrap
inference) and CV+ intervals with deployment-specific calibration folds for P4 (leave-element-out), P5 (forward
chaining) and P6 (leave-element-out primary, system-grouped sensitivity)."""
from tabpfn_cells import CELLS as OLD

CELLS = OLD[:2] + [
r'''# deployment-specific calibration folds (identical to conformal.py in the repository)
def element_folds(sys_keys, k, min_share=0.05, max_share=0.6, min_cal=10):
    els = [set(s.split('-')) for s in sys_keys]; n = len(els)
    cnt = pd.Series([e for s in els for e in s]).value_counts()
    elig = [e for e, c in cnt.items() if min_share * n <= c <= max_share * n]
    assigned = np.zeros(n, bool); out, used = [], []
    for e in elig:
        has = np.array([e in s for s in els]); cal = np.where(has & ~assigned)[0]
        if len(cal) < min_cal: continue
        out.append((np.where(~has)[0], cal)); used.append(e); assigned[cal] = True
        if len(out) == k: break
    return out, used
def forward_chaining_folds(years, k):
    years = np.asarray(years); uy = np.sort(np.unique(years))[::-1][:k]
    return [(np.where(years < v)[0], np.where(years == v)[0]) for v in sorted(uy)], [int(v) for v in sorted(uy)]
def group_folds(groups, k, seed):
    f = balanced_group_folds(np.asarray(groups), k, np.random.default_rng(seed))
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]
def scheme(rtr, protocol, k, seed):
    if protocol in ('element', 'external'):
        f, u = element_folds(rtr.sys_key.values, k); return f, 'LEO:' + ','.join(u)
    if protocol == 'temporal':
        f, u = forward_chaining_folds(rtr.year.values, k); return f, 'FC:' + ','.join(map(str, u))
    if protocol == 'external_system':
        return group_folds(rtr.sys_key.values, k, seed), 'system'
    raise ValueError(protocol)
def cvplus_folds(Xtr, ytr, Xte, folds, seed, alphas=(0.1, 0.32)):
    R, MU = [], []
    for tr, cal in folds:
        pr = fit_predict(Xtr[tr], ytr[tr], np.vstack([Xtr[cal], Xte]), seed)
        R.append(np.abs(pr[:len(cal)] - ytr[cal])); MU.append(np.repeat(pr[len(cal):][None, :], len(cal), 0))
    R = np.concatenate(R); MU = np.vstack(MU); n = len(R); out = {}
    for a in alphas:
        kl = int(np.floor(a * (n + 1))); ku = int(np.ceil((1 - a) * (n + 1)))
        lo = np.sort(MU - R[:, None], 0)[kl - 1] if kl >= 1 else np.full(MU.shape[1], -np.inf)
        hi = np.sort(MU + R[:, None], 0)[ku - 1] if ku <= n else np.full(MU.shape[1], np.inf)
        out[a] = (lo, hi)
    return out, n
print("schemes ok")
''',
r'''from scipy.stats import spearmanr
splits = json.load(open(f"{DATA}/splits.json"))
def run_dataset(ds):
    rec, Xdf = load(ds); cols = feature_columns(Xdf); X = Xdf[cols].values.astype(np.float32); y = rec.target.values.astype(float)
    rid2i = {r: i for i, r in enumerate(rec.rid.values)}
    jobs = list(splits[ds])
    if ds == 'HV':
        jobs.append(dict(label='external-GE', protocol='external', rep=0, fold=0, test=None))
    rows, lines = [], []
    for sp in jobs:
        t0 = time.time()
        if sp['protocol'] == 'external':
            rge, Xge = load('GE'); Xte = Xge[cols].values.astype(np.float32); yte = rge.target.values.astype(float)
            tr = np.arange(len(y)); rid_te = rge.rid.values
        else:
            te = np.array([rid2i[r] for r in sp['test']]); mask = np.ones(len(y), bool); mask[te] = False; tr = np.where(mask)[0]
            Xte, yte, rid_te = X[te], y[te], rec.rid.values[te]
        seed = (sum(map(ord, sp['label'])) * 7919) % (2 ** 31)
        mu = fit_predict(X[tr], y[tr], Xte, seed)
        df = pd.DataFrame({'label': sp['label'], 'protocol': sp['protocol'], 'rid': rid_te, 'y': yte, 'yhat': mu})
        if sp['protocol'] in ('element', 'temporal', 'external'):
            variants = [('m', sp['protocol'])] + ([('s', 'external_system')] if sp['protocol'] == 'external' else [])
            for vn, sch in variants:
                folds, desc = scheme(rec.iloc[tr], sch, 5, seed + 11)
                iv, ncal = cvplus_folds(X[tr], y[tr], Xte, folds, seed)
                for a in (0.1, 0.32):
                    tag = int(round((1 - a) * 100)); lo, hi = iv[a]
                    cov = float(((yte >= lo) & (yte <= hi)).mean()); w = float(np.mean(hi - lo))
                    lines.append(f"C|{ds}|{sp['label']}|{vn}|{tag}|{len(yte)}|{ncal}|{cov:.4f}|{w:.2f}|{desc}")
        rows.append(df)
        e = df.yhat - df.y
        print(f"{ds} {sp['label']:<22} n={len(df):4d} RMSE={np.sqrt((e**2).mean()):8.2f} t={time.time()-t0:5.1f}s", flush=True)
    return pd.concat(rows).assign(dataset=ds), lines
T0 = time.time(); res, CL = {}, []
for ds in ['HV', 'YS']:
    res[ds], l = run_dataset(ds); CL += l
print("TOTAL_MIN", round((time.time() - T0) / 60, 1))
''',
r'''# transfer block: split RMSEs, CV+ coverage lines and gzip+base64 per-record predictions (split order)
import gzip, base64, hashlib, io
allp = pd.concat(res.values()); allp.to_csv("/kaggle/working/tabpfn_v2_pred.csv", index=False)
buf = io.StringIO()
for (ds, lab), d in allp.groupby(['dataset', 'label'], sort=False):
    buf.write(f"{ds}|{lab}|" + ";".join(f"{v:.1f}" for v in d.yhat.values) + "\n")
raw = buf.getvalue().encode(); gz = gzip.compress(raw, 9); b64 = base64.b64encode(gz).decode()
print("=====C_BEGIN====="); print("\n".join(CL)); print("=====C_END=====")
for (ds, lab), d in allp.groupby(['dataset', 'label'], sort=False):
    e = d.yhat - d.y; print(f"R|{ds}|{lab}|{len(d)}|{np.sqrt((e**2).mean()):.3f}")
print("B64_LEN", len(b64), "RAW_MD5", hashlib.md5(raw).hexdigest(), "B64_MD5", hashlib.md5(b64.encode()).hexdigest())
with open("/kaggle/working/b64.txt", "w") as fh: fh.write(b64)
''',
r'''# print base64 in fixed-width lines between markers (read back in chunks)
print("=====B64_BEGIN=====")
for i in range(0, len(b64), 2000): print(b64[i:i + 2000])
print("=====B64_END=====")
''',
]

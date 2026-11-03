"""Shared data loading, feature preprocessing and split generation."""
import numpy as np
import pandas as pd

DATA = "/home/claude/hea/data"
PROC_LEVELS = ['CAST', 'ANNEAL', 'WROUGHT', 'POWDER', 'OTHER', 'UNKNOWN']
DROP_PHYS = ['PHYS:Weight Fraction', 'PHYS:Atomic Fraction', 'PHYS:Total weight']


def load(dataset, data_dir=DATA):
    """Return records dataframe (meta) and full feature frame aligned to records."""
    fname = {'HV': 'hv_records.csv', 'YS': 'ys_records.csv', 'GE': 'ext_ge_records.csv'}[dataset]
    rec = pd.read_csv(f"{data_dir}/{fname}")
    F = pd.read_csv(f"{data_dir}/composition_features.csv")
    F = F.drop(columns=[c for c in DROP_PHYS + ['PHYS_FAIL'] if c in F.columns])
    X = rec[['comp_key']].merge(F, on='comp_key', how='left').drop(columns=['comp_key'])
    assert len(X) == len(rec) and X.notna().all().all(), "feature merge failed"
    for p in PROC_LEVELS:
        X[f"PROC:{p}"] = (rec.proc == p).astype(float).values
    if dataset == 'YS':
        X['TEST:T_C'] = rec.test_T.values
        X['TEST:tension'] = (rec.test == 'T').astype(float).values
    for ph in ['FCC', 'BCC', 'other']:
        X[f"PHASE:{ph}"] = (rec.phase3 == ph).astype(float).values
    X.index = rec.rid.values
    return rec, X


def feature_columns(X, blocks):
    """blocks: iterable of prefixes, e.g. ('FRAC', 'PHYS', 'MAG', 'PROC', 'TEST')."""
    return [c for c in X.columns if c.split(':')[0] in blocks]


DEFAULT_BLOCKS = ('FRAC', 'PHYS', 'MAG', 'PROC', 'TEST')


class Prep:
    """Train-fold-only preprocessing: drop constant columns, drop near-duplicate columns
    (|Pearson r| > 0.98, keep first), standardize. Fitted on training rows only."""

    def __init__(self, corr_thresh=0.98, scale=True, clip=True):
        self.corr_thresh = corr_thresh
        self.scale = scale
        self.clip = clip  # clamp features to the training-fold range (guards linear/NN extrapolation)

    def fit(self, X):
        X = np.asarray(X, float)
        sd = X.std(0)
        keep = np.where(sd > 1e-10)[0]
        Xk = X[:, keep]
        C = np.abs(np.corrcoef(Xk, rowvar=False))
        np.fill_diagonal(C, 0)
        drop = set()
        for i in range(C.shape[0]):
            if i in drop:
                continue
            for j in np.where(C[i, i + 1:] > self.corr_thresh)[0] + i + 1:
                drop.add(j)
        self.keep = keep[[i for i in range(len(keep)) if i not in drop]]
        self.mu = X[:, self.keep].mean(0)
        self.sd = X[:, self.keep].std(0)
        self.lo = X[:, self.keep].min(0)
        self.hi = X[:, self.keep].max(0)
        return self

    def transform(self, X):
        X = np.asarray(X, float)[:, self.keep]
        if self.clip:
            X = np.clip(X, self.lo, self.hi)
        return (X - self.mu) / self.sd if self.scale else X


def balanced_group_folds(groups, n_splits, rng):
    """Random assignment of groups to folds, balancing record counts (largest groups first,
    random tie-breaking). Returns fold id per record."""
    groups = np.asarray(groups)
    ug, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(ug))
    order = order[np.argsort(-counts[order], kind='stable')]
    load_ = np.zeros(n_splits)
    gfold = {}
    for gi in order:
        f = int(np.argmin(load_ + rng.random(n_splits) * 1e-6))
        gfold[ug[gi]] = f
        load_[f] += counts[gi]
    return np.array([gfold[g] for g in groups])


def make_splits(rec, dataset, n_rep=3, n_splits=5, seed=2026, min_elem=40, max_frac=0.6):
    """Outer evaluation protocols. Returns list of dicts:
    {protocol, rep, fold, label, train_idx, test_idx}."""
    n = len(rec)
    out = []
    for r in range(n_rep):
        rng = np.random.default_rng(seed + r)
        perm_fold = np.empty(n, int)
        perm_fold[rng.permutation(n)] = np.arange(n) % n_splits
        for proto, fold_id in [('random', perm_fold),
                               ('composition', balanced_group_folds(rec.comp_key.values, n_splits, rng)),
                               ('system', balanced_group_folds(rec.sys_key.values, n_splits, rng))]:
            for f in range(n_splits):
                te = np.where(fold_id == f)[0]
                tr = np.where(fold_id != f)[0]
                out.append(dict(protocol=proto, rep=r, fold=f, label=f"{proto}-r{r}f{f}", train_idx=tr, test_idx=te))
    # leave-one-element-out
    els = pd.Series([e for s in rec.sys_key for e in s.split('-')]).value_counts()
    for el, cnt in els.items():
        frac = cnt / n
        if cnt >= min_elem and frac <= max_frac:
            has = rec.sys_key.apply(lambda s: el in s.split('-')).values
            out.append(dict(protocol='element', rep=0, fold=0, label=f"element-{el}",
                            train_idx=np.where(~has)[0], test_idx=np.where(has)[0]))
    # temporal: train on literature up to 2018, test on 2019 papers
    yr = rec.year.values
    out.append(dict(protocol='temporal', rep=0, fold=0, label='temporal-2019',
                    train_idx=np.where(yr <= 2018)[0], test_idx=np.where(yr == 2019)[0]))
    return out


def inner_groups(rec_tr, protocol):
    """Grouping used for inner (tuning / calibration) folds; mirrors the deployment scenario."""
    if protocol == 'random':
        return None
    if protocol == 'composition':
        return rec_tr.comp_key.values
    return rec_tr.sys_key.values


def metrics(y, p):
    from scipy.stats import spearmanr
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = p - y
    ss = ((y - y.mean()) ** 2).sum()
    return dict(n=len(y), MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                R2=1 - (e ** 2).sum() / ss if ss > 0 else np.nan,
                rho=spearmanr(y, p).correlation if len(y) > 2 else np.nan, bias=e.mean())

"""Experimental reproducibility ('noise floor') from repeated records of the same nominal alloy.
For groups of records sharing composition and processing route (and, for yield strength, test type and
temperature), the pooled within-group SD estimates the irreducible scatter that no composition-based model can remove.
Also reported separately for repeats coming from different publications (inter-laboratory scatter)."""
import json
import numpy as np
import pandas as pd

D = "/home/claude/hea/data"


def pooled_sd(df, keys):
    g = df.groupby(keys)
    rows = [(len(x), x.target.var(ddof=1)) for _, x in g if len(x) >= 2]
    n = sum(r[0] - 1 for r in rows)
    return (np.sqrt(sum((r[0] - 1) * r[1] for r in rows) / n) if n else np.nan), len(rows), int(sum(r[0] for r in rows))


def main():
    out = {}
    hv = pd.read_csv(f"{D}/hv_records.csv"); ys = pd.read_csv(f"{D}/ys_records.csv")
    for name, df, keys in [('HV', hv, ['comp_key', 'proc']), ('YS', ys, ['comp_key', 'proc', 'test', 'test_T'])]:
        sd, ng, nr = pooled_sd(df, keys)
        # inter-publication: keep only groups whose records come from >= 2 different DOIs
        multi = df.groupby(keys).doi.transform('nunique') >= 2
        sd_x, ng_x, nr_x = pooled_sd(df[multi], keys)
        sd_c, ng_c, nr_c = pooled_sd(df, ['comp_key'])
        out[name] = dict(pooled_sd_same_proc=sd, groups=ng, records=nr, pooled_sd_interlab=sd_x, groups_interlab=ng_x,
                         records_interlab=nr_x, pooled_sd_same_comp_any_proc=sd_c, groups_comp=ng_c, records_comp=nr_c,
                         target_sd=float(df.target.std()))
    print(json.dumps(out, indent=1))
    json.dump(out, open("/home/claude/hea/results/tables/noise_floor.json", 'w'), indent=1)


if __name__ == '__main__':
    main()

"""Freeze outer splits to JSON so every environment (local CPU, Kaggle GPU) uses identical folds.
Training set of every split = complement of its test set within the dataset."""
import json
from common import load, make_splits

out = {}
for ds in ['HV', 'YS']:
    rec, _ = load(ds)
    sp = make_splits(rec, ds)
    out[ds] = [dict(label=s['label'], protocol=s['protocol'], rep=s['rep'], fold=s['fold'],
                    test=rec.rid.values[s['test_idx']].tolist()) for s in sp]
    # sanity: complement property
    for s in sp:
        assert len(set(s['train_idx']) | set(s['test_idx'])) == len(rec) and not set(s['train_idx']) & set(s['test_idx'])
json.dump(out, open('/home/claude/hea/data/splits.json', 'w'))
print({k: len(v) for k, v in out.items()})

import sys
sys.path.insert(0, '/home/claude/hea')
from common import load, feature_columns, DEFAULT_BLOCKS


def nfeat(ds):
    _, X = load(ds)
    return len(feature_columns(X, DEFAULT_BLOCKS))

CELLS = [
r'''import subprocess, sys, importlib, os
def pip(*a):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], capture_output=True, text=True)
    print("PIP", a, "rc", r.returncode, r.stderr[-800:])
pip("tabpfn")
pip("crabnet", "--no-deps")
for m in ["tabpfn", "crabnet", "torch"]:
    try:
        mod = importlib.import_module(m); print("VER", m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print("VER", m, "MISSING", repr(e)[:300])
''',
r'''import numpy as np, time, traceback
X = np.random.RandomState(0).rand(400, 189); y = X[:, :12] @ np.arange(12) + np.random.RandomState(1).randn(400) * 0.1
from tabpfn import TabPFNRegressor
for tag in ["default", "v2"]:
    try:
        if tag == "default":
            m = TabPFNRegressor(device="cuda", random_state=0)
        else:
            from tabpfn.constants import ModelVersion
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", random_state=0)
        t = time.time(); m.fit(X[:300], y[:300]); p = m.predict(X[300:])
        q = m.predict(X[300:], output_type="quantiles", quantiles=[0.05, 0.5, 0.95])
        print("TABPFN", tag, "OK", round(time.time() - t, 2), "rmse", float(np.sqrt(np.mean((p - y[300:]) ** 2))), "qshape", np.shape(q))
        print("TABPFN_INFO", tag, getattr(m, "model_path", None), getattr(m, "model_version", None))
    except Exception as e:
        print("TABPFN", tag, "FAIL", traceback.format_exc()[-1200:])
''',
r'''try:
    import pandas as pd, numpy as np, time
    from crabnet.crabnet_ import CrabNet
    df = pd.DataFrame({"formula": ["Al0.5CoCrFeNi", "CoCrFeNi", "AlCoCrFeNi", "HfNbTaTiZr", "MoNbTaW", "CoCrFeMnNi"] * 30,
                       "target": [300, 150, 500, 390, 520, 140] * 30})
    df["proc"] = np.arange(len(df)) % 2
    t = time.time()
    cb = CrabNet(mat_prop="test", epochs=40, extend_features=["proc"], verbose=False, losscurve=False, learningcurve=False)
    cb.fit(df)
    out = cb.predict(df.head(6), return_uncertainty=True)
    print("CRABNET_OK", round(time.time() - t, 1), [np.round(np.asarray(o), 1).tolist() for o in out])
except Exception as e:
    import traceback; print("CRABNET_FAIL", traceback.format_exc()[-2500:])
''',
]

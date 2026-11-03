CELLS = [
r'''import subprocess, sys, os, platform, importlib
print("PY", platform.python_version(), "CPUS", os.cpu_count())
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], capture_output=True, text=True).stdout)
for m in ["sklearn","xgboost","lightgbm","catboost","torch","shap","scipy","pandas","numpy","pymatgen","matminer","optuna"]:
    try:
        mod = importlib.import_module(m); print("VER", m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print("VER", m, "MISSING", type(e).__name__)
''',
r'''r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn", "crabnet", "matminer"], capture_output=True, text=True)
print("PIPRC", r.returncode); print(r.stdout[-1500:]); print(r.stderr[-2500:])
for m in ["tabpfn","crabnet","matminer","pymatgen","torch","sklearn"]:
    try:
        mod = importlib.import_module(m); print("VER2", m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print("VER2", m, "MISSING", repr(e)[:300])
''',
r'''import json, urllib.request, io, pandas as pd
meta = json.load(urllib.request.urlopen("https://api.figshare.com/v2/articles/17032760"))
print("FIGSHARE", meta.get("title"), meta.get("published_date"), meta.get("license", {}).get("name"))
for f in meta["files"]:
    print("FILE", f["name"], f["size"], f["download_url"], f.get("computed_md5"))
url = meta["files"][0]["download_url"]
raw = urllib.request.urlopen(url).read()
open("/kaggle/working/GE_RefractoryAlloyScreeningDataset_FINAL.csv", "wb").write(raw)
import hashlib; print("MD5", hashlib.md5(raw).hexdigest(), len(raw))
print("=====GE_CSV_BEGIN=====")
print(raw.decode("utf-8", errors="replace"))
print("=====GE_CSV_END=====")
''',
r'''import numpy as np, time, traceback
X = np.random.RandomState(0).rand(300, 12); y = X @ np.arange(12) + np.random.RandomState(1).randn(300) * 0.1
try:
    from tabpfn import TabPFNRegressor
    t = time.time(); m = TabPFNRegressor(device="cuda", random_state=0); m.fit(X[:250], y[:250]); p = m.predict(X[250:])
    print("TABPFN_DEFAULT_OK", round(time.time() - t, 2), float(np.sqrt(np.mean((p - y[250:]) ** 2))))
except Exception as e:
    print("TABPFN_DEFAULT_FAIL", repr(e)[:800])
try:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda")
    m.fit(X[:250], y[:250]); p = m.predict(X[250:])
    print("TABPFN_V2_OK", float(np.sqrt(np.mean((p - y[250:]) ** 2))))
except Exception as e:
    print("TABPFN_V2_FAIL", repr(e)[:800])
''',
r'''try:
    import pandas as pd, numpy as np
    from crabnet.crabnet_ import CrabNet
    df = pd.DataFrame({"formula": ["Al0.5CoCrFeNi", "CoCrFeNi", "AlCoCrFeNi", "HfNbTaTiZr", "MoNbTaW", "CoCrFeMnNi"] * 20,
                       "target": [300, 150, 500, 390, 520, 140] * 20})
    cb = CrabNet(mat_prop="test", epochs=40, verbose=False, losscurve=False, learningcurve=False)
    cb.fit(df)
    print("CRABNET_OK", cb.predict(df.head(6)).round(1).tolist())
except Exception as e:
    import traceback; print("CRABNET_FAIL", traceback.format_exc()[-1500:])
''',
]

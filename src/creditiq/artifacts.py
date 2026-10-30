"""Model bundle, manifest/checksums, export zip and recovery."""
from __future__ import annotations
import hashlib, json, platform, sys, time, zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BUNDLE_NAME = "creditiq_bundle.joblib"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def dependency_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "shap", "duckdb", "fastapi", "streamlit", "joblib"]:
        try: out[m] = __import__(m).__version__
        except Exception: out[m] = "n/a"
    return out


def build_schema(X: pd.DataFrame) -> dict:
    """Input schema from training data: dtype, range (numeric) or allowed categories."""
    sch = {}
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            sch[c] = {"type": "numeric", "min": float(s.min()) if s.notna().any() else None, "max": float(s.max()) if s.notna().any() else None,
                      "missing_frac": float(s.isna().mean()), "median": float(s.median()) if s.notna().any() else None}
        else:
            vc = s.astype(object).dropna().value_counts()
            sch[c] = {"type": "categorical", "categories": [str(x) for x in vc.index[:60]], "missing_frac": float(s.isna().mean())}
    return sch


def save_bundle(paths, bundle: dict) -> Path:
    bundle = {**bundle, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "dependencies": dependency_versions()}
    f = paths.models / BUNDLE_NAME
    joblib.dump(bundle, f, compress=3)
    return f


def load_bundle(path: Path, verify: bool = True) -> dict:
    path = Path(path)
    man = path.parent / "manifest.json"
    if verify and man.exists():
        recorded = json.loads(man.read_text()).get("files", {}).get(path.name)
        if recorded and recorded != sha256(path): raise ValueError("Checksum mismatch: bundle changed since export")
    return joblib.load(path)


def write_manifest(paths, version: str) -> Path:
    files = {}
    for f in [paths.models / BUNDLE_NAME, *sorted(paths.metrics.glob("*.json")), *sorted(paths.reports.glob("*.csv")), *sorted(paths.reports.glob("*.md"))]:
        if f.exists(): files[str(f.relative_to(paths.root))] = sha256(f)
    man = {"project": "CreditIQ", "model_version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "files": files, "dependencies": dependency_versions(),
           "reproduce": ["pip install -r requirements.txt", "python -m creditiq.pipeline --stage all", "pytest -q"]}
    f = paths.models / "manifest.json"
    f.write_text(json.dumps(man, indent=2))
    # keep bundle checksum discoverable by load_bundle(): short name key
    man["files"][BUNDLE_NAME] = files.get(str((paths.models / BUNDLE_NAME).relative_to(paths.root)), "")
    f.write_text(json.dumps(man, indent=2))
    return f


def export_zip(paths, version: str) -> Path:
    """Portable bundle: model + metrics + reports + docs + configs + manifest (no raw data)."""
    man = write_manifest(paths, version)
    dst = paths.exports / f"creditiq_export_{version}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(man, "manifest.json")
        for sub in [paths.models, paths.metrics, paths.reports, paths.docs, paths.configs]:
            for f in sub.rglob("*"):
                if f.is_file() and f.name != ".gitkeep": z.write(f, str(f.relative_to(paths.root)))
        z.writestr("REPRODUCE.txt", "Unzip into MyDrive/CreditIQ (or any folder), set CREDITIQ_ROOT, then:\n"
                   "pip install -r requirements.txt\nstreamlit run dashboard/app.py\nuvicorn api.main:app\n")
    return dst

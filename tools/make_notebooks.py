"""Generates the Colab notebooks (thin: all logic lives in src/creditiq). Run: python tools/make_notebooks.py"""
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"; OUT.mkdir(exist_ok=True)

SETUP = '''# --- Colab setup (safe to re-run) ---
import os, sys, subprocess
from pathlib import Path
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass
ROOT = Path("/content/drive/MyDrive/CreditIQ") if Path("/content/drive/MyDrive").exists() else Path.cwd().parent
os.environ["CREDITIQ_ROOT"] = str(ROOT); sys.path.insert(0, str(ROOT / "src"))
if not (ROOT / "src").exists():
    raise SystemExit(f"Project not found at {ROOT}. Unzip CreditIQ into MyDrive/CreditIQ first.")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")], check=False)
from creditiq.config import get_paths, load_config, read_state
paths = get_paths(); cfg = load_config(paths)
print("Project root:", ROOT); print("Last completed stage:", read_state(paths).get("last_completed"))'''


def nb(name, title, purpose, inputs, outputs, deps, order, checks, cells):
    n = nbf.v4.new_notebook()
    head = (f"# {title}\n\n**Purpose:** {purpose}\n\n**Inputs:** {inputs}\n\n**Outputs / persisted artifacts:** {outputs}\n\n"
            f"**Dependencies:** {deps}\n\n**Execution order:** {order}\n\n**Validation checks:** {checks}\n\n"
            "_All logic lives in `src/creditiq`; cells are safe to re-run (completed stages are skipped, artifacts live on Google Drive)._")
    n.cells = [nbf.v4.new_markdown_cell(head), nbf.v4.new_code_cell(SETUP)] + [nbf.v4.new_code_cell(c) if not c.startswith("#md ") else nbf.v4.new_markdown_cell(c[4:]) for c in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(n, OUT / name)


nb("01_setup_and_ingestion.ipynb", "01 - Setup, Dataset Acquisition, Audit", "Inspect Colab resources, acquire Home Credit data (Kaggle API or manual upload), validate, convert to Parquet, audit, data dictionary, SQL analysis.",
   "Kaggle secrets (KAGGLE_USERNAME, KAGGLE_KEY) OR CSVs uploaded to data/raw/", "data/interim/*.parquet, artifacts/reports/{dataset_audit,relationship_checks,data_dictionary,leakage_audit}.csv, sql_*.csv",
   "requirements.txt", "1 of 6", "File validation passes; 0 relationship violations; leakage audit lists every DAYS_* column",
   ['''import psutil, shutil, subprocess
print(f"RAM available: {psutil.virtual_memory().available/1e9:.1f} GB | Disk free: {shutil.disk_usage('/content').free/1e9:.0f} GB | CPUs: {os.cpu_count()}")
print(subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout or "No GPU (not required: LightGBM/XGBoost run on CPU)")''',
    '''#md **Dataset access.** Option A: add Colab Secrets `KAGGLE_USERNAME` / `KAGGLE_KEY` and accept the competition rules on Kaggle. Option B: upload the 8 CSVs to `MyDrive/CreditIQ/data/raw/`. Never paste tokens into cells.''',
    '''from creditiq import pipeline
pipeline.run("ingest")      # download (if needed) -> validate -> CSV to Parquet (resumable)
pipeline.run("audit")      # audit, relationships, data dictionary, leakage audit, SQL analysis queries''',
    '''import pandas as pd
display(pd.read_csv(paths.reports / "dataset_audit.csv")); display(pd.read_csv(paths.reports / "relationship_checks.csv"))
display(pd.read_csv(paths.reports / "leakage_audit.csv"))
for f in sorted(paths.reports.glob("sql_*.csv")): print(f.name); display(pd.read_csv(f))'''])

nb("02_features_and_leakage.ipynb", "02 - Relational Aggregation, Feature Engineering, Leakage Audit", "Aggregate history tables with DuckDB SQL into one row per SK_ID_CURR, engineer ratios, document features and availability.",
   "data/interim/*.parquet, sql/*.sql", "data/processed/features_{train,test}.parquet, feature_documentation.csv, feature_availability.csv, train_test_consistency.json",
   "Notebook 01", "2 of 6", "Unique SK_ID_CURR, no row multiplication, train/test column consistency",
   ['''from creditiq import pipeline
pipeline.run("aggregate"); pipeline.run("features")''',
    '''import pandas as pd
tr = pd.read_parquet(paths.processed / "features_train.parquet"); print(tr.shape, "unique ids:", tr.SK_ID_CURR.is_unique)
display(pd.read_csv(paths.reports / "feature_documentation.csv")); display(pd.read_csv(paths.reports / "feature_availability.csv").head(30))''',
    '''#md **Leakage note.** The dataset provides relative day offsets only. The audit checks consistency with the documented convention (<= 0 = before application) and neutralises future-dated values; it cannot prove temporal validity independently.'''])

nb("03_eda_and_statistics.ipynb", "03 - EDA and Statistical Analysis", "Professional EDA (with what/why/observation/interpretation/limitations) and hypothesis tests with effect sizes and multiple-testing correction.",
   "data/processed/features_train.parquet", "artifacts/reports/eda_report.md, figures/*.png, statistical_tests.csv, core_questions.json",
   "Notebook 02", "3 of 6", "Report observations are computed from data; Holm/BH corrections present",
   ['''from creditiq import pipeline
pipeline.run("eda"); pipeline.run("stats")''',
    '''from IPython.display import Markdown, display
display(Markdown((paths.reports / "eda_report.md").read_text().replace("](figures/", f"]({paths.figures}/")))''',
    '''import pandas as pd, json
t = pd.read_csv(paths.reports / "statistical_tests.csv"); display(t.head(25)); print(json.dumps(json.load(open(paths.reports / "core_questions.json")), indent=2))'''])

nb("04_modeling_evaluation_calibration.ipynb", "04 - Modelling, Evaluation, Calibration", "Baseline-first model comparison (Dummy, LogReg, RF, XGBoost, LightGBM), stratified CV, untouched holdout, calibration, thresholds, error and decile analysis.",
   "data/processed/features_train.parquet, configs/config.yaml", "artifacts/models/creditiq_bundle.joblib, artifacts/metrics/*.csv|json, figures/{roc,pr,calibration,confusion}.png",
   "Notebook 02", "4 of 6", "Holdout never used for fitting/calibration/threshold; metrics.json holds only executed values",
   ['''# Edit configs/config.yaml first if needed (cv.sample_n bounds CV runtime; tuning.optuna_trials=0 means no tuning).
from creditiq import pipeline
pipeline.run("train")      # skip-able: set force=True to retrain, CV results are cached''',
    '''import pandas as pd, json
display(pd.read_csv(paths.metrics / "cv_comparison.csv")); display(pd.read_csv(paths.metrics / "holdout_comparison.csv"))
m = json.load(open(paths.metrics / "metrics.json")); print(json.dumps(m["test_calibrated"], indent=2)); print(m["calibration_comparison"])
imb = paths.metrics / "imbalance_experiment.csv"
if imb.exists(): display(pd.read_csv(imb))''',
    '''from IPython.display import Image, display
for f in ["roc", "pr", "calibration", "confusion"]: display(Image(str(paths.figures / f"{f}.png")))
display(pd.read_csv(paths.metrics / "threshold_analysis.csv")); display(pd.read_csv(paths.metrics / "risk_deciles_test.csv")); display(pd.read_csv(paths.metrics / "error_analysis.csv"))'''])

nb("05_explain_risk_ecl_portfolio_fairness_monitoring.ipynb", "05 - Explainability, Risk Engine, ECL, Portfolio, Fairness, Monitoring", "SHAP, PD risk bands, illustrative ECL, portfolio analytics, subgroup fairness audit, drift monitoring.",
   "creditiq_bundle.joblib, holdout_features.parquet, holdout_scored.parquet", "shap_*.csv/png, portfolio_summary.json, ecl_summary.json, portfolio_*.csv, fairness_*.csv, drift_report.csv",
   "Notebook 04", "5 of 6", "Bands cover [0,1]; ECL = PD x LGD x EAD; observed rate and predicted PD reported separately",
   ['''from creditiq import pipeline
pipeline.run("analytics")''',
    '''from creditiq.risk_engine import RiskEngine
import pandas as pd
eng = RiskEngine(paths=paths); hf = pd.read_parquet(paths.processed / "holdout_features.parquet")
res = eng.assess(hf.head(3))
for r in res: print({k: r[k] for k in ("applicant_id", "pd", "risk_band", "recommendation")}, "|", r["explanation"])''',
    '''import json
display(pd.read_csv(paths.metrics / "shap_global_importance.csv").head(20))
from IPython.display import Image; display(Image(str(paths.figures / "shap_summary.png")))
print(json.dumps(json.load(open(paths.metrics / "portfolio_summary.json")), indent=2)); print(json.dumps(json.load(open(paths.metrics / "ecl_summary.json")), indent=2))
display(pd.read_csv(paths.reports / "fairness_disparity.csv")); display(pd.read_csv(paths.metrics / "drift_report.csv").head(20))'''])

nb("06_application_testing_export.ipynb", "06 - Application, Testing, Reports, Export", "Run pytest, generate docs from executed artifacts, launch API + dashboard inside Colab, export the portable bundle, optional GitHub push.",
   "All previous artifacts", "docs/*.md (auto-generated), exports/creditiq_export_*.zip, manifest.json with SHA-256 checksums",
   "Notebooks 01-05", "6 of 6", "pytest passes (executed), API /health ok, dashboard loads, export manifest verifies",
   ['''!cd "$CREDITIQ_ROOT" && python -m pytest -q 2>&1 | tail -15''',
    '''from creditiq import pipeline
pipeline.run("report"); pipeline.run("export")''',
    '''#md **Launch API (port 8000) and dashboard (port 8501).** Colab proxies the ports privately to your session - no public tunnel, no secrets exposed. Colab is not permanent hosting.''',
    '''import subprocess, time
api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"], cwd=ROOT, env={**os.environ})
dash = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.headless", "true",
                         "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"], cwd=ROOT, env={**os.environ})
time.sleep(15)
import requests; print(requests.get("http://localhost:8000/health").json())''',
    '''from google.colab import output
output.serve_kernel_port_as_window(8501)      # dashboard (opens a Colab-proxied link)
# output.serve_kernel_port_as_iframe(8501, height=800)   # or inline''',
    '''# Optional: push SOURCE to GitHub (token from Colab Secrets, never hardcoded)
# from google.colab import userdata
# token = userdata.get("GITHUB_TOKEN"); repo = "https://{}@github.com/<user>/CreditIQ.git".format(token)
# !cd "$CREDITIQ_ROOT" && git init -q && git add -A && git commit -qm "CreditIQ" && git branch -M main && git remote add origin "{repo}" && git push -u origin main''',
    '''api.terminate(); dash.terminate()   # stop servers when done'''])
print("notebooks written:", sorted(p.name for p in OUT.glob("*.ipynb")))

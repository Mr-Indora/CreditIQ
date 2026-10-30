"""CreditIQ prediction API (FastAPI). Run: uvicorn api.main:app --port 8000
Decision-support only. Does not log or echo raw personal data beyond what the caller supplied."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from creditiq.config import get_paths, read_json
from creditiq.risk_engine import RiskEngine

app = FastAPI(title="CreditIQ API", version="1.0.0", description="Probability-of-default decision support. Not an automated lending decision.")
paths = get_paths()
_engine: RiskEngine | None = None


def engine() -> RiskEngine:
    global _engine
    if _engine is None:
        try: _engine = RiskEngine(paths=paths)
        except FileNotFoundError: raise HTTPException(503, "Model bundle not found - run the training pipeline first")
    return _engine


class Applicant(BaseModel):
    applicant_id: int | None = None
    features: dict[str, float | int | str | None] = Field(..., description="Raw application/history features by column name; missing features are imputed")
    ead: float | None = Field(None, gt=0, description="Proposed exposure for illustrative ECL (defaults to AMT_CREDIT)")
    explain: bool = True


class BatchRequest(BaseModel):
    applicants: list[Applicant] = Field(..., max_length=1000)
    explain: bool = False


def _frame(items: list[Applicant]) -> pd.DataFrame:
    rows = []
    for a in items:
        r = dict(a.features)
        if a.applicant_id is not None: r["SK_ID_CURR"] = a.applicant_id
        rows.append(r)
    return pd.DataFrame(rows)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": (paths.models / "creditiq_bundle.joblib").exists()}


@app.get("/model/info")
def model_info():
    e = engine(); b = e.bundle
    return {"model_version": e.version, "model_name": b["model_name"], "calibration": b["calibrator"].method, "n_input_features": len(e.cols),
            "risk_bands": e.bands, "decision_threshold": b["decision_threshold"], "lgd_scenarios": e.lgd, "created": b["created"], "dependencies": b["dependencies"]}


@app.post("/predict")
def predict(a: Applicant):
    e = engine()
    try: return e.assess(_frame([a]), ead=[a.ead], explain=a.explain)[0]
    except ValueError as ex: raise HTTPException(422, str(ex))


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    e = engine()
    try: return {"count": len(req.applicants), "results": e.assess(_frame(req.applicants), ead=[a.ead for a in req.applicants], explain=req.explain)}
    except ValueError as ex: raise HTTPException(422, str(ex))


def _json_or_404(name):
    f = paths.metrics / name
    if not f.exists(): raise HTTPException(404, f"{name} not available - run the pipeline analytics stage")
    return read_json(f)


@app.get("/risk/portfolio")
def risk_portfolio(): return {"portfolio": _json_or_404("portfolio_summary.json"), "ecl": _json_or_404("ecl_summary.json")}


@app.get("/model/metrics")
def model_metrics(): return _json_or_404("metrics.json")


@app.get("/monitoring/drift")
def monitoring_drift():
    f = paths.metrics / "drift_report.csv"
    if not f.exists(): raise HTTPException(404, "drift_report.csv not available")
    d = pd.read_csv(f).fillna("")
    return {"alerts": int((d.alert == "ALERT").sum()), "warnings": int((d.alert == "WARN").sum()), "features": d.head(100).to_dict("records"),
            "prediction_monitoring": _json_or_404("prediction_monitoring.json")}

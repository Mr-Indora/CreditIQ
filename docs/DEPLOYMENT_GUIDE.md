# Deployment Guide
**Colab (demo):** notebook 06 starts uvicorn (8000) and Streamlit (8501) and exposes the dashboard through Colab's private port proxy (no public tunnel). Sessions are temporary.

**Portable hosting:** unzip the export bundle (`creditiq_export_*.zip`), `pip install -r requirements.txt`, set `CREDITIQ_ROOT`, run `uvicorn api.main:app` and `streamlit run dashboard/app.py`. The bundle loads without retraining; `artifacts/models/manifest.json` carries SHA-256 checksums verified at load.

Hardening before any real use: authentication, TLS, request logging without personal data, model-risk governance and legal review of attribute use.

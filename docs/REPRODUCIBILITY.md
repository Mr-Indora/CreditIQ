# Reproducibility
Fixed seed (42) for splits, models and CV; versions recorded in the bundle and manifest; every stage resumable. `pytest -q` includes a retrain-and-compare reproducibility test. Reproduce: `pip install -r requirements.txt && python -m creditiq.pipeline --stage all`.
Results can differ slightly across library versions and CPU thread counts (boosting/RF).

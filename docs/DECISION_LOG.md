# Decision Log
| Decision | Reason |
|---|---|
| DuckDB SQL for aggregation | Streams 13M-row tables with low RAM; SQL is auditable; satisfies SQL deliverable |
| Parquet checkpoints | Resumable after Colab resets |
| Exclude CODE_GENDER from model | Keep sensitive attribute for audit only; age retained but configurable |
| Holdout + calibration split | Calibration/threshold never touch holdout labels |
| SMOTE off by default | Class weighting/calibration preferred; SMOTE only inside training pipeline if enabled |
| No GPU | Tabular boosting on CPU is stable on Colab |
| Colab private port proxy | Avoids public tunnels/secret exposure |
| 6 consolidated notebooks | Avoid fragmentation; logic in modules |

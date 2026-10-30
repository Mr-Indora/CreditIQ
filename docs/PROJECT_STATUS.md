# Project Status (from pipeline_state.json)

| stage | at | seconds | raw_dir | synthetic | tables | rows | features | tests | model | test_auc | drift_alerts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ingest | 2026-10-30T07:43:06 | 209.5 | /content/drive/MyDrive/CreditIQ/data/raw | False | nan | nan | nan | nan | nan | nan | nan |
| audit | 2026-10-30T07:43:55 | 48.8 | nan | nan | 8.0 | nan | nan | nan | nan | nan | nan |
| aggregate | 2026-10-30T07:44:12 | 17.1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| features | 2026-10-30T07:44:28 | 15.8 | nan | nan | nan | 307511.0 | 179.0 | nan | nan | nan | nan |
| eda | 2026-10-30T07:45:14 | 46.5 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| stats | 2026-10-30T07:57:37 | 79.8 | nan | nan | nan | nan | nan | 128.0 | nan | nan | nan |
| train | 2026-10-30T08:35:09 | 1852.7 | nan | nan | nan | nan | nan | nan | lightgbm | 0.7823 | nan |
| analytics | 2026-10-30T08:35:39 | 30.4 | nan | nan | nan | nan | nan | nan | nan | nan | 0.0 |

Last completed: analytics

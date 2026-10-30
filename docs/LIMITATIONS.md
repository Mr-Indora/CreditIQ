# Limitations
- Label is a historical competition definition, not a verified default event; PD is a model estimate.
- Temporal validity relies on the dataset's relative day-offset convention and cannot be independently verified; no out-of-time validation is possible.
- Single static snapshot: no live monitoring; the drift module compares a reference to a supplied current dataset. Calibration/performance monitoring needs realised outcomes.
- Risk bands and LGD scenarios are illustrative assumptions; ECL is not IFRS 9 and not realised loss.
- Age is a model input (configurable); CODE_GENDER is excluded from the model. Fairness metrics are descriptive; they do not prove or exclude discrimination and lawful-use rules vary by jurisdiction.
- Manual single-applicant entry in the dashboard supplies only a feature subset; unspecified history features are imputed, so those scores are less reliable.
- The final model is fitted on the fit split (calibration data held back); it is not refit on all development data.
- Colab sessions are temporary and are not production hosting.

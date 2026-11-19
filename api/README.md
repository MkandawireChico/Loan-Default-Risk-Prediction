# API service

The FastAPI service provides synthetic portfolio analytics, calibrated model inference, batch scoring and model contribution explanations.

## Run locally

```bash
pip install -r requirements-api.txt
python scripts/train_model_v2.py --output-dir models/v2
python scripts/run_temporal_validation.py --output-dir models/temporal
uvicorn api.main:app --reload
```

The service expects the generated v2 model artifact, model metadata, synthetic raw portfolio CSV and cleaned synthetic sample CSV.

## Endpoints

- `GET /health`: API, model, portfolio, database and model-registry readiness.
- `GET /api/v1/model`: model version, threshold and feature governance.
- `GET /api/v1/model/evaluation`: held-out metrics, bootstrap intervals, calibration bins and subgroup diagnostics.
- `GET /api/v1/model/temporal-evaluation`: synthetic chronological comparison of a prior baseline, calibrated Logistic Regression and calibrated Random Forest.
- `GET /api/v1/model/registry`: registered model versions, active/archived status, artifact and dataset fingerprints, training metadata and final-test summary.
- `GET /api/v1/portfolio/summary`: synthetic portfolio summary.
- `GET /api/v1/portfolio/breakdowns`: branch, segment and purpose summaries.
- `GET /api/v1/assessments/sample`: synthetic example application.
- `POST /api/v1/assessments`: individual inference.
- `POST /api/v1/assessments/explain`: inference with model contribution explanation.
- `POST /api/v1/assessments/batch`: batch inference.
- `GET /api/v1/audit/summary`: aggregate operational assessment counts.

Interactive API documentation is available at `/docs`.

## Database and audit boundary

SQLAlchemy stores model-version metadata and assessment metadata in PostgreSQL in Docker Compose or SQLite during local standalone development. Every training run creates a unique version using its UTC build time and training dataset hash prefix. Model artifacts remain in the application filesystem; the registry stores SHA-256 fingerprints, paths, training metadata and evaluation summaries. A new version becomes active and earlier versions are archived. This is a metadata registry, not an artifact store: old model files are not preserved and rollback is not implemented. Audit records contain the model version, probability, risk band, review flag, request source and timestamp. Raw features, names, identifiers and contact information are not persisted.

Set `DATABASE_URL` to configure the database. Do not use real customer data in this demonstration.

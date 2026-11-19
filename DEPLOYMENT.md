# Deployment guide

## Local full-stack deployment

Requirements: Docker Engine and Docker Compose.

From the repository root:

```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose ps
```

Open the application at http://localhost:8080 and API documentation at http://localhost:8000/docs.

The stack contains PostgreSQL, the FastAPI service and the React frontend. The API image trains the v2 calibrated demonstration model from the committed synthetic training CSV during image construction. Each training run generates a version identifier containing the UTC build time and training dataset hash prefix. The model artifact and metadata are generated inside the image. At container startup, Alembic applies pending database migrations before Uvicorn starts. PostgreSQL stores model version records, artifact and dataset fingerprints, training provenance, evaluation summaries, and metadata-only assessment audit events. Raw application features are not stored. The registry does not archive historical model files or provide rollback; those require a separate artifact-storage phase. The frontend is compiled as a static production build and served by Nginx.

To stop the stack:

```bash
docker compose down
```

## Database migrations

The schema is versioned with Alembic. For a local API process, apply migrations before startup:

```bash
alembic upgrade head
```

Inspect the current revision:

```bash
alembic current
```

Create a future migration after changing SQLAlchemy models:

```bash
alembic revision --autogenerate -m "describe schema change"
```

Review every generated migration before applying it. Do not use `Base.metadata.create_all()` as a production schema-management strategy.

## Environment configuration

- `CREDIT_RISK_THRESHOLD`: optional override for the validation-selected model review threshold. Leave unset to use model metadata.
- `VITE_API_URL`: API origin embedded into the frontend at build time.
- `CORS_ORIGINS`: comma-separated frontend origins allowed by the API.
- `DATABASE_URL`: SQLAlchemy connection string. Docker Compose sets the PostgreSQL URL.
- `POSTGRES_PASSWORD`: local PostgreSQL password. Change before any shared deployment.
- `CREDIT_RISK_MODEL_PATH`: optional model artifact path.
- `CREDIT_RISK_METADATA_PATH`: optional model metadata path.
- `CREDIT_RISK_EVALUATION_PATH`: optional evaluation report path.
- `CREDIT_RISK_DATA_PATH`: optional synthetic portfolio data path.
- `CREDIT_RISK_SAMPLE_PATH`: optional cleaned synthetic sample data path.

Never commit secrets, personal information or real financial records.

## Public deployment

A Render Blueprint is provided in `render.yaml`. It defines a Docker-based FastAPI service, a static frontend, and managed PostgreSQL in Frankfurt. The API accepts only the frontend's configured origin. The database connection URL is normalised for the installed Psycopg 3 driver.

Review the selected plans in the Render dashboard before provisioning. The API uses the `1c-2g` service plan and PostgreSQL uses `basic-256mb`. These resources may incur charges. Do not provision them until you have reviewed and approved the current prices.

To deploy:
1. Push the deployment configuration to the GitHub branch you intend to deploy.
2. In Render, create a new Blueprint and connect this repository.
3. Select `render.yaml` and inspect every resource and estimated charge.
4. Approve provisioning only after reviewing the costs.
5. Wait for the database, API and frontend services to finish deploying.
6. Verify the API `/health`, API documentation and frontend URL.
7. Run the manual end-to-end checks below.

Render handles TLS termination and static hosting. Never commit a populated `.env` file.

Before publishing:
1. Build and run the complete Docker Compose stack.
2. Verify API health, portfolio metrics, sample assessment and batch scoring.
3. Verify model report and threshold metadata.
4. Run automated tests and inspect container logs.
5. Confirm the synthetic-data notice is visible in the application.
6. Confirm no secrets or real customer data exist in the image or repository.

## Scope and safety

This project uses entirely synthetic data. A successful deployment demonstrates engineering and reproducibility only. It does not establish model suitability for real credit decisions.

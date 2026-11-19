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

The stack contains PostgreSQL, the FastAPI service and the React frontend. The API image trains the v2 calibrated demonstration model from the committed synthetic training CSV during image construction. The model artifact and metadata are generated inside the image. PostgreSQL stores metadata-only assessment audit events. Raw application features are not stored. The frontend is compiled as a static production build and served by Nginx.

To stop the stack:

```bash
docker compose down
```

## Environment configuration

- `CREDIT_RISK_THRESHOLD`: optional override for the validation-selected model review threshold. Leave unset to use model metadata.
- `VITE_API_URL`: API origin embedded into the frontend at build time.
- `CORS_ORIGINS`: comma-separated frontend origins allowed by the API.
- `DATABASE_URL`: SQLAlchemy connection string. Docker Compose sets the PostgreSQL URL.
- `POSTGRES_PASSWORD`: local PostgreSQL password. Change before any shared deployment.
- `CREDIT_RISK_MODEL_PATH`: optional model artifact path.
- `CREDIT_RISK_DATA_PATH`: optional synthetic portfolio data path.
- `CREDIT_RISK_SAMPLE_PATH`: optional cleaned synthetic sample data path.

Never commit secrets, personal information or real financial records.

## Public deployment

The first deployment target should run the API and frontend as separate services. Configure the frontend build with the public API URL and configure API CORS to allow only the public frontend origin. Terminate TLS at the hosting provider or reverse proxy. Use managed PostgreSQL with encrypted connections, automated backups, restricted network access and rotated credentials. Set the API database URL and frontend CORS origin through the hosting platform's secret/configuration manager. Never commit a populated `.env` file.

Before publishing:
1. Build and run the complete Docker Compose stack.
2. Verify API health, portfolio metrics, sample assessment and batch scoring.
3. Verify model report and threshold metadata.
4. Run automated tests and inspect container logs.
5. Confirm the synthetic-data notice is visible in the application.
6. Confirm no secrets or real customer data exist in the image or repository.

## Scope and safety

This project uses entirely synthetic data. A successful deployment demonstrates engineering and reproducibility only. It does not establish model suitability for real credit decisions.

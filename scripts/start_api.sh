#!/bin/sh
set -eu

echo "Applying database migrations..."
alembic upgrade head

echo "Starting Nyanja Credit Risk Intelligence API..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000

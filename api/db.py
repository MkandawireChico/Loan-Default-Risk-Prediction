"""Database setup, model registry and privacy-preserving assessment audit operations."""

from __future__ import annotations

import json
import os
from typing import Any

from sqlalchemy import Integer, create_engine, func, inspect, select, update
from sqlalchemy.orm import Session, sessionmaker

from api.models import AssessmentAudit, ModelRegistry


def normalize_database_url(url: str) -> str:
    """Use the installed Psycopg 3 driver for PostgreSQL connection URLs."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


DATABASE_URL = normalize_database_url(
    os.getenv("DATABASE_URL", "sqlite:///./nyanja_credit_risk.db")
)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Verify that all versioned database tables exist after Alembic migrations."""
    with engine.connect() as connection:
        inspector = inspect(connection)
        if not inspector.has_table("alembic_version"):
            raise RuntimeError("Database migrations have not been applied. Run 'alembic upgrade head'.")
        for table in (AssessmentAudit.__tablename__, ModelRegistry.__tablename__):
            if not inspector.has_table(table):
                raise RuntimeError(
                    f"The {table} table is missing. Check the Alembic migration state."
                )


def register_model_version(
    session: Session,
    *,
    model_version: str,
    model_type: str,
    artifact_path: str,
    artifact_sha256: str,
    dataset_sha256: str,
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
) -> ModelRegistry:
    """Register the loaded artifact and make its version the sole active version."""
    session.execute(update(ModelRegistry).values(status="archived"))
    record = session.get(ModelRegistry, model_version)
    if record is None:
        record = ModelRegistry(model_version=model_version)
        session.add(record)

    record.model_type = model_type
    record.status = "active"
    record.artifact_path = artifact_path
    record.artifact_sha256 = artifact_sha256
    record.dataset_sha256 = dataset_sha256
    record.training_rows = metadata.get(
        "training_rows", metadata.get("dataset_rows_after_exact_deduplication")
    )
    record.training_seed = metadata.get("random_seed")
    record.decision_threshold = metadata.get("decision_threshold")
    record.metadata_json = json.dumps(metadata, sort_keys=True)
    record.evaluation_json = json.dumps(evaluation, sort_keys=True)
    session.commit()
    session.refresh(record)
    return record


def get_model_registry(session: Session) -> list[dict[str, Any]]:
    """Return version metadata and compact held-out metrics, newest registration first."""
    records = session.scalars(
        select(ModelRegistry).order_by(ModelRegistry.registered_at.desc())
    ).all()
    result = []
    for record in records:
        metadata = json.loads(record.metadata_json)
        evaluation = json.loads(record.evaluation_json)
        result.append(
            {
                "model_version": record.model_version,
                "model_type": record.model_type,
                "status": record.status,
                "artifact_path": record.artifact_path,
                "artifact_sha256": record.artifact_sha256,
                "dataset_sha256": record.dataset_sha256,
                "training_rows": record.training_rows,
                "training_seed": record.training_seed,
                "decision_threshold": record.decision_threshold,
                "registered_at": record.registered_at.isoformat(),
                "metadata": metadata,
                "final_test": evaluation.get("final_test", {}),
                "data_notice": "Model and evaluation data are synthetic demonstrations only.",
            }
        )
    return result


def record_assessment(
    session: Session,
    *,
    model_version: str,
    probability_of_default: float,
    risk_band: str,
    review_flag: bool,
    source: str = "single",
) -> AssessmentAudit:
    """Persist model output metadata only. Never persist applicant feature values."""
    if source not in {"single", "batch"}:
        raise ValueError("source must be 'single' or 'batch'")
    record = AssessmentAudit(
        model_version=model_version,
        probability_of_default=probability_of_default,
        risk_band=risk_band,
        review_flag=review_flag,
        source=source,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def get_audit_summary(session: Session) -> dict[str, Any]:
    total, mean_probability, flagged = session.execute(
        select(
            func.count(AssessmentAudit.id),
            func.avg(AssessmentAudit.probability_of_default),
            func.sum(AssessmentAudit.review_flag.cast(Integer)),
        )
    ).one()
    groups = session.execute(
        select(AssessmentAudit.risk_band, func.count(AssessmentAudit.id))
        .group_by(AssessmentAudit.risk_band)
    ).all()
    return {
        "assessments_recorded": int(total or 0),
        "average_probability_of_default": round(float(mean_probability or 0), 5),
        "review_flags": int(flagged or 0),
        "risk_band_counts": {str(name): int(count) for name, count in groups},
        "stored_fields": [
            "model_version",
            "probability_of_default",
            "risk_band",
            "review_flag",
            "source",
            "created_at",
        ],
        "privacy_notice": "Applicant features, names, contact details and identifiers are not stored.",
    }

"""Database setup and privacy-preserving assessment audit operations."""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import Integer, create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from api.models import AssessmentAudit, Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nyanja_credit_risk.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


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

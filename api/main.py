"""FastAPI inference and portfolio analytics service for synthetic credit-risk research."""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from api.db import (
    SessionLocal,
    get_audit_summary,
    get_model_registry,
    init_db,
    record_assessment,
    register_model_version,
)
from src.models.schema import get_inference_feature_columns
from src.monitoring.drift import compare_distributions

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("CREDIT_RISK_MODEL_PATH", ROOT / "models/v2/credit_risk_calibrated_v2.joblib"))
METADATA_PATH = Path(os.getenv("CREDIT_RISK_METADATA_PATH", ROOT / "models/v2/metadata.json"))
PORTFOLIO_PATH = Path(os.getenv("CREDIT_RISK_DATA_PATH", ROOT / "data/raw/loan_applications_synthetic.csv"))
MODEL_VERSION = os.getenv("CREDIT_RISK_MODEL_VERSION", "credit-risk-calibrated-v2")
EVALUATION_PATH = Path(os.getenv("CREDIT_RISK_EVALUATION_PATH", ROOT / "models/v2/evaluation_v2.json"))
TEMPORAL_EVALUATION_PATH = Path(
    os.getenv("CREDIT_RISK_TEMPORAL_EVALUATION_PATH", ROOT / "models/temporal/temporal_validation_report.json")
)
DEFAULT_THRESHOLD = float(os.getenv("CREDIT_RISK_THRESHOLD", "0.50"))

FEATURES = get_inference_feature_columns()
NUMERIC_FIELDS = set(FEATURES["numeric_cols"])
CATEGORICAL_FIELDS = set(FEATURES["categorical_cols"])
BINARY_FIELDS = {"Has_Credit_Card", "Has_Mobile_Banking"}
REQUIRED_FIELDS = NUMERIC_FIELDS | CATEGORICAL_FIELDS
DISCLAIMER = (
    "Synthetic educational demonstration only. This estimate is not a credit decision, "
    "does not establish creditworthiness, and must never be used for real lending."
)


class AssessmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any] = Field(..., description="Exact governed model feature schema.")


class BatchAssessmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applications: list[dict[str, Any]] = Field(..., min_length=1, max_length=500)


class AssessmentOutput(BaseModel):
    model_version: str
    probability_of_default: float
    risk_band: str
    review_flag: bool
    interpretation: str
    disclaimer: str


def validate_features(values: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_FIELDS - values.keys())
    unexpected = sorted(values.keys() - REQUIRED_FIELDS)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"Missing fields: {', '.join(missing)}")
        if unexpected:
            details.append(f"Unexpected fields: {', '.join(unexpected)}")
        raise HTTPException(status_code=422, detail="; ".join(details))

    for name, value in values.items():
        if value is None:
            continue
        if name in NUMERIC_FIELDS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(status_code=422, detail=f"{name} must be a number or null")
            if not math.isfinite(float(value)):
                raise HTTPException(status_code=422, detail=f"{name} must be finite")
        elif name in BINARY_FIELDS:
            if isinstance(value, bool) or value not in (0, 1, "0", "1"):
                raise HTTPException(status_code=422, detail=f"{name} must be 0, 1 or null")
        elif not isinstance(value, str):
            raise HTTPException(status_code=422, detail=f"{name} must be text or null")
    return values


def risk_band(probability: float) -> str:
    if probability < 0.20:
        return "Low"
    if probability < 0.50:
        return "Moderate"
    return "Elevated"


def score_features(
    model: Any,
    values: dict[str, Any],
    threshold: float,
    model_version: str = MODEL_VERSION,
) -> AssessmentOutput:
    validated = validate_features(values)
    columns = FEATURES["numeric_cols"] + FEATURES["categorical_cols"]
    frame = pd.DataFrame([validated], columns=columns)
    try:
        probability = float(model.predict_proba(frame)[0, 1])
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail="Feature values do not match the model schema") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Model inference failed") from exc

    flagged = probability >= threshold
    return AssessmentOutput(
        model_version=model_version,
        probability_of_default=round(probability, 6),
        risk_band=risk_band(probability),
        review_flag=flagged,
        interpretation=(
            "The demonstration score exceeds the configured review threshold. "
            "This is not an approval or rejection recommendation."
            if flagged
            else "The demonstration score is below the configured review threshold. "
            "Independent human assessment is still required."
        ),
        disclaimer=DISCLAIMER,
    )


def explain_features(model: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Explain the base logistic score using mean fold-wise feature contributions.

    Contributions are on the uncalibrated log-odds scale. They are not additive
    probability changes and do not explain the subsequent sigmoid calibration.
    """
    validated = validate_features(values)
    columns = FEATURES["numeric_cols"] + FEATURES["categorical_cols"]
    frame = pd.DataFrame([validated], columns=columns)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    fitted = getattr(model, "calibrated_classifiers_", [])
    if not fitted:
        raise HTTPException(status_code=501, detail="This model artifact does not expose fold estimators")

    for calibrated_fold in fitted:
        estimator = getattr(calibrated_fold, "estimator", None)
        if estimator is None:
            continue
        preprocessor = estimator.named_steps.get("preprocessor")
        classifier = estimator.named_steps.get("classifier")
        if preprocessor is None or classifier is None or not hasattr(classifier, "coef_"):
            continue
        transformed = preprocessor.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_values = transformed[0]
        feature_names = preprocessor.get_feature_names_out()
        contributions = feature_values * classifier.coef_[0]
        for name, contribution in zip(feature_names, contributions, strict=False):
            key = str(name)
            totals[key] = totals.get(key, 0.0) + float(contribution)
            counts[key] = counts.get(key, 0) + 1

    if not totals:
        raise HTTPException(status_code=501, detail="Feature contributions are unavailable for this model")

    averaged = [
        {"feature": name, "contribution_log_odds": round(value / counts[name], 5)}
        for name, value in totals.items()
    ]
    positive = sorted(
        (item for item in averaged if item["contribution_log_odds"] > 0),
        key=lambda item: item["contribution_log_odds"],
        reverse=True,
    )[:5]
    negative = sorted(
        (item for item in averaged if item["contribution_log_odds"] < 0),
        key=lambda item: item["contribution_log_odds"],
    )[:5]
    return {
        "method": "Mean fold-wise logistic coefficient contributions",
        "scale": "Uncalibrated log-odds contribution, not probability points",
        "risk_increasing": positive,
        "risk_decreasing": negative,
        "warning": (
            "These are model-specific associations, not causal effects. "
            "Contributions explain the base logistic score, not the calibrated probability. "
            "They do not establish why a person will default."
        ),
    }


def build_portfolio_summary(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"Default", "Loan_Amount", "Monthly_Income"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Portfolio data is missing required fields: {missing}")
    return {
        "applications": int(len(frame)),
        "defaults": int(frame["Default"].sum()),
        "default_rate": round(float(frame["Default"].mean()), 4),
        "total_requested_exposure_mwk": round(float(frame["Loan_Amount"].sum()), 2),
        "average_loan_amount_mwk": round(float(frame["Loan_Amount"].mean()), 2),
        "median_monthly_income_mwk": round(float(frame["Monthly_Income"].median()), 2),
        "average_loan_to_income_ratio": round(float(frame["Loan_Amount"].div(frame["Monthly_Income"].where(frame["Monthly_Income"] > 0) * 12).mean()), 4),
        "data_notice": "Descriptive statistics from the committed synthetic dataset.",
    }


def grouped_portfolio(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in frame.columns:
        return []
    grouped = (
        frame.groupby(column, dropna=False)
        .agg(applications=("Default", "size"), defaults=("Default", "sum"), default_rate=("Default", "mean"), exposure_mwk=("Loan_Amount", "sum"))
        .reset_index()
        .sort_values("applications", ascending=False)
    )
    return [
        {
            "name": str(row[column]) if pd.notna(row[column]) else "Unknown",
            "applications": int(row["applications"]),
            "defaults": int(row["defaults"]),
            "default_rate": round(float(row["default_rate"]), 4),
            "exposure_mwk": round(float(row["exposure_mwk"]), 2),
        }
        for _, row in grouped.iterrows()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Run scripts/train_model_v2.py.")
    if not PORTFOLIO_PATH.exists():
        raise RuntimeError(f"Synthetic portfolio dataset not found at {PORTFOLIO_PATH}.")
    init_db()
    app.state.database_ready = True
    app.state.model = joblib.load(MODEL_PATH)
    app.state.portfolio = pd.read_csv(PORTFOLIO_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8")) if EVALUATION_PATH.exists() else {}
    app.state.metadata = metadata
    app.state.monitoring_reference = metadata.get("monitoring_reference")
    app.state.model_version = (
        os.getenv("CREDIT_RISK_MODEL_VERSION") or metadata.get("model_version", MODEL_VERSION)
    )
    app.state.threshold = float(os.getenv("CREDIT_RISK_THRESHOLD", metadata.get("decision_threshold", DEFAULT_THRESHOLD)))
    with SessionLocal() as session:
        register_model_version(
            session,
            model_version=app.state.model_version,
            model_type=metadata.get("model_type", "Unknown model type"),
            artifact_path=(
                str(MODEL_PATH.relative_to(ROOT))
                if MODEL_PATH.is_relative_to(ROOT)
                else str(MODEL_PATH)
            ),
            artifact_sha256=sha256_file(MODEL_PATH),
            dataset_sha256=metadata.get("dataset_sha256", sha256_file(PORTFOLIO_PATH)),
            metadata=metadata,
            evaluation=evaluation,
        )
    app.state.model_registry_ready = True
    yield
    app.state.model = None
    app.state.portfolio = None
    app.state.database_ready = False
    app.state.model_registry_ready = False


app = FastAPI(
    title="Nyanja Credit Risk Intelligence API",
    version="0.3.0",
    description=(
        "Synthetic credit-risk research API. The dataset and model are simulated. "
        "Predictions are not validated for real-world use and are not lending decisions."
    ),
    lifespan=lifespan,
)

origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/health", tags=["Operations"])
def health(request: Request) -> dict[str, Any]:
    loaded = getattr(request.app.state, "model", None) is not None
    data_loaded = getattr(request.app.state, "portfolio", None) is not None
    database_ready = getattr(request.app.state, "database_ready", False)
    registry_ready = getattr(request.app.state, "model_registry_ready", False)
    ready = loaded and data_loaded and database_ready and registry_ready
    return {
        "status": "ok" if ready else "degraded",
        "model_loaded": loaded,
        "portfolio_loaded": data_loaded,
        "database_ready": database_ready,
        "model_registry_ready": registry_ready,
        "model_version": getattr(request.app.state, "model_version", MODEL_VERSION),
    }


@app.get("/api/v1/model", tags=["Model"])
def model_information(request: Request) -> dict[str, Any]:
    metadata = getattr(request.app.state, "metadata", {})
    return {
        "model_version": getattr(request.app.state, "model_version", MODEL_VERSION),
        "model_type": metadata.get("model_type", "Calibrated Logistic Regression"),
        "feature_count": len(REQUIRED_FIELDS),
        "numeric_features": len(NUMERIC_FIELDS),
        "categorical_features": len(CATEGORICAL_FIELDS),
        "excluded_demographic_fields": FEATURES["audit_only_cols"],
        "review_threshold": getattr(request.app.state, "threshold", DEFAULT_THRESHOLD),
        "data_notice": "All training records are synthetic and created for demonstration.",
        "use_restriction": "Educational research only. Never use for real lending decisions.",
    }


@app.get("/api/v1/model/registry", tags=["Model"])
def model_registry(request: Request) -> dict[str, Any]:
    if not getattr(request.app.state, "model_registry_ready", False):
        raise HTTPException(status_code=503, detail="Model registry is not ready")
    with SessionLocal() as session:
        versions = get_model_registry(session)
    return {
        "active_version": getattr(request.app.state, "model_version", MODEL_VERSION),
        "versions": versions,
        "data_notice": DISCLAIMER,
    }


@app.get("/api/v1/model/evaluation", tags=["Model"])
def model_evaluation() -> dict[str, Any]:
    if not EVALUATION_PATH.exists():
        raise HTTPException(status_code=503, detail="Model evaluation report is unavailable")
    return json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))


@app.get("/api/v1/model/temporal-evaluation", tags=["Model"])
def temporal_model_evaluation() -> dict[str, Any]:
    if not TEMPORAL_EVALUATION_PATH.exists():
        raise HTTPException(status_code=503, detail="Synthetic chronological evaluation is unavailable")
    return json.loads(TEMPORAL_EVALUATION_PATH.read_text(encoding="utf-8"))


@app.get("/api/v1/portfolio/summary", tags=["Portfolio"])
def portfolio_summary(request: Request) -> dict[str, Any]:
    frame = getattr(request.app.state, "portfolio", None)
    if frame is None:
        raise HTTPException(status_code=503, detail="Portfolio data is not ready")
    return build_portfolio_summary(frame)


@app.get("/api/v1/portfolio/breakdowns", tags=["Portfolio"])
def portfolio_breakdowns(request: Request) -> dict[str, Any]:
    frame = getattr(request.app.state, "portfolio", None)
    if frame is None:
        raise HTTPException(status_code=503, detail="Portfolio data is not ready")
    return {
        "by_branch": grouped_portfolio(frame, "Branch_Location"),
        "by_customer_segment": grouped_portfolio(frame, "Customer_Segment"),
        "by_loan_purpose": grouped_portfolio(frame, "Loan_Purpose"),
        "data_notice": "All breakdowns describe synthetic records, not real banking performance.",
    }


@app.get("/api/v1/assessments/sample", tags=["Assessments"])
def sample_application() -> dict[str, Any]:
    sample_path = Path(os.getenv("CREDIT_RISK_SAMPLE_PATH", ROOT / "data/processed/loan_applications_clean.csv"))
    if not sample_path.exists():
        raise HTTPException(status_code=503, detail="Synthetic sample records are unavailable")
    sample_frame = pd.read_csv(sample_path, nrows=1)
    if sample_frame.empty:
        raise HTTPException(status_code=503, detail="No synthetic sample records are available")
    row = sample_frame.iloc[0]
    features = {}
    for name in FEATURES["numeric_cols"] + FEATURES["categorical_cols"]:
        value = row.get(name)
        if pd.isna(value):
            features[name] = None
        elif hasattr(value, "item"):
            features[name] = value.item()
        else:
            features[name] = value
    return {
        "features": features,
        "data_notice": "One synthetic demonstration application. Not a real customer record.",
    }


@app.post("/api/v1/assessments", response_model=AssessmentOutput, tags=["Assessments"])
def assess_application(payload: AssessmentInput, request: Request) -> AssessmentOutput:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    result = score_features(
        model, payload.features, request.app.state.threshold, request.app.state.model_version
    )
    with SessionLocal() as session:
        record_assessment(
            session,
            model_version=result.model_version,
            probability_of_default=result.probability_of_default,
            risk_band=result.risk_band,
            review_flag=result.review_flag,
            source="single",
        )
    return result


@app.post("/api/v1/assessments/explain", tags=["Assessments"])
def explain_application(payload: AssessmentInput, request: Request) -> dict[str, Any]:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    assessment = score_features(
        model, payload.features, request.app.state.threshold, request.app.state.model_version
    )
    explanation = explain_features(model, payload.features)
    with SessionLocal() as session:
        record_assessment(
            session,
            model_version=assessment.model_version,
            probability_of_default=assessment.probability_of_default,
            risk_band=assessment.risk_band,
            review_flag=assessment.review_flag,
            source="single",
        )
    return {**assessment.model_dump(), "explanation": explanation}


@app.post("/api/v1/monitoring/drift", tags=["Monitoring"])
def monitor_batch_drift(payload: BatchAssessmentInput, request: Request) -> dict[str, Any]:
    """Compare an unlabeled scoring batch with the model's training reference."""
    model = getattr(request.app.state, "model", None)
    reference = getattr(request.app.state, "monitoring_reference", None)
    if model is None or reference is None:
        raise HTTPException(status_code=503, detail="Model monitoring reference is unavailable")

    valid_rows: list[dict[str, Any]] = []
    probabilities: list[float] = []
    errors: list[dict[str, Any]] = []
    for index, features in enumerate(payload.applications):
        try:
            validated = validate_features(features)
            frame = pd.DataFrame(
                [validated],
                columns=FEATURES["numeric_cols"] + FEATURES["categorical_cols"],
            )
            probability = float(model.predict_proba(frame)[0, 1])
            valid_rows.append(validated)
            probabilities.append(probability)
        except HTTPException as exc:
            errors.append({"row": index + 1, "status_code": exc.status_code, "detail": exc.detail})
        except (ValueError, TypeError, KeyError):
            errors.append({
                "row": index + 1,
                "status_code": 422,
                "detail": "Feature values do not match the model schema",
            })

    if not valid_rows:
        raise HTTPException(status_code=422, detail="No valid rows available for drift monitoring")
    report = compare_distributions(
        reference,
        pd.DataFrame(valid_rows, columns=FEATURES["numeric_cols"] + FEATURES["categorical_cols"]),
        np.asarray(probabilities, dtype=float),
    )
    return {
        "model_version": request.app.state.model_version,
        "submitted": len(payload.applications),
        "monitored": len(valid_rows),
        "rejected": len(errors),
        "drift": report,
        "errors": errors,
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/v1/assessments/batch", tags=["Assessments"])
def assess_batch(payload: BatchAssessmentInput, request: Request) -> dict[str, Any]:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    results, errors = [], []
    for index, features in enumerate(payload.applications):
        try:
            scored = score_features(
                model, features, request.app.state.threshold, request.app.state.model_version
            )
            with SessionLocal() as session:
                record_assessment(
                    session,
                    model_version=scored.model_version,
                    probability_of_default=scored.probability_of_default,
                    risk_band=scored.risk_band,
                    review_flag=scored.review_flag,
                    source="batch",
                )
            results.append({"row": index + 1, **scored.model_dump()})
        except HTTPException as exc:
            errors.append({"row": index + 1, "status_code": exc.status_code, "detail": exc.detail})
    return {
        "model_version": request.app.state.model_version,
        "submitted": len(payload.applications),
        "scored": len(results),
        "rejected": len(errors),
        "results": results,
        "errors": errors,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/v1/audit/summary", tags=["Audit"])
def audit_summary(request: Request) -> dict[str, Any]:
    if not getattr(request.app.state, "database_ready", False):
        raise HTTPException(status_code=503, detail="Audit database is not ready")
    with SessionLocal() as session:
        return get_audit_summary(session)

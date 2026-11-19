from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from api.main import (
    BINARY_FIELDS,
    CATEGORICAL_FIELDS,
    NUMERIC_FIELDS,
    REQUIRED_FIELDS,
    build_portfolio_summary,
    explain_features,
    grouped_portfolio,
    health,
    risk_band,
    validate_features,
)


def valid_payload():
    values = dict.fromkeys(NUMERIC_FIELDS, 0)
    values.update(dict.fromkeys(CATEGORICAL_FIELDS, "Unknown"))
    values.update(dict.fromkeys(BINARY_FIELDS, 0))
    return values


def test_feature_schema_is_complete_and_excludes_demographic_fields():
    from src.models.schema import AUDIT_ONLY_FEATURES

    assert len(REQUIRED_FIELDS) == 33
    assert set(AUDIT_ONLY_FEATURES).isdisjoint(REQUIRED_FIELDS)


def test_feature_validation_rejects_missing_fields():
    with pytest.raises(HTTPException) as error:
        validate_features({})
    assert error.value.status_code == 422


def test_feature_validation_rejects_identifiers_and_unexpected_fields():
    values = valid_payload()
    values["Customer_ID"] = "C-0001"
    with pytest.raises(HTTPException) as error:
        validate_features(values)
    assert error.value.status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_feature_validation_rejects_non_finite_values(value):
    values = valid_payload()
    numeric_name = next(iter(NUMERIC_FIELDS))
    values[numeric_name] = value
    with pytest.raises(HTTPException) as error:
        validate_features(values)
    assert error.value.status_code == 422


def test_binary_features_only_accept_zero_or_one():
    values = valid_payload()
    values["Has_Credit_Card"] = 2
    with pytest.raises(HTTPException) as error:
        validate_features(values)
    assert error.value.status_code == 422


def test_valid_feature_payload_is_accepted():
    values = valid_payload()
    assert set(validate_features(values)) == REQUIRED_FIELDS


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, "Low"), (0.199, "Low"), (0.2, "Moderate"), (0.499, "Moderate"), (0.5, "Elevated")],
)
def test_risk_band_boundaries(probability, expected):
    assert risk_band(probability) == expected


def test_health_endpoint_reports_model_state():
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(model=None, portfolio=None)
        )
    )
    result = health(request)
    assert result["status"] == "degraded"
    assert result["model_loaded"] is False
    assert result["portfolio_loaded"] is False


def test_portfolio_summary_uses_annual_income_denominator():
    frame = pd.DataFrame(
        {
            "Default": [0, 1],
            "Loan_Amount": [12000, 24000],
            "Monthly_Income": [1000, 2000],
        }
    )
    result = build_portfolio_summary(frame)
    assert result["applications"] == 2
    assert result["default_rate"] == 0.5
    assert result["average_loan_to_income_ratio"] == 1.0


def test_grouped_portfolio_returns_aggregates():
    frame = pd.DataFrame(
        {
            "Branch_Location": ["Blantyre", "Blantyre", "Lilongwe"],
            "Default": [0, 1, 0],
            "Loan_Amount": [100, 200, 300],
        }
    )
    result = grouped_portfolio(frame, "Branch_Location")
    assert result[0]["name"] == "Blantyre"
    assert result[0]["applications"] == 2
    assert result[0]["default_rate"] == 0.5


def test_local_explanation_returns_positive_and_negative_contributors():
    class Preprocessor:
        def transform(self, frame):
            return np.array([[2.0, -1.0]])

        def get_feature_names_out(self):
            return np.array(["numeric__Monthly_Income", "categorical__Loan_Purpose"])

    class Classifier:
        coef_ = np.array([[0.5, 1.0]])

    estimator = SimpleNamespace(
        named_steps={
            "preprocessor": Preprocessor(),
            "classifier": Classifier(),
        }
    )
    model = SimpleNamespace(
        calibrated_classifiers_=[SimpleNamespace(estimator=estimator)]
    )
    result = explain_features(model, valid_payload())
    assert result["risk_increasing"][0]["feature"] == "numeric__Monthly_Income"
    assert result["risk_decreasing"][0]["feature"] == "categorical__Loan_Purpose"
    assert "not causal" in result["warning"]


def test_audit_repository_stores_only_prediction_metadata():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from api.db import get_audit_summary, record_assessment
    from api.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        record_assessment(
            session,
            model_version="test-v1",
            probability_of_default=0.37,
            risk_band="Moderate",
            review_flag=False,
            source="single",
        )
        summary = get_audit_summary(session)
    assert summary["assessments_recorded"] == 1
    assert summary["review_flags"] == 0
    assert summary["risk_band_counts"] == {"Moderate": 1}
    assert "applicant_features" not in summary["stored_fields"]
    engine.dispose()


def test_database_url_normalisation_uses_psycopg_three():
    from api.db import normalize_database_url

    assert normalize_database_url("postgres://user:pass@host/db") == (
        "postgresql+psycopg://user:pass@host/db"
    )
    assert normalize_database_url("postgresql://user:pass@host/db") == (
        "postgresql+psycopg://user:pass@host/db"
    )
    assert normalize_database_url("sqlite:///local.db") == "sqlite:///local.db"

"""Tests for synthetic chronological evaluation safeguards."""

import numpy as np
import pandas as pd
import pytest

from scripts.run_temporal_validation import (
    build_temporal_scenario,
    chronological_split,
)


def source_frame(rows: int = 2400) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    employment = rng.choice(
        ["Employed - Government", "Employed - Private", "Self-Employed", "Unemployed"],
        size=rows,
    )
    segment = rng.choice(
        ["Salaried Mass Market", "Salaried Premium", "SME", "Self-Employed", "Youth/Student"],
        size=rows,
    )
    income = rng.lognormal(12.5, 0.4, rows)
    loan = income * rng.uniform(1.5, 8, rows)
    return pd.DataFrame(
        {
            "Customer_ID": [f"C{i}" for i in range(rows)],
            "Loan_ID": [f"L{i}" for i in range(rows)],
            "Monthly_Income": income,
            "Monthly_Repayment": loan / 18,
            "Loan_Amount": loan,
            "Debt_to_Income_Ratio": rng.uniform(0.05, 0.8, rows),
            "Loan_to_Income_Ratio": loan / (income * 12),
            "Previous_Default_Count": rng.poisson(0.2, rows),
            "Number_of_Late_Payments": rng.poisson(1.0, rows),
            "Days_Past_Due_History": rng.integers(0, 40, rows),
            "Existing_Loan_Count": rng.poisson(0.6, rows),
            "Savings_Balance": rng.lognormal(10, 1, rows),
            "Account_Tenure_Months": rng.integers(1, 120, rows),
            "Credit_Score": rng.integers(400, 850, rows),
            "Employment_Status": employment,
            "Customer_Segment": segment,
            "Default": rng.binomial(1, 0.2, rows),
        }
    )


def test_temporal_scenario_is_reproducible_and_explicitly_synthetic():
    source = source_frame()
    first = build_temporal_scenario(source, seed=9)
    second = build_temporal_scenario(source, seed=9)
    pd.testing.assert_frame_equal(first, second)
    assert first["Application_Date"].notna().all()
    assert first["Macro_Stress_Index"].between(0.05, 0.90).all()
    assert first["Default"].isin([0, 1]).all()
    assert first["Scenario_Month"].nunique() == 36


def test_chronological_split_has_no_time_overlap():
    scenario = build_temporal_scenario(source_frame(), seed=7)
    train, validation, test = chronological_split(scenario)
    assert train["Scenario_Month"].max() < validation["Scenario_Month"].min()
    assert validation["Scenario_Month"].max() < test["Scenario_Month"].min()
    assert train["Application_Date"].max() < validation["Application_Date"].min()
    assert validation["Application_Date"].max() < test["Application_Date"].min()


def test_chronological_split_rejects_invalid_windows():
    scenario = build_temporal_scenario(source_frame(), seed=3)
    with pytest.raises(ValueError):
        chronological_split(scenario, train_months=34, validation_months=2)


def test_dates_and_scenario_variables_are_not_in_model_schema():
    from src.models.schema import get_inference_feature_columns

    groups = get_inference_feature_columns()
    fields = set(groups["numeric_cols"] + groups["categorical_cols"])
    assert {
        "Application_Date",
        "Scenario_Month",
        "Macro_Stress_Index",
        "Scenario_Default_Probability",
    }.isdisjoint(fields)

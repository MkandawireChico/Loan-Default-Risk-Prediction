"""Run a chronological validation experiment on explicitly synthetic credit data.

This is a methodological stress test, not evidence of real-world model validity.
The script creates synthetic application dates and a time-varying macro-stress
scenario, regenerates the synthetic outcome, and evaluates on later periods.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_model_v2 import (  # noqa: E402
    build_pipeline,
    clean_partition,
    evaluate,
    threshold_table,
)
from src.models.schema import get_inference_feature_columns  # noqa: E402

LOGGER = logging.getLogger("temporal_validation")
START_DATE = "2023-01-01"


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    """Return a finite numeric vector for deterministic scenario construction."""
    if column not in frame:
        return np.full(len(frame), default, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.fillna(values.median() if values.notna().any() else default).to_numpy(dtype=float)


def build_temporal_scenario(
    source: pd.DataFrame, seed: int = 42, months: int = 36
) -> pd.DataFrame:
    """Assign dates and simulate a documented macro-stress shift in the target."""
    if months < 12:
        raise ValueError("At least 12 months are required for temporal evaluation")
    if "Default" not in source:
        raise ValueError("Source data must contain the original synthetic Default field")

    frame = source.drop_duplicates().reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    month_index = rng.integers(0, months, size=len(frame))
    day = rng.integers(1, 28, size=len(frame))
    dates = pd.to_datetime({
        "year": 2023 + month_index // 12,
        "month": month_index % 12 + 1,
        "day": day,
    })
    frame["Application_Date"] = dates
    frame["Scenario_Month"] = month_index

    seasonal = 0.08 * np.sin(2 * np.pi * month_index / 12)
    structural_shift = np.where(month_index >= months - 8, 0.32, 0.0)
    stress = np.clip(0.30 + seasonal + structural_shift, 0.05, 0.90)
    frame["Macro_Stress_Index"] = np.round(stress, 4)

    income = np.maximum(_numeric(frame, "Monthly_Income", 1.0), 1.0)
    repayment_ratio = _numeric(frame, "Monthly_Repayment", 0.0) / income
    dti = _numeric(frame, "Debt_to_Income_Ratio")
    lti = _numeric(frame, "Loan_to_Income_Ratio")
    previous_defaults = _numeric(frame, "Previous_Default_Count")
    late_payments = _numeric(frame, "Number_of_Late_Payments")
    days_past_due = _numeric(frame, "Days_Past_Due_History")
    existing_loans = _numeric(frame, "Existing_Loan_Count")
    savings = np.maximum(_numeric(frame, "Savings_Balance"), 0.0)
    tenure = np.maximum(_numeric(frame, "Account_Tenure_Months"), 0.0)
    credit_score = _numeric(frame, "Credit_Score", 650.0)
    employment = frame.get(
        "Employment_Status", pd.Series("Unknown", index=frame.index)
    ).fillna("Unknown")
    segment = frame.get(
        "Customer_Segment", pd.Series("Unknown", index=frame.index)
    ).fillna("Unknown")

    log_odds = (
        -2.1
        + 2.6 * dti
        + 1.4 * lti
        + 0.9 * previous_defaults
        + 0.18 * late_payments
        + 0.010 * days_past_due
        + 1.1 * repayment_ratio
        + 0.5 * existing_loans
        - 0.35 * np.log1p(savings / 10_000)
        - 0.30 * np.log1p(tenure)
        - 0.004 * (credit_score - 650) / 10
        - 0.15 * (employment == "Employed - Government").to_numpy(dtype=int)
        - 0.10 * (employment == "Employed - Private").to_numpy(dtype=int)
        + 0.35 * (employment == "Unemployed").to_numpy(dtype=int)
        + 0.15 * (segment == "Youth/Student").to_numpy(dtype=int)
        - 0.10 * (segment == "Salaried Premium").to_numpy(dtype=int)
        + 0.75 * stress
        + 0.30 * stress * (employment == "Self-Employed").to_numpy(dtype=int)
        + rng.normal(0, 0.55, len(frame))
    )
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -20, 20)))
    frame["Default"] = rng.binomial(1, probabilities)
    frame["Scenario_Default_Probability"] = probabilities
    return frame.sort_values("Application_Date").reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame, train_months: int = 24, validation_months: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split whole calendar months; never shuffle records across time boundaries."""
    required = {"Application_Date", "Scenario_Month", "Default"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Temporal dataset is missing fields: {sorted(missing)}")
    month_count = int(frame["Scenario_Month"].nunique())
    if (
        train_months < 6
        or validation_months < 2
        or train_months + validation_months >= month_count
    ):
        raise ValueError("Train and validation windows must leave at least one future test month")

    train = frame.loc[frame["Scenario_Month"] < train_months].copy()
    validation = frame.loc[
        (frame["Scenario_Month"] >= train_months)
        & (frame["Scenario_Month"] < train_months + validation_months)
    ].copy()
    test = frame.loc[frame["Scenario_Month"] >= train_months + validation_months].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Each chronological partition must contain records")
    return train, validation, test


def monthly_diagnostics(
    frame: pd.DataFrame, probabilities: np.ndarray
) -> list[dict[str, float | int | str | None]]:
    """Report performance by future month to expose instability hidden by aggregates."""
    scored = frame[["Application_Date", "Default"]].copy()
    scored["probability"] = probabilities
    scored["month"] = pd.to_datetime(scored["Application_Date"]).dt.to_period("M").astype(str)
    result = []
    for month, group in scored.groupby("month", sort=True):
        y = group["Default"].to_numpy(dtype=int)
        p = group["probability"].to_numpy(dtype=float)
        result.append(
            {
                "month": month,
                "applications": int(len(group)),
                "observed_default_rate": round(float(y.mean()), 4),
                "mean_predicted_probability": round(float(p.mean()), 4),
                "roc_auc": round(float(roc_auc_score(y, p)), 4)
                if len(np.unique(y)) == 2
                else None,
                "brier_score": round(float(brier_score_loss(y, p)), 4),
            }
        )
    return result


def run_experiment(
    source: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    review_capacity: float = 0.30,
) -> dict:
    if not 0 < review_capacity <= 1:
        raise ValueError("review_capacity must be in (0, 1]")
    scenario = build_temporal_scenario(source, seed=seed)
    train_raw, validation_raw, test_raw = chronological_split(scenario)

    # Fit all clipping limits on the historical training window only.
    income_cap = float(train_raw["Monthly_Income"].quantile(0.995))
    loan_cap = float(train_raw["Loan_Amount"].quantile(0.995))
    train = clean_partition(train_raw, income_cap, loan_cap)
    validation = clean_partition(validation_raw, income_cap, loan_cap)
    test = clean_partition(test_raw, income_cap, loan_cap)

    groups = get_inference_feature_columns()
    numeric, categorical = groups["numeric_cols"], groups["categorical_cols"]
    features = numeric + categorical
    missing = sorted(set(features + ["Default"]) - set(train.columns))
    if missing:
        raise ValueError(f"Training data is missing model fields: {missing}")

    # Training rows are ordered before expanding-window calibration folds.
    train = train.sort_values("Application_Date")
    base = build_pipeline(numeric, categorical)
    calibrated = CalibratedClassifierCV(
        estimator=base, method="sigmoid", cv=TimeSeriesSplit(n_splits=5)
    )
    calibrated.fit(train[features], train["Default"])

    validation_probabilities = calibrated.predict_proba(validation[features])[:, 1]
    candidates = threshold_table(validation["Default"].to_numpy(), validation_probabilities)
    feasible = [row for row in candidates if row["review_rate"] <= review_capacity]
    if not feasible:
        raise RuntimeError("No validation threshold meets the review-capacity constraint")
    selected = max(feasible, key=lambda row: (row["recall"], row["precision"]))
    threshold = float(selected["threshold"])

    test_probabilities = calibrated.predict_proba(test[features])[:, 1]
    test_metrics = evaluate(test["Default"].to_numpy(), test_probabilities, threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "temporal_synthetic_model.joblib"
    joblib.dump(calibrated, model_path)
    report = {
        "experiment": "synthetic_chronological_stress_test_v1",
        "data_notice": (
            "Dates, macro-stress values and outcomes are simulated. "
            "This experiment is not evidence about Malawi or any real lender."
        ),
        "seed": seed,
        "scenario": {
            "start_date": START_DATE,
            "months": int(scenario["Scenario_Month"].nunique()),
            "structural_stress_shift_starts_at_month": int(
                scenario["Scenario_Month"].max() - 7
            ),
            "features_excluded_from_model": [
                "Application_Date",
                "Scenario_Month",
                "Macro_Stress_Index",
                "Scenario_Default_Probability",
            ],
        },
        "partitions": {
            "train": {
                "rows": int(len(train)),
                "start": str(train["Application_Date"].min().date()),
                "end": str(train["Application_Date"].max().date()),
            },
            "validation": {
                "rows": int(len(validation)),
                "start": str(validation["Application_Date"].min().date()),
                "end": str(validation["Application_Date"].max().date()),
            },
            "test": {
                "rows": int(len(test)),
                "start": str(test["Application_Date"].min().date()),
                "end": str(test["Application_Date"].max().date()),
            },
        },
        "training": {
            "calibration": (
                "Sigmoid calibration with expanding-window TimeSeriesSplit "
                "inside the training window."
            ),
            "income_cap": income_cap,
            "loan_cap": loan_cap,
            "review_capacity": review_capacity,
            "threshold_selection": "Highest validation recall subject to review capacity.",
            "selected_validation_threshold": selected,
        },
        "future_test": test_metrics,
        "monthly_future_test_diagnostics": monthly_diagnostics(test, test_probabilities),
        "model_artifact": (
            str(model_path.relative_to(ROOT))
            if model_path.is_relative_to(ROOT)
            else str(model_path)
        ),
        "limitations": [
            "Entirely synthetic scenario and target.",
            "The macro-stress shift is deliberately simulated.",
            "No external validation or real borrower outcomes.",
            "No lending decisions or policy recommendations.",
        ],
    }
    report_path = output_dir / "temporal_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    scenario.to_csv(output_dir / "temporal_synthetic_dataset.csv", index=False)
    LOGGER.info("Temporal experiment report: %s", report_path)
    LOGGER.info("Future-window test metrics: %s", json.dumps(test_metrics, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/raw/loan_applications_synthetic.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/temporal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-capacity", type=float, default=0.30)
    args = parser.parse_args()
    source = pd.read_csv(args.data)
    run_experiment(source, args.output_dir, args.seed, args.review_capacity)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

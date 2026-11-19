"""Train a leakage-conscious, calibrated demonstration model.

The source dataset is synthetic. This experiment uses stratified train,
validation and test partitions. Calibration is fitted within training data;
the validation partition selects a review threshold; the test partition is
used once for final reporting. Demographic audit fields are excluded.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.engineering import engineer_all_features  # noqa: E402
from src.models.schema import get_inference_feature_columns  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_v2")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if "Default" not in frame:
        raise ValueError("Dataset must contain the binary Default target")
    frame = frame.drop_duplicates().reset_index(drop=True)
    frame["Default"] = pd.to_numeric(frame["Default"], errors="raise").astype(int)
    if not set(frame["Default"].unique()).issubset({0, 1}):
        raise ValueError("Default target must contain only 0 and 1")
    return frame


def prepare_partitions(frame: pd.DataFrame, seed: int):
    train_val, test = train_test_split(
        frame, test_size=0.20, random_state=seed, stratify=frame["Default"]
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=seed,
        stratify=train_val["Default"],
    )
    return train.copy(), validation.copy(), test.copy()


def clean_partition(frame: pd.DataFrame, income_cap: float, loan_cap: float):
    result = frame.copy()
    employment_map = {
        "employed - government": "Employed - Government",
        "Employed-Private": "Employed - Private",
        "SELF EMPLOYED": "Self-Employed",
        "Buisness Owner": "Business Owner",
        "unemployed ": "Unemployed",
    }
    purpose_map = {
        "personal ": "Personal",
        "biz/sme": "Business/SME",
        "home improvement": "Home Improvement",
        "AGRICULTURE": "Agriculture",
    }
    if "Employment_Status" in result:
        result["Employment_Status"] = result["Employment_Status"].replace(employment_map)
    if "Loan_Purpose" in result:
        result["Loan_Purpose"] = result["Loan_Purpose"].replace(purpose_map)
    result["Monthly_Income"] = result["Monthly_Income"].clip(upper=income_cap)
    result["Loan_Amount"] = result["Loan_Amount"].clip(upper=loan_cap)
    result.loc[(result["Age"] < 18) | (result["Age"] > 90), "Age"] = np.nan
    result["Branch_Location"] = result["Branch_Location"].fillna("Unknown")
    return engineer_all_features(result)


def build_pipeline(numeric: list[str], categorical: list[str]):
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )


def threshold_table(y_true: np.ndarray, probabilities: np.ndarray):
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "review_rate": float(predictions.mean()),
            }
        )
    return rows


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "sample_size": int(len(y_true)),
        "default_rate": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "review_rate": float(predictions.mean()),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96):
    """Return a two-sided 95% Wilson interval for a binomial proportion."""
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [round(float(max(0, centre - margin)), 4), round(float(min(1, centre + margin)), 4)]


def group_diagnostics(
    audit_frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    group_columns: list[str],
    minimum_group_size: int = 30,
):
    """Summarise subgroup outcomes with uncertainty intervals and small-cell suppression."""
    predictions = (probabilities >= threshold).astype(int)
    results = {}
    for column in group_columns:
        if column not in audit_frame:
            continue
        frame = audit_frame[[column]].copy()
        if column == "Age":
            frame["Age_Group"] = pd.cut(
                frame[column],
                bins=[0, 24, 34, 44, 54, 64, 100],
                labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
                right=True,
            )
            group_column = "Age_Group"
        else:
            group_column = column

        rows = []
        for group_name, indices in frame.groupby(group_column, dropna=False).groups.items():
            positions = audit_frame.index.get_indexer(indices)
            group_y = y_true[positions]
            group_pred = predictions[positions]
            n = len(positions)
            positives = int(group_y.sum())
            negatives = n - positives
            tp = int(((group_y == 1) & (group_pred == 1)).sum())
            fp = int(((group_y == 0) & (group_pred == 1)).sum())
            if n < minimum_group_size:
                rows.append({
                    "group": "Unknown" if pd.isna(group_name) else str(group_name),
                    "sample_size": n,
                    "suppressed": True,
                    "suppression_reason": f"Fewer than {minimum_group_size} records",
                })
                continue
            rows.append({
                "group": "Unknown" if pd.isna(group_name) else str(group_name),
                "sample_size": n,
                "suppressed": False,
                "observed_default_rate": round(float(group_y.mean()), 4),
                "observed_default_rate_95ci": wilson_interval(positives, n),
                "model_review_rate": round(float(group_pred.mean()), 4),
                "recall": round(tp / positives, 4) if positives else None,
                "recall_95ci": wilson_interval(tp, positives),
                "false_positive_rate": round(fp / negatives, 4) if negatives else None,
                "false_positive_rate_95ci": wilson_interval(fp, negatives),
                "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "precision_95ci": wilson_interval(tp, tp + fp),
            })
        results[group_column] = rows
    return {
        "minimum_group_size": minimum_group_size,
        "groups": results,
        "interpretation": (
            "Descriptive subgroup diagnostics with 95% Wilson intervals. "
            "Small groups are suppressed. These metrics do not prove fairness, "
            "causality, legal compliance or suitability for lending."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "raw" / "loan_applications_synthetic.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-capacity", type=float, default=0.30)
    args = parser.parse_args()

    if not 0 < args.review_capacity <= 1:
        raise ValueError("--review-capacity must be in (0, 1]")
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir

    raw = load_dataset(args.data)
    train_raw, validation_raw, test_raw = prepare_partitions(raw, args.seed)

    # Learn outlier caps from training data only. Apply identical caps elsewhere.
    income_cap = float(train_raw["Monthly_Income"].quantile(0.995))
    loan_cap = float(train_raw["Loan_Amount"].quantile(0.995))
    train = clean_partition(train_raw, income_cap, loan_cap)
    validation = clean_partition(validation_raw, income_cap, loan_cap)
    test = clean_partition(test_raw, income_cap, loan_cap)

    feature_groups = get_inference_feature_columns()
    numeric = feature_groups["numeric_cols"]
    categorical = feature_groups["categorical_cols"]
    feature_names = numeric + categorical

    missing = sorted(set(feature_names + ["Default"]) - set(train.columns))
    if missing:
        raise ValueError(f"Dataset is missing required fields: {missing}")

    x_train, y_train = train[feature_names], train["Default"]
    x_validation, y_validation = validation[feature_names], validation["Default"]
    x_test, y_test = test[feature_names], test["Default"]

    base = build_pipeline(numeric, categorical)
    calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
    calibrated.fit(x_train, y_train)

    validation_probabilities = calibrated.predict_proba(x_validation)[:, 1]
    candidates = threshold_table(y_validation.to_numpy(), validation_probabilities)

    # Choose the highest-recall threshold within the stated operational review capacity.
    feasible = [row for row in candidates if row["review_rate"] <= args.review_capacity]
    if not feasible:
        raise RuntimeError("No threshold satisfies the requested review capacity")
    selected = max(feasible, key=lambda row: (row["recall"], row["precision"]))
    threshold = selected["threshold"]

    # The final test partition is touched only after model and threshold are fixed.
    test_probabilities = calibrated.predict_proba(x_test)[:, 1]
    test_report = evaluate(y_test.to_numpy(), test_probabilities, threshold)
    fairness_report = group_diagnostics(
        test_raw,
        y_test.to_numpy(),
        test_probabilities,
        threshold,
        ["Gender", "Age", "Employment_Status", "Customer_Segment", "Branch_Location"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "credit_risk_calibrated_v2.joblib"
    joblib.dump(calibrated, model_path)

    report = {
        "project": "Nyanja Credit Risk Intelligence",
        "data_notice": "Entirely synthetic demonstration data. Not evidence about real borrowers.",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "random_seed": args.seed,
        "dataset_rows_after_exact_deduplication": int(len(raw)),
        "partition_sizes": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "preprocessing": {
            "income_upper_cap_learned_from_train": income_cap,
            "loan_amount_upper_cap_learned_from_train": loan_cap,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "excluded_demographic_audit_fields": feature_groups["audit_only_cols"],
        },
        "calibration": "Sigmoid calibration using five-fold CV within training partition.",
        "threshold_selection": {
            "method": "Highest validation recall under review capacity constraint",
            "review_capacity": args.review_capacity,
            "selected_validation_operating_point": selected,
            "all_validation_thresholds": candidates,
        },
        "final_test": test_report,
        "subgroup_diagnostics": fairness_report,
        "model_artifact": str(model_path.relative_to(ROOT)),
        "limitations": [
            "Synthetic cross-sectional data only.",
            "No external or temporal validation.",
            "Threshold is a demonstration operating point, not a lending policy.",
            "Model must not be used for real credit decisions.",
        ],
    }
    report_path = args.output_dir / "evaluation_v2.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    metadata = {
        "model_version": "credit-risk-calibrated-v2",
        "model_type": "Logistic Regression with sigmoid probability calibration",
        "decision_threshold": float(threshold),
        "review_capacity_target": float(args.review_capacity),
        "feature_count": int(len(feature_names)),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "audit_only_features": feature_groups["audit_only_cols"],
        "trained_at_utc": report["created_at_utc"],
        "synthetic_data": True,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved model: %s", model_path)
    logger.info("Saved report: %s", report_path)
    logger.info("Held-out test metrics: %s", json.dumps(test_report, indent=2))


if __name__ == "__main__":
    main()

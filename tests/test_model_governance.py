import pandas as pd

from scripts.train_model_v2 import (
    bootstrap_confidence_intervals,
    calibration_diagnostics,
    group_diagnostics,
    prepare_partitions,
    threshold_table,
    wilson_interval,
)
from src.models.schema import AUDIT_ONLY_FEATURES, get_inference_feature_columns


def test_demographic_fields_are_excluded_from_model_features():
    groups = get_inference_feature_columns()
    model_fields = set(groups["numeric_cols"] + groups["categorical_cols"])
    assert set(AUDIT_ONLY_FEATURES).isdisjoint(model_fields)
    assert set(AUDIT_ONLY_FEATURES).issubset(groups["audit_only_cols"])


def test_partitioning_is_reproducible_and_disjoint():
    frame = pd.DataFrame(
        {"Loan_ID": range(100), "Default": [0, 1] * 50, "value": range(100)}
    )
    train, validation, test = prepare_partitions(frame, seed=19)
    assert len(train) == 60
    assert len(validation) == 20
    assert len(test) == 20
    assert set(train.Loan_ID).isdisjoint(validation.Loan_ID)
    assert set(train.Loan_ID).isdisjoint(test.Loan_ID)
    assert set(validation.Loan_ID).isdisjoint(test.Loan_ID)


def test_threshold_table_reports_review_capacity():
    import numpy as np

    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    rows = threshold_table(y, probabilities)
    assert rows
    assert all(0 <= row["review_rate"] <= 1 for row in rows)
    assert all(0 <= row["precision"] <= 1 for row in rows)



def test_wilson_interval_returns_valid_bounds():
    lower, upper = wilson_interval(40, 100)
    assert 0 <= lower < 0.4 < upper <= 1


def test_group_diagnostics_suppresses_small_groups():
    import numpy as np

    frame = pd.DataFrame({"Gender": ["A"] * 40 + ["B"] * 10})
    y = np.array([0, 1] * 25)
    probabilities = np.array([0.2, 0.8] * 25)
    report = group_diagnostics(frame, y, probabilities, 0.5, ["Gender"], minimum_group_size=30)
    rows = report["groups"]["Gender"]
    assert rows[0]["suppressed"] is False
    assert rows[1]["suppressed"] is True



def test_bootstrap_metric_intervals_are_reproducible_and_bounded():
    import numpy as np

    y = np.array([0, 1] * 50)
    probabilities = np.linspace(0.01, 0.99, 100)
    first = bootstrap_confidence_intervals(y, probabilities, n_bootstrap=100, seed=17)
    second = bootstrap_confidence_intervals(y, probabilities, n_bootstrap=100, seed=17)
    assert first == second
    for interval in first.values():
        assert 0 <= interval["lower_95"] <= interval["upper_95"] <= 1
        assert interval["bootstrap_replicates"] == 100


def test_calibration_diagnostics_cover_every_record():
    import numpy as np

    y = np.array([0, 1] * 50)
    probabilities = np.linspace(0.01, 0.99, 100)
    report = calibration_diagnostics(y, probabilities, n_bins=10)
    assert sum(row["applications"] for row in report["bins"]) == len(y)
    assert 0 <= report["expected_calibration_error"] <= 1
    assert all(
        row["observed_default_rate_95ci"] is not None for row in report["bins"]
    )

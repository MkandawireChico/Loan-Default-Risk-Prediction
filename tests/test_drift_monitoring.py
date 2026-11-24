import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import build_reference_profile, compare_distributions


def reference_frame():
    return pd.DataFrame(
        {
            "income": np.arange(100, dtype=float),
            "loan_amount": np.linspace(1000, 10000, 100),
            "purpose": ["Personal"] * 60 + ["Business"] * 40,
        }
    )


def test_identical_distribution_has_low_drift():
    reference = reference_frame()
    profile = build_reference_profile(
        reference, ["income", "loan_amount"], ["purpose"],
        prediction_probabilities=np.linspace(0.05, 0.80, len(reference)),
    )
    report = compare_distributions(
        profile, reference,
        prediction_probabilities=np.linspace(0.05, 0.80, len(reference)),
    )
    assert report["summary"]["overall_status"] == "low"
    assert report["summary"]["maximum_psi"] < 0.10
    assert report["summary"]["features_monitored"] == 3


def test_shifted_numeric_and_categorical_values_raise_drift():
    reference = reference_frame()
    profile = build_reference_profile(reference, ["income"], ["purpose"])
    current = pd.DataFrame(
        {"income": np.full(100, 10000.0), "purpose": ["Agriculture"] * 100}
    )
    report = compare_distributions(profile, current)
    assert report["features"]["income"]["status"] == "high"
    assert report["features"]["purpose"]["status"] == "high"
    assert report["features"]["purpose"]["unseen_category_rate"] == 1.0


def test_missingness_is_reported_separately_from_distribution_shift():
    reference = reference_frame()
    profile = build_reference_profile(reference, ["income"], [])
    current = pd.DataFrame({"income": [None] * 50 + list(range(50))})
    report = compare_distributions(profile, current)
    assert report["features"]["income"]["current_missing_rate"] == 0.5
    assert report["features"]["income"]["missing_rate_delta"] == 0.5


def test_empty_current_batch_is_rejected():
    reference = reference_frame()
    profile = build_reference_profile(reference, ["income"], [])
    with pytest.raises(ValueError, match="at least one row"):
        compare_distributions(profile, pd.DataFrame({"income": []}))


def test_reference_profile_requires_all_features():
    with pytest.raises(ValueError, match="missing features"):
        build_reference_profile(pd.DataFrame({"income": [1, 2]}), ["income"], ["purpose"])

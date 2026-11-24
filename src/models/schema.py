"""Feature governance for model training and post-hoc fairness auditing."""

from __future__ import annotations

from src.data.preprocessing import get_feature_columns

# Demographic attributes are retained in the dataset for descriptive and
# post-hoc fairness analysis, but are excluded from model inference.
AUDIT_ONLY_FEATURES = ["Age", "Gender", "Marital_Status", "Education_Level"]


def get_inference_feature_columns() -> dict[str, list[str]]:
    """Return model inputs with demographic audit fields excluded."""
    columns = get_feature_columns()
    return {
        "numeric_cols": [
            name for name in columns["numeric_cols"] if name not in AUDIT_ONLY_FEATURES
        ],
        "categorical_cols": [
            name for name in columns["categorical_cols"] if name not in AUDIT_ONLY_FEATURES
        ],
        "audit_only_cols": AUDIT_ONLY_FEATURES.copy(),
    }

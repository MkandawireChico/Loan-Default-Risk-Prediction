"""Reference-based data and prediction drift diagnostics.

Population Stability Index (PSI) is used as a descriptive alerting statistic.
Thresholds are configurable conventions, not universal evidence of model failure.
All outputs in this project describe synthetic data only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPSILON = 1e-6
DEFAULT_THRESHOLDS = {"low": 0.10, "moderate": 0.20}


def _normalise_counts(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return np.full(len(values), 1.0 / max(len(values), 1))
    return values / total


def _psi(reference: np.ndarray, current: np.ndarray) -> float:
    ref = np.clip(_normalise_counts(reference), EPSILON, None)
    cur = np.clip(_normalise_counts(current), EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _severity(value: float, thresholds: dict[str, float]) -> str:
    if value >= thresholds["moderate"]:
        return "high"
    if value >= thresholds["low"]:
        return "moderate"
    return "low"


def _numeric_profile(series: pd.Series, bins: int) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
    observed = values.dropna().to_numpy(dtype=float)
    if not len(observed):
        return {"kind": "numeric", "edges": [], "proportions": [], "missing_rate": 1.0}
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(observed, quantiles)).astype(float)
    if len(edges) == 1:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    counts, _ = np.histogram(observed, bins=edges)
    proportions = _normalise_counts(counts).tolist()
    return {
        "kind": "numeric",
        "edges": [float(value) for value in edges],
        "proportions": [float(value) for value in proportions],
        "missing_rate": float(values.isna().mean()),
    }


def _categorical_profile(series: pd.Series, max_categories: int) -> dict[str, Any]:
    values = series.fillna("__MISSING__").astype(str)
    top = values.value_counts().head(max_categories).index.tolist()
    categories = sorted(set(top) | {"__OTHER__"})
    counts = values.where(values.isin(top), "__OTHER__").value_counts()
    proportions = _normalise_counts(np.array([counts.get(key, 0) for key in categories])).tolist()
    return {
        "kind": "categorical",
        "categories": categories,
        "proportions": [float(value) for value in proportions],
        "missing_rate": float(series.isna().mean()),
    }


def build_reference_profile(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    prediction_probabilities: np.ndarray | None = None,
    bins: int = 10,
    max_categories: int = 20,
) -> dict[str, Any]:
    """Create a compact reference distribution from training features."""
    missing = sorted(set(numeric_features + categorical_features) - set(frame.columns))
    if missing:
        raise ValueError(f"Reference frame is missing features: {missing}")
    profile: dict[str, Any] = {
        "method": "Quantile-binned numeric PSI and top-category categorical PSI",
        "reference_rows": int(len(frame)),
        "features": {},
    }
    for name in numeric_features:
        profile["features"][name] = _numeric_profile(frame[name], bins)
    for name in categorical_features:
        profile["features"][name] = _categorical_profile(frame[name], max_categories)
    if prediction_probabilities is not None:
        profile["prediction_probability"] = _numeric_profile(
            pd.Series(np.asarray(prediction_probabilities, dtype=float)), bins
        )
    return profile


def _numeric_drift(reference: dict[str, Any], current: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(current, errors="coerce")
    edges = np.asarray(reference["edges"], dtype=float)
    observed = values.dropna().to_numpy(dtype=float)
    if len(edges) < 2:
        return {"psi": None, "missing_rate": float(values.isna().mean()), "status": "unavailable"}
    clipped = np.clip(observed, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    ref = np.asarray(reference["proportions"], dtype=float)
    psi = _psi(ref, counts)
    return {
        "psi": round(psi, 6),
        "reference_missing_rate": round(float(reference["missing_rate"]), 6),
        "current_missing_rate": round(float(values.isna().mean()), 6),
        "missing_rate_delta": round(float(values.isna().mean() - reference["missing_rate"]), 6),
        "status": _severity(psi, DEFAULT_THRESHOLDS),
    }


def _categorical_drift(reference: dict[str, Any], current: pd.Series) -> dict[str, Any]:
    values = current.fillna("__MISSING__").astype(str)
    categories = reference["categories"]
    counts = values.where(values.isin(categories), "__OTHER__").value_counts()
    current_counts = np.array([counts.get(key, 0) for key in categories])
    psi = _psi(np.asarray(reference["proportions"], dtype=float), current_counts)
    return {
        "psi": round(psi, 6),
        "reference_missing_rate": round(float(reference["missing_rate"]), 6),
        "current_missing_rate": round(float(current.isna().mean()), 6),
        "missing_rate_delta": round(float(current.isna().mean() - reference["missing_rate"]), 6),
        "unseen_category_rate": round(float((~values.isin(categories)).mean()), 6),
        "status": _severity(psi, DEFAULT_THRESHOLDS),
    }


def compare_distributions(
    reference_profile: dict[str, Any],
    current_frame: pd.DataFrame,
    prediction_probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare an unlabeled current batch with the saved reference profile."""
    if current_frame.empty:
        raise ValueError("Current batch must contain at least one row")
    results: dict[str, Any] = {}
    for name, profile in reference_profile["features"].items():
        if name not in current_frame:
            raise ValueError(f"Current batch is missing feature: {name}")
        if profile["kind"] == "numeric":
            results[name] = _numeric_drift(profile, current_frame[name])
        else:
            results[name] = _categorical_drift(profile, current_frame[name])
    prediction_drift = None
    if prediction_probabilities is not None and "prediction_probability" in reference_profile:
        prediction_drift = _numeric_drift(
            reference_profile["prediction_probability"],
            pd.Series(np.asarray(prediction_probabilities, dtype=float)),
        )
    feature_scores = [item["psi"] for item in results.values() if item.get("psi") is not None]
    all_scores = feature_scores + (
        [prediction_drift["psi"]] if prediction_drift and prediction_drift.get("psi") is not None else []
    )
    maximum = max(all_scores, default=0.0)
    return {
        "method": reference_profile["method"],
        "reference_rows": reference_profile["reference_rows"],
        "current_rows": int(len(current_frame)),
        "features": results,
        "prediction_probability": prediction_drift,
        "summary": {
            "features_monitored": len(reference_profile["features"]),
            "features_high_drift": sum(item["status"] == "high" for item in results.values()),
            "features_moderate_drift": sum(item["status"] == "moderate" for item in results.values()),
            "maximum_psi": round(float(maximum), 6),
            "prediction_psi": (
                round(float(prediction_drift["psi"]), 6)
                if prediction_drift and prediction_drift.get("psi") is not None
                else None
            ),
            "overall_status": _severity(maximum, DEFAULT_THRESHOLDS),
        },
        "thresholds": {
            "low": DEFAULT_THRESHOLDS["low"],
            "moderate": DEFAULT_THRESHOLDS["moderate"],
            "interpretation": "Conventional PSI alert bands, not universal decision thresholds.",
        },
        "limitations": [
            "All reference data and outputs are synthetic.",
            "PSI detects distribution change, not changes in predictive accuracy or causality.",
            "Unlabeled batches do not support outcome-based performance monitoring.",
            "Small batches produce unstable distribution estimates.",
            "A drift alert requires investigation and is not an automatic reason to reject or retrain a model.",
        ],
    }

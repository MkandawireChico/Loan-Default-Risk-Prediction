"""
Probability calibration functions for loan default risk prediction.

This module provides functions for calibrating model probabilities and
generating calibration plots and metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import logging

logger = logging.getLogger(__name__)


def calibrate_model(pipeline: Any,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    method: str = "sigmoid",
                    cv: int = 5) -> Any:
    """
    Calibrate a trained model pipeline.
    
    Args:
        pipeline: Fitted model pipeline
        X_train: Training features
        y_train: Training target
        method: Calibration method ('sigmoid' for Platt scaling, 'isotonic' for isotonic regression)
        cv: Number of CV folds for calibration
        
    Returns:
        Calibrated model pipeline
    """
    # Extract the underlying model from the pipeline
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    
    # Transform training data
    X_train_transformed = preprocessor.transform(X_train)
    
    # Create calibrated classifier
    calibrated = CalibratedClassifierCV(
        model,
        method=method,
        cv=cv
    )
    
    logger.info(f"Calibrating model using {method} method with {cv}-fold CV...")
    calibrated.fit(X_train_transformed, y_train)
    
    # Create new pipeline with calibrated model
    from sklearn.pipeline import Pipeline
    calibrated_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", calibrated)
    ])
    
    logger.info("Model calibration complete")
    
    return calibrated_pipeline


def plot_calibration_curve(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          y_prob_calibrated: Optional[np.ndarray] = None,
                          n_bins: int = 10,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot calibration curve comparing predicted vs observed default rates.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities (uncalibrated)
        y_prob_calibrated: Predicted probabilities (calibrated, optional)
        n_bins: Number of bins for calibration curve
        save_path: Path to save the plot
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calibration curve for uncalibrated model
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label="Uncalibrated", color="#4C72B0")
    
    # Calibration curve for calibrated model (if provided)
    if y_prob_calibrated is not None:
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_calibrated, n_bins=n_bins, strategy="quantile")
        ax.plot(prob_pred_cal, prob_true_cal, marker='s', linewidth=2, label="Calibrated", color="#DD8452")
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
    
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        logger.info(f"Calibration plot saved to {save_path}")
    
    return fig


def plot_threshold_analysis(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           thresholds: Optional[List[float]] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot metrics across different classification thresholds.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        thresholds: List of thresholds to evaluate
        save_path: Path to save the plot
        
    Returns:
        Matplotlib Figure object
    """
    from src.evaluation.metrics import analyze_thresholds, calculate_threshold_metrics
    
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    
    results = analyze_thresholds(y_true, y_prob, thresholds)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Precision and Recall
    axes[0, 0].plot(results["threshold"], results["precision"], marker='o', label="Precision", color="#4C72B0")
    axes[0, 0].plot(results["threshold"], results["recall"], marker='s', label="Recall", color="#DD8452")
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Precision and Recall vs Threshold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Specificity and F1
    axes[0, 1].plot(results["threshold"], results["specificity"], marker='o', label="Specificity", color="#55A868")
    axes[0, 1].plot(results["threshold"], results["f1"], marker='s', label="F1 Score", color="#C44E52")
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Specificity and F1 vs Threshold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Borrowers Flagged
    axes[1, 0].plot(results["threshold"], results["borrowers_flagged_pct"], marker='o', color="#8172B2")
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("Borrowers Flagged (%)")
    axes[1, 0].set_title("Percentage of Borrowers Flagged vs Threshold")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confusion Matrix Components
    axes[1, 1].plot(results["threshold"], results["true_positives"], marker='o', label="True Positives", color="#55A868")
    axes[1, 1].plot(results["threshold"], results["false_positives"], marker='s', label="False Positives", color="#C44E52")
    axes[1, 1].plot(results["threshold"], results["false_negatives"], marker='^', label="False Negatives", color="#DD8452")
    axes[1, 1].set_xlabel("Threshold")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Confusion Matrix Components vs Threshold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        logger.info(f"Threshold analysis plot saved to {save_path}")
    
    return fig


def plot_roc_curve(y_true: np.ndarray,
                  y_prob: np.ndarray,
                  y_prob_calibrated: Optional[np.ndarray] = None,
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curve with AUC.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities (uncalibrated)
        y_prob_calibrated: Predicted probabilities (calibrated, optional)
        save_path: Path to save the plot
        
    Returns:
        Matplotlib Figure object
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ROC curve for uncalibrated model
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, linewidth=2, label=f"Uncalibrated (AUC = {auc:.4f})", color="#4C72B0")
    
    # ROC curve for calibrated model (if provided)
    if y_prob_calibrated is not None:
        fpr_cal, tpr_cal, _ = roc_curve(y_true, y_prob_calibrated)
        auc_cal = roc_auc_score(y_true, y_prob_calibrated)
        ax.plot(fpr_cal, tpr_cal, linewidth=2, label=f"Calibrated (AUC = {auc_cal:.4f})", color="#DD8452")
    
    # Random classifier line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Classifier")
    
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")
    
    return fig


def find_optimal_threshold(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          metric: str = "f1") -> Tuple[float, Dict[str, any]]:
    """
    Find the optimal threshold for a given metric.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'specificity', 'accuracy')
        
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    from src.evaluation.metrics import analyze_thresholds
    
    thresholds = np.arange(0.01, 0.99, 0.01)
    results = analyze_thresholds(y_true, y_prob, thresholds)
    
    if metric not in results.columns:
        raise ValueError(f"Metric {metric} not available. Available: {list(results.columns)}")
    
    # Find threshold that maximizes the metric
    optimal_idx = results[metric].idxmax()
    optimal_threshold = results.loc[optimal_idx, "threshold"]
    optimal_metrics = results.loc[optimal_idx].to_dict()
    
    logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.2f} ({metric} = {optimal_metrics[metric]:.4f})")
    
    return optimal_threshold, optimal_metrics


def generate_calibration_report(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               y_prob_calibrated: Optional[np.ndarray] = None) -> Dict[str, any]:
    """
    Generate a comprehensive calibration report.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities (uncalibrated)
        y_prob_calibrated: Predicted probabilities (calibrated, optional)
        
    Returns:
        Dictionary with calibration metrics and analysis
    """
    from src.evaluation.metrics import (
        calculate_calibration_metrics,
        calculate_ks_statistic,
        calculate_gini,
        roc_auc_score
    )
    
    report = {
        "uncalibrated": {},
        "calibrated": {}
    }
    
    # Uncalibrated metrics
    report["uncalibrated"]["calibration"] = calculate_calibration_metrics(y_true, y_prob)
    report["uncalibrated"]["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    report["uncalibrated"]["gini"] = float(calculate_gini(report["uncalibrated"]["roc_auc"]))
    ks_stat, ks_thresh = calculate_ks_statistic(y_true, y_prob)
    report["uncalibrated"]["ks_statistic"] = float(ks_stat)
    report["uncalibrated"]["ks_threshold"] = float(ks_thresh)
    
    # Calibrated metrics (if provided)
    if y_prob_calibrated is not None:
        report["calibrated"]["calibration"] = calculate_calibration_metrics(y_true, y_prob_calibrated)
        report["calibrated"]["roc_auc"] = float(roc_auc_score(y_true, y_prob_calibrated))
        report["calibrated"]["gini"] = float(calculate_gini(report["calibrated"]["roc_auc"]))
        ks_stat_cal, ks_thresh_cal = calculate_ks_statistic(y_true, y_prob_calibrated)
        report["calibrated"]["ks_statistic"] = float(ks_stat_cal)
        report["calibrated"]["ks_threshold"] = float(ks_thresh_cal)
    
    return report

"""
Credit-risk-specific evaluation metrics for loan default prediction.

This module provides metrics appropriate for credit risk modeling, including
discrimination metrics (ROC-AUC, Gini, KS), calibration metrics (Brier score,
calibration curves), and threshold analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    brier_score_loss, log_loss
)
import logging

logger = logging.getLogger(__name__)


def calculate_gini(roc_auc: float) -> float:
    """
    Calculate Gini coefficient from ROC-AUC.
    
    Gini = 2 * ROC-AUC - 1
    
    Args:
        roc_auc: ROC-AUC score
        
    Returns:
        Gini coefficient
    """
    return 2 * roc_auc - 1


def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov (KS) statistic.
    
    KS measures the maximum separation between the cumulative distributions
    of predicted probabilities for default and non-default observations.
    
    Args:
        y_true: True binary labels (0 = non-default, 1 = default)
        y_prob: Predicted probabilities of default
        
    Returns:
        Tuple of (KS statistic, threshold at which KS is maximized)
    """
    # Separate probabilities by class
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    
    # Sort probabilities
    pos_probs_sorted = np.sort(pos_probs)
    neg_probs_sorted = np.sort(neg_probs)
    
    # Calculate cumulative distributions
    n_pos = len(pos_probs_sorted)
    n_neg = len(neg_probs_sorted)
    
    pos_cdf = np.arange(1, n_pos + 1) / n_pos
    neg_cdf = np.arange(1, n_neg + 1) / n_neg
    
    # Interpolate to common grid
    all_probs = np.sort(np.concatenate([pos_probs_sorted, neg_probs_sorted]))
    pos_interp = np.interp(all_probs, pos_probs_sorted, pos_cdf, left=0, right=1)
    neg_interp = np.interp(all_probs, neg_probs_sorted, neg_cdf, left=0, right=1)
    
    # KS is maximum separation
    ks_values = np.abs(pos_interp - neg_interp)
    ks_statistic = np.max(ks_values)
    ks_threshold = all_probs[np.argmax(ks_values)]
    
    return ks_statistic, ks_threshold


def calculate_threshold_metrics(y_true: np.ndarray, 
                                y_prob: np.ndarray,
                                threshold: float) -> Dict[str, any]:
    """
    Calculate classification metrics at a specific threshold.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary with metrics at the threshold
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    n_flagged = (y_prob >= threshold).sum()
    flagged_pct = n_flagged / len(y_prob) * 100
    
    return {
        "threshold": threshold,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "borrowers_flagged": int(n_flagged),
        "borrowers_flagged_pct": float(flagged_pct)
    }


def analyze_thresholds(y_true: np.ndarray,
                      y_prob: np.ndarray,
                      thresholds: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Analyze multiple classification thresholds.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        thresholds: List of thresholds to evaluate (default: 0.1 to 0.9)
        
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = []
    for thresh in thresholds:
        metrics = calculate_threshold_metrics(y_true, y_prob, thresh)
        results.append(metrics)
    
    return pd.DataFrame(results)


def calculate_calibration_metrics(y_true: np.ndarray,
                                    y_prob: np.ndarray,
                                    n_bins: int = 10) -> Dict[str, any]:
    """
    Calculate calibration metrics and bin-wise statistics.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration analysis
        
    Returns:
        Dictionary with calibration metrics and bin statistics
    """
    brier = brier_score_loss(y_true, y_prob)
    logloss = log_loss(y_true, y_prob)
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True) - 1
    
    bin_stats = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_pred_mean = y_prob[mask].mean()
            bin_observed_rate = y_true[mask].mean()
            bin_count = mask.sum()
            
            bin_stats.append({
                "bin": i,
                "bin_lower": bin_edges[i],
                "bin_upper": bin_edges[i + 1],
                "count": int(bin_count),
                "predicted_prob": float(bin_pred_mean),
                "observed_rate": float(bin_observed_rate),
                "calibration_error": float(abs(bin_pred_mean - bin_observed_rate))
            })
    
    return {
        "brier_score": float(brier),
        "log_loss": float(logloss),
        "bin_statistics": bin_stats
    }


def calculate_risk_segment_metrics(y_true: np.ndarray,
                                    y_prob: np.ndarray,
                                    boundaries: List[float] = None) -> pd.DataFrame:
    """
    Calculate metrics for risk segments based on probability bands.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        boundaries: List of probability boundaries (default: [0.20, 0.50])
        
    Returns:
        DataFrame with risk segment statistics
    """
    if boundaries is None:
        boundaries = [0.20, 0.50]
    
    # Assign risk categories
    def categorize_risk(p):
        if p < boundaries[0]:
            return "Low Risk"
        elif p < boundaries[1]:
            return "Medium Risk"
        return "High Risk"
    
    risk_category = pd.Series(y_prob).apply(categorize_risk)
    
    results = []
    for category in ["Low Risk", "Medium Risk", "High Risk"]:
        mask = risk_category == category
        if mask.sum() > 0:
            segment_data = {
                "risk_category": category,
                "count": int(mask.sum()),
                "count_pct": float(mask.mean() * 100),
                "avg_predicted_prob": float(y_prob[mask].mean()),
                "observed_default_rate": float(y_true[mask].mean()),
                "avg_loan_amount": None  # Would need loan_amount column
            }
            results.append(segment_data)
    
    return pd.DataFrame(results)


def comprehensive_evaluation(y_true: np.ndarray,
                            y_prob: np.ndarray,
                            y_pred: Optional[np.ndarray] = None,
                            threshold: float = 0.5) -> Dict[str, any]:
    """
    Perform comprehensive model evaluation.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        y_pred: Predicted class labels (optional, calculated from threshold if None)
        threshold: Threshold for class prediction if y_pred is None
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    
    # Discrimination metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    gini = calculate_gini(roc_auc)
    ks_stat, ks_thresh = calculate_ks_statistic(y_true, y_prob)
    
    # Calibration metrics
    brier = brier_score_loss(y_true, y_prob)
    logloss = log_loss(y_true, y_prob)
    
    # Classification metrics at threshold
    thresh_metrics = calculate_threshold_metrics(y_true, y_prob, threshold)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Classification report
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        "discrimination": {
            "roc_auc": float(roc_auc),
            "gini": float(gini),
            "ks_statistic": float(ks_stat),
            "ks_threshold": float(ks_thresh)
        },
        "calibration": {
            "brier_score": float(brier),
            "log_loss": float(logloss)
        },
        "classification_at_threshold": thresh_metrics,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "classification_report": class_report
    }


def print_evaluation_report(eval_results: Dict[str, any]) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        eval_results: Results from comprehensive_evaluation
    """
    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    
    print("\n[Discrimination Metrics]")
    disc = eval_results["discrimination"]
    print(f"  ROC-AUC: {disc['roc_auc']:.4f}")
    print(f"  Gini: {disc['gini']:.4f}")
    print(f"  KS Statistic: {disc['ks_statistic']:.4f}")
    print(f"  KS Threshold: {disc['ks_threshold']:.4f}")
    
    print("\n[Calibration Metrics]")
    cal = eval_results["calibration"]
    print(f"  Brier Score: {cal['brier_score']:.4f}")
    print(f"  Log Loss: {cal['log_loss']:.4f}")
    
    print("\n[Classification at Threshold]")
    cls = eval_results["classification_at_threshold"]
    print(f"  Threshold: {cls['threshold']:.2f}")
    print(f"  Accuracy: {cls['accuracy']:.4f}")
    print(f"  Precision: {cls['precision']:.4f}")
    print(f"  Recall: {cls['recall']:.4f}")
    print(f"  Specificity: {cls['specificity']:.4f}")
    print(f"  F1 Score: {cls['f1']:.4f}")
    print(f"  Borrowers Flagged: {cls['borrowers_flagged']:,} ({cls['borrowers_flagged_pct']:.1f}%)")
    
    print("\n[Confusion Matrix]")
    cm = eval_results["confusion_matrix"]
    print(f"  True Negatives: {cm['true_negatives']:,}")
    print(f"  False Positives: {cm['false_positives']:,}")
    print(f"  False Negatives: {cm['false_negatives']:,}")
    print(f"  True Positives: {cm['true_positives']:,}")
    
    print("\n" + "=" * 60)

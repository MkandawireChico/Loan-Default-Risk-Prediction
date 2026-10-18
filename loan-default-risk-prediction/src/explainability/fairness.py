"""
Fairness analysis functions for loan default risk prediction.

This module provides functions for assessing model fairness across
different demographic and segment groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_group_metrics(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           group_mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate performance metrics for a specific group.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        group_mask: Boolean mask indicating group membership
        
    Returns:
        Dictionary with group metrics
    """
    if group_mask.sum() == 0:
        return {
            "count": 0,
            "default_rate": np.nan,
            "mean_predicted_prob": np.nan,
            "roc_auc": np.nan
        }
    
    y_true_group = y_true[group_mask]
    y_prob_group = y_prob[group_mask]
    
    from sklearn.metrics import roc_auc_score
    
    metrics = {
        "count": int(group_mask.sum()),
        "default_rate": float(y_true_group.mean()),
        "mean_predicted_prob": float(y_prob_group.mean()),
        "roc_auc": float(roc_auc_score(y_true_group, y_prob_group)) if len(np.unique(y_true_group)) > 1 else np.nan
    }
    
    return metrics


def fairness_by_attribute(df: pd.DataFrame,
                        y_true: np.ndarray,
                        y_prob: np.ndarray,
                        attribute: str,
                        min_group_size: int = 100) -> pd.DataFrame:
    """
    Analyze fairness across values of a categorical attribute.
    
    Args:
        df: DataFrame with the attribute column
        y_true: True binary labels
        y_prob: Predicted probabilities
        attribute: Column name to analyze
        min_group_size: Minimum group size for reporting
        
    Returns:
        DataFrame with fairness metrics by group
    """
    if attribute not in df.columns:
        logger.warning(f"Attribute {attribute} not found in DataFrame")
        return pd.DataFrame()
    
    results = []
    
    for value in df[attribute].dropna().unique():
        mask = df[attribute] == value
        
        if mask.sum() < min_group_size:
            logger.info(f"Skipping {attribute}={value} (size {mask.sum()} < {min_group_size})")
            continue
        
        metrics = calculate_group_metrics(y_true, y_prob, mask)
        metrics["attribute"] = attribute
        metrics["value"] = str(value)
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Calculate disparity metrics
    if len(results_df) > 1:
        max_default_rate = results_df["default_rate"].max()
        min_default_rate = results_df["default_rate"].min()
        results_df["default_rate_disparity"] = max_default_rate - min_default_rate
        
        max_mean_pred = results_df["mean_predicted_prob"].max()
        min_mean_pred = results_df["mean_predicted_prob"].min()
        results_df["prediction_disparity"] = max_mean_pred - min_mean_pred
    
    return results_df


def demographic_parity_difference(y_pred: np.ndarray,
                                  group_mask: np.ndarray,
                                  reference_mask: np.ndarray) -> float:
    """
    Calculate demographic parity difference.
    
    Measures the difference in positive prediction rates between groups.
    
    Args:
        y_pred: Predicted class labels
        group_mask: Boolean mask for group of interest
        reference_mask: Boolean mask for reference group
        
    Returns:
        Demographic parity difference
    """
    group_ppr = y_pred[group_mask].mean()
    ref_ppr = y_pred[reference_mask].mean()
    
    return float(abs(group_ppr - ref_ppr))


def equalized_odds_difference(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             group_mask: np.ndarray,
                             reference_mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate equalized odds difference.
    
    Measures the difference in TPR and FPR between groups.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted class labels
        group_mask: Boolean mask for group of interest
        reference_mask: Boolean mask for reference group
        
    Returns:
        Dictionary with TPR and FPR differences
    """
    from sklearn.metrics import confusion_matrix
    
    # Group metrics
    tn_g, fp_g, fn_g, tp_g = confusion_matrix(
        y_true[group_mask], y_pred[group_mask], labels=[0, 1]
    ).ravel()
    
    tpr_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
    fpr_g = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0
    
    # Reference metrics
    tn_r, fp_r, fn_r, tp_r = confusion_matrix(
        y_true[reference_mask], y_pred[reference_mask], labels=[0, 1]
    ).ravel()
    
    tpr_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0
    fpr_r = fp_r / (fp_r + tn_r) if (fp_r + tn_r) > 0 else 0
    
    return {
        "tpr_difference": float(abs(tpr_g - tpr_r)),
        "fpr_difference": float(abs(fpr_g - fpr_r))
    }


def generate_fairness_report(df: pd.DataFrame,
                            y_true: np.ndarray,
                            y_prob: np.ndarray,
                            y_pred: np.ndarray,
                            protected_attributes: List[str],
                            min_group_size: int = 100) -> Dict[str, any]:
    """
    Generate a comprehensive fairness report.
    
    Args:
        df: DataFrame with protected attribute columns
        y_true: True binary labels
        y_prob: Predicted probabilities
        y_pred: Predicted class labels
        protected_attributes: List of protected attribute column names
        min_group_size: Minimum group size for analysis
        
    Returns:
        Dictionary with fairness analysis results
    """
    report = {
        "by_attribute": {},
        "overall_disparities": {}
    }
    
    for attr in protected_attributes:
        if attr not in df.columns:
            logger.warning(f"Skipping {attr} (not found in DataFrame)")
            continue
        
        attr_report = fairness_by_attribute(df, y_true, y_prob, attr, min_group_size)
        report["by_attribute"][attr] = attr_report.to_dict("records") if not attr_report.empty else []
        
        # Calculate overall disparity for this attribute
        if not attr_report.empty and "default_rate_disparity" in attr_report.columns:
            report["overall_disparities"][attr] = {
                "default_rate_disparity": float(attr_report["default_rate_disparity"].iloc[0]),
                "prediction_disparity": float(attr_report["prediction_disparity"].iloc[0])
            }
    
    return report


def print_fairness_report(report: Dict[str, any]) -> None:
    """
    Print a formatted fairness report.
    
    Args:
        report: Fairness report from generate_fairness_report
    """
    print("=" * 60)
    print("FAIRNESS ANALYSIS REPORT")
    print("=" * 60)
    
    for attr, groups in report["by_attribute"].items():
        if not groups:
            continue
        
        print(f"\n[{attr}]")
        for group in groups:
            print(f"  {group['value']}:")
            print(f"    Count: {group['count']:,}")
            print(f"    Default Rate: {group['default_rate']:.2%}")
            print(f"    Mean Predicted Prob: {group['mean_predicted_prob']:.2%}")
            if not np.isnan(group['roc_auc']):
                print(f"    ROC-AUC: {group['roc_auc']:.4f}")
    
    print("\n[Overall Disparities]")
    for attr, disparities in report["overall_disparities"].items():
        print(f"  {attr}:")
        print(f"    Default Rate Disparity: {disparities['default_rate_disparity']:.2%}")
        print(f"    Prediction Disparity: {disparities['prediction_disparity']:.2%}")
    
    print("\n" + "=" * 60)

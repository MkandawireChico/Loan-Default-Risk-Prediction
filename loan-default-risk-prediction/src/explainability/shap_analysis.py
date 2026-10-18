"""
SHAP-based model explainability for loan default risk prediction.

This module provides functions for computing and visualizing SHAP values
to understand feature importance and individual predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def compute_shap_values(pipeline: Any,
                       X: pd.DataFrame,
                       background_samples: int = 100) -> Tuple[Any, np.ndarray]:
    """
    Compute SHAP values for a trained model pipeline.
    
    Args:
        pipeline: Fitted model pipeline
        X: Feature DataFrame
        background_samples: Number of background samples for SHAP
        
    Returns:
        Tuple of (shap_explainer, shap_values)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP package is required. Install with: pip install shap")
    
    # Extract preprocessor and model
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    
    # Transform features
    X_transformed = preprocessor.transform(X)
    
    # Get feature names after transformation
    feature_names = get_feature_names(preprocessor, X.columns)
    
    # Create background dataset for SHAP
    background_idx = np.random.choice(len(X_transformed), min(background_samples, len(X_transformed)), replace=False)
    background = X_transformed[background_idx]
    
    # Create explainer based on model type
    model_name = model.__class__.__name__
    
    if model_name == "LogisticRegression":
        explainer = shap.LinearExplainer(model, background)
    elif model_name == "RandomForestClassifier":
        explainer = shap.TreeExplainer(model)
    elif model_name in ["XGBClassifier", "CalibratedClassifierCV"]:
        # For XGBoost and calibrated models, use TreeExplainer
        if hasattr(model, "base_estimator"):
            # CalibratedClassifierCV
            base_model = model.base_estimator
            explainer = shap.TreeExplainer(base_model)
        else:
            explainer = shap.TreeExplainer(model)
    else:
        # Fallback to KernelExplainer (slower)
        logger.warning(f"Using KernelExplainer for {model_name} (slower)")
        explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # Compute SHAP values
    logger.info("Computing SHAP values...")
    shap_values = explainer.shap_values(X_transformed)
    
    # For binary classification, shap_values is a list - take the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    logger.info(f"SHAP values computed: shape {shap_values.shape}")
    
    return explainer, shap_values, feature_names


def get_feature_names(preprocessor: Any, original_columns: List[str]) -> List[str]:
    """
    Get feature names after preprocessing transformation.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        original_columns: Original column names before transformation
        
    Returns:
        List of feature names after transformation
    """
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            # Numeric features keep their names
            feature_names.extend(columns)
        elif name == "cat":
            # Categorical features are one-hot encoded
            if hasattr(transformer.named_steps["onehot"], "get_feature_names_out"):
                cat_names = transformer.named_steps["onehot"].get_feature_names_out(columns)
            else:
                # Fallback for older sklearn versions
                cat_names = [f"{col}_{i}" for col in columns for i in range(transformer.named_steps["onehot"].transformers_[0].categories_[0].size)]
            feature_names.extend(cat_names)
    
    return feature_names


def plot_feature_importance(shap_values: np.ndarray,
                           feature_names: List[str],
                           top_n: int = 20,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot SHAP feature importance (mean absolute SHAP values).
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Path to save the plot
        
    Returns:
        Matplotlib Figure object
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP package is required")
    
    # Compute mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame for sorting
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_shap
    }).sort_values("importance", ascending=True)
    
    # Take top N
    importance_df = importance_df.tail(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} Feature Importance (SHAP)")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return fig


def plot_shap_summary(shap_values: np.ndarray,
                      feature_names: List[str],
                      max_display: int = 20,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot SHAP summary plot (beeswarm plot).
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        max_display: Maximum number of features to display
        save_path: Path to save the plot
        
    Returns:
        Matplotlib Figure object
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP package is required")
    
    # Create DataFrame for SHAP
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    
    # Plot summary
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_df.values, shap_df.columns, max_display=max_display, show=False)
    
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        logger.info(f"SHAP summary plot saved to {save_path}")
    
    return fig


def plot_shap_waterfall(pipeline: Any,
                       preprocessor: Any,
                       shap_values: np.ndarray,
                       feature_names: List[str],
                       X: pd.DataFrame,
                       sample_idx: int = 0,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot SHAP waterfall plot for a single prediction.
    
    Args:
        pipeline: Fitted model pipeline
        preprocessor: Fitted preprocessor
        shap_values: SHAP values array
        feature_names: List of feature names
        X: Feature DataFrame
        sample_idx: Index of sample to explain
        save_path: Path to save the plot
        
    Returns:
        Matplotlib Figure object
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP package is required")
    
    # Get model
    model = pipeline.named_steps["model"]
    
    # Transform single sample
    X_transformed = preprocessor.transform(X.iloc[[sample_idx]])
    
    # Create Explanation object
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=model.predict_proba(X_transformed)[0, 0],
        data=X_transformed[0],
        feature_names=feature_names
    )
    
    # Plot waterfall
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        logger.info(f"SHAP waterfall plot saved to {save_path}")
    
    return fig


def generate_shap_report(shap_values: np.ndarray,
                        feature_names: List[str],
                        X: pd.DataFrame,
                        y: pd.Series) -> Dict[str, any]:
    """
    Generate a comprehensive SHAP analysis report.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        Dictionary with SHAP analysis results
    """
    # Compute mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_shap
    }).sort_values("mean_abs_shap", ascending=False)
    
    # Compute SHAP interaction with target
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df["target"] = y.values
    
    # Average SHAP by target class
    shap_by_target = shap_df.groupby("target").mean()
    
    return {
        "feature_importance": importance_df.to_dict("records"),
        "top_features": importance_df.head(10).to_dict("records"),
        "shap_by_target": {
            "non_default": shap_by_target.loc[0].to_dict(),
            "default": shap_by_target.loc[1].to_dict()
        }
    }

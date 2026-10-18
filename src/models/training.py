"""
Model training functions for loan default risk prediction.

This module provides reusable functions for training baseline and
advanced models with proper leakage prevention.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_baseline_model(strategy: str = "prevalence") -> Any:
    """
    Create a simple baseline model for comparison.
    
    Args:
        strategy: Baseline strategy ('prevalence' or 'random')
        
    Returns:
        Baseline model object
    """
    from sklearn.dummy import DummyClassifier
    
    if strategy == "prevalence":
        # Predicts the most frequent class (non-default in this case)
        model = DummyClassifier(strategy="most_frequent", random_state=42)
    elif strategy == "random":
        # Predicts randomly according to class distribution
        model = DummyClassifier(strategy="stratified", random_state=42)
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")
    
    logger.info(f"Created baseline model: {strategy}")
    return model


def create_logistic_regression(class_weight: str = "balanced",
                               random_state: int = 42) -> Any:
    """
    Create a logistic regression model.
    
    Args:
        class_weight: How to handle class imbalance ('balanced' or None)
        random_state: Random seed
        
    Returns:
        LogisticRegression model
    """
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(
        class_weight=class_weight,
        random_state=random_state,
        max_iter=1000,
        n_jobs=-1
    )
    
    logger.info("Created Logistic Regression model")
    return model


def create_random_forest(n_estimators: int = 200,
                        max_depth: int = 10,
                        min_samples_leaf: int = 5,
                        class_weight: str = "balanced",
                        random_state: int = 42) -> any:
    """
    Create a random forest model.
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples per leaf
        class_weight: How to handle class imbalance
        random_state: Random seed
        
    Returns:
        RandomForestClassifier model
    """
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    
    logger.info(f"Created Random Forest model: {n_estimators} trees, max_depth={max_depth}")
    return model


def create_xgboost(n_estimators: int = 200,
                  max_depth: int = 6,
                  learning_rate: float = 0.1,
                  scale_pos_weight: Optional[float] = None,
                  random_state: int = 42) -> Any:
    """
    Create an XGBoost model.
    
    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate (eta)
        scale_pos_weight: Scaling for positive class (for imbalance)
        random_state: Random seed
        
    Returns:
        XGBClassifier model
    """
    import xgboost as xgb
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss"
    )
    
    logger.info(f"Created XGBoost model: {n_estimators} rounds, max_depth={max_depth}")
    return model


def create_model_pipeline(preprocessor: any, model: any) -> any:
    """
    Combine preprocessing and model into a single pipeline.
    
    Args:
        preprocessor: ColumnTransformer for preprocessing
        model: Model object
        
    Returns:
        Scikit-learn Pipeline
    """
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    logger.info("Created model pipeline")
    return pipeline


def train_model(pipeline: Any,
                X_train: pd.DataFrame,
                y_train: pd.Series) -> Any:
    """
    Train a model pipeline.
    
    Args:
        pipeline: Model pipeline
        X_train: Training features
        y_train: Training target
        
    Returns:
        Fitted pipeline
    """
    logger.info(f"Training model on {len(X_train):,} samples...")
    pipeline.fit(X_train, y_train)
    logger.info("Model training complete")
    
    return pipeline


def cross_validate_model(pipeline: any,
                        X: pd.DataFrame,
                        y: pd.Series,
                        cv: int = 5,
                        scoring: List[str] = None) -> Dict[str, any]:
    """
    Perform cross-validation on a model pipeline.
    
    Args:
        pipeline: Model pipeline (unfitted)
        X: Features
        y: Target
        cv: Number of CV folds
        scoring: List of scoring metrics
        
    Returns:
        Dictionary with CV results
    """
    from sklearn.model_selection import cross_validate
    
    if scoring is None:
        scoring = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    
    logger.info(f"Performing {cv}-fold cross-validation...")
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    # Aggregate results
    results = {}
    for metric in scoring:
        test_metric = f"test_{metric}"
        if test_metric in cv_results:
            results[metric] = {
                "mean": float(np.mean(cv_results[test_metric])),
                "std": float(np.std(cv_results[test_metric])),
                "values": cv_results[test_metric].tolist()
            }
    
    logger.info(f"Cross-validation complete. ROC-AUC: {results['roc_auc']['mean']:.4f} (+/- {results['roc_auc']['std']:.4f})")
    
    return results


def benchmark_models(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   preprocessor: any,
                   models_config: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
    """
    Benchmark multiple models using cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Preprocessing ColumnTransformer
        models_config: Dictionary of model configurations
        
    Returns:
        DataFrame with benchmark results
    """
    if models_config is None:
        models_config = {
            "Baseline (Prevalence)": {
                "model": create_baseline_model(strategy="prevalence")
            },
            "Logistic Regression": {
                "model": create_logistic_regression()
            },
            "Random Forest": {
                "model": create_random_forest()
            },
            "XGBoost": {
                "model": create_xgboost()
            }
        }
    
    results = []
    
    for name, config in models_config.items():
        logger.info(f"Benchmarking: {name}")
        
        pipeline = create_model_pipeline(preprocessor, config["model"])
        cv_results = cross_validate_model(pipeline, X_train, y_train, cv=5)
        
        result_row = {
            "model": name,
            "roc_auc_mean": cv_results["roc_auc"]["mean"],
            "roc_auc_std": cv_results["roc_auc"]["std"],
            "accuracy_mean": cv_results["accuracy"]["mean"],
            "precision_mean": cv_results["precision"]["mean"],
            "recall_mean": cv_results["recall"]["mean"],
            "f1_mean": cv_results["f1"]["mean"]
        }
        results.append(result_row)
    
    results_df = pd.DataFrame(results).sort_values("roc_auc_mean", ascending=False)
    
    logger.info("\nBenchmark Results:")
    logger.info(results_df.to_string(index=False))
    
    return results_df


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost to handle class imbalance.
    
    scale_pos_weight = sum(negative instances) / sum(positive instances)
    
    Args:
        y_train: Training target
        
    Returns:
        Scale weight for positive class
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    
    weight = n_neg / n_pos
    logger.info(f"Calculated scale_pos_weight: {weight:.2f} (neg={n_neg}, pos={n_pos})")
    
    return weight

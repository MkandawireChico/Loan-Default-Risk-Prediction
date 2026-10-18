"""
Feature engineering functions for loan default risk prediction.

This module provides reusable functions for creating credit-risk-relevant
features from raw loan application data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_repayment_ratio(df: pd.DataFrame, 
                           repayment_col: str = "Monthly_Repayment",
                           income_col: str = "Monthly_Income",
                           output_col: str = "Monthly_Repayment_Ratio") -> pd.DataFrame:
    """
    Calculate monthly repayment as a ratio of income.
    
    This is a direct affordability measure - the share of income consumed
    by this loan alone.
    
    Args:
        df: Input DataFrame
        repayment_col: Column name for monthly repayment amount
        income_col: Column name for monthly income
        output_col: Name for the output column
        
    Returns:
        DataFrame with the new feature added
    """
    df = df.copy()
    df[output_col] = df[repayment_col] / df[income_col]
    
    # Handle division by zero
    df[output_col] = df[output_col].replace([np.inf, -np.inf], np.nan)
    
    return df


def create_savings_to_income_ratio(df: pd.DataFrame,
                                   savings_col: str = "Savings_Balance",
                                   income_col: str = "Monthly_Income",
                                   output_col: str = "Savings_to_Income_Ratio") -> pd.DataFrame:
    """
    Calculate savings as a ratio of annual income.
    
    This represents financial cushion relative to annual earnings - a buffer
    against income shocks.
    
    Args:
        df: Input DataFrame
        savings_col: Column name for savings balance
        income_col: Column name for monthly income
        output_col: Name for the output column
        
    Returns:
        DataFrame with the new feature added
    """
    df = df.copy()
    df[output_col] = df[savings_col] / (df[income_col] * 12)
    
    # Handle division by zero
    df[output_col] = df[output_col].replace([np.inf, -np.inf], np.nan)
    
    return df


def create_previous_default_flag(df: pd.DataFrame,
                                 prev_default_col: str = "Previous_Default_Count",
                                 output_col: str = "Previous_Default_Flag") -> pd.DataFrame:
    """
    Create a binary flag for any prior default history.
    
    A simple binary version of default history, useful where the exact count
    matters less than "has this happened before at all".
    
    Args:
        df: Input DataFrame
        prev_default_col: Column name for previous default count
        output_col: Name for the output column
        
    Returns:
        DataFrame with the new feature added
    """
    df = df.copy()
    df[output_col] = (df[prev_default_col] > 0).astype(int)
    
    return df


def create_late_payment_frequency(df: pd.DataFrame,
                                  late_payments_col: str = "Number_of_Late_Payments",
                                  credit_history_col: str = "Credit_History_Length_Months",
                                  output_col: str = "Late_Payment_Frequency") -> pd.DataFrame:
    """
    Calculate late payments normalized by credit history length.
    
    Normalizes late payments by how long the credit history actually is.
    3 late payments over 6 months is very different from 3 over 10 years.
    
    Args:
        df: Input DataFrame
        late_payments_col: Column name for number of late payments
        credit_history_col: Column name for credit history length in months
        output_col: Name for the output column
        
    Returns:
        DataFrame with the new feature added
    """
    df = df.copy()
    
    # Replace zero history with NaN to avoid division by zero
    history_safe = df[credit_history_col].replace(0, np.nan)
    df[output_col] = df[late_payments_col] / history_safe
    
    # Handle division by zero
    df[output_col] = df[output_col].replace([np.inf, -np.inf], np.nan)
    
    return df


def create_loan_burden_category(df: pd.DataFrame,
                                ratio_col: str = "Monthly_Repayment_Ratio",
                                output_col: str = "Loan_Burden_Category",
                                thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Categorize repayment burden into readable risk bands.
    
    A label a loan officer can act on directly, not just a raw ratio.
    
    Args:
        df: Input DataFrame
        ratio_col: Column name for repayment ratio
        output_col: Name for the output column
        thresholds: Dictionary mapping category names to upper bounds.
                    Default: Low (<0.20), Moderate (<0.35), High (<0.50), Severe (>=0.50)
        
    Returns:
        DataFrame with the new feature added
    """
    if thresholds is None:
        thresholds = {
            "Low": 0.20,
            "Moderate": 0.35,
            "High": 0.50,
            "Severe": float('inf')
        }
    
    df = df.copy()
    
    def categorize(ratio):
        if pd.isna(ratio):
            return "Unknown"
        if ratio < thresholds["Low"]:
            return "Low"
        elif ratio < thresholds["Moderate"]:
            return "Moderate"
        elif ratio < thresholds["High"]:
            return "High"
        return "Severe"
    
    df[output_col] = df[ratio_col].apply(categorize)
    
    return df


def create_credit_risk_category(df: pd.DataFrame,
                                credit_score_col: str = "Credit_Score",
                                output_col: str = "Credit_Risk_Category",
                                thresholds: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Categorize credit score into standard bureau bands.
    
    Mirrors how credit bureaus typically communicate score bands.
    
    Args:
        df: Input DataFrame
        credit_score_col: Column name for credit score
        output_col: Name for the output column
        thresholds: Dictionary mapping category names to score boundaries.
                    Default: Poor (<580), Fair (<670), Good (<740), Very Good (>=740)
        
    Returns:
        DataFrame with the new feature added
    """
    if thresholds is None:
        thresholds = {
            "Poor": 580,
            "Fair": 670,
            "Good": 740,
            "Very Good": float('inf')
        }
    
    df = df.copy()
    
    def categorize(score):
        if pd.isna(score):
            return "Unknown"
        if score < thresholds["Poor"]:
            return "Poor"
        elif score < thresholds["Fair"]:
            return "Fair"
        elif score < thresholds["Good"]:
            return "Good"
        return "Very Good"
    
    df[output_col] = df[credit_score_col].apply(categorize)
    
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    
    Args:
        df: Input DataFrame with raw loan application data
        
    Returns:
        DataFrame with all engineered features added
    """
    df = df.copy()
    
    df = create_repayment_ratio(df)
    df = create_savings_to_income_ratio(df)
    df = create_previous_default_flag(df)
    df = create_late_payment_frequency(df)
    df = create_loan_burden_category(df)
    df = create_credit_risk_category(df)
    
    logger.info(f"Engineered features. New shape: {df.shape}")
    
    return df


def get_feature_definitions() -> Dict[str, Dict[str, str]]:
    """
    Return a dictionary documenting all engineered features.
    
    Returns:
        Dictionary mapping feature names to their definitions and business meanings
    """
    return {
        "Monthly_Repayment_Ratio": {
            "definition": "Monthly_Repayment / Monthly_Income",
            "business_meaning": "Share of income consumed by this loan alone - direct affordability measure"
        },
        "Savings_to_Income_Ratio": {
            "definition": "Savings_Balance / (Monthly_Income * 12)",
            "business_meaning": "Financial cushion relative to annual earnings - buffer against income shocks"
        },
        "Previous_Default_Flag": {
            "definition": "1 if Previous_Default_Count > 0, else 0",
            "business_meaning": "Binary indicator of any prior default history"
        },
        "Late_Payment_Frequency": {
            "definition": "Number_of_Late_Payments / Credit_History_Length_Months",
            "business_meaning": "Normalizes late payments by credit history length"
        },
        "Loan_Burden_Category": {
            "definition": "Banded Monthly_Repayment_Ratio (Low/Moderate/High/Severe)",
            "business_meaning": "Readable label for loan officer action"
        },
        "Credit_Risk_Category": {
            "definition": "Banded Credit_Score (Poor/Fair/Good/Very Good)",
            "business_meaning": "Mirrors standard credit bureau score bands"
        }
    }

"""
Data preprocessing pipeline for loan default risk prediction.

This module provides reusable functions for cleaning raw data, handling
missing values, standardizing categories, and preparing data for modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data cleaning transformations to raw loan application data.
    
    This function handles the 10 categories of deliberately injected
    data-quality issues in the synthetic dataset:
    1. Missing income, credit score, employment status - left for pipeline imputation
    2. Extreme income/loan outliers - winsorized at 99.5th percentile
    3. Inconsistent category labels - standardized via mapping
    4. Duplicate records - dropped
    5. Invalid ages - treated as missing
    6. Missing branch names - filled with "Unknown"
    
    Args:
        df: Raw DataFrame from data/raw/loan_applications_synthetic.csv
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # 1. Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped:,} duplicate rows")
    
    # 2. Standardize employment status categories
    employment_fix_map = {
        "employed - government": "Employed - Government",
        "Employed-Private": "Employed - Private",
        "SELF EMPLOYED": "Self-Employed",
        "Buisness Owner": "Business Owner",
        "unemployed ": "Unemployed",
    }
    df["Employment_Status"] = df["Employment_Status"].replace(employment_fix_map)
    
    # 3. Standardize loan purpose categories
    purpose_fix_map = {
        "personal ": "Personal",
        "biz/sme": "Business/SME",
        "home improvement": "Home Improvement",
        "AGRICULTURE": "Agriculture",
    }
    df["Loan_Purpose"] = df["Loan_Purpose"].replace(purpose_fix_map)
    
    # 4. Winsorize extreme income at 99.5th percentile
    income_cap = df["Monthly_Income"].quantile(0.995)
    n_income_capped = (df["Monthly_Income"] > income_cap).sum()
    df["Monthly_Income"] = df["Monthly_Income"].clip(upper=income_cap)
    if n_income_capped > 0:
        logger.info(f"Capped {n_income_capped:,} extreme income values at {income_cap:,.0f}")
    
    # 5. Winsorize extreme loan amounts at 99.5th percentile
    loan_cap = df["Loan_Amount"].quantile(0.995)
    n_loan_capped = (df["Loan_Amount"] > loan_cap).sum()
    df["Loan_Amount"] = df["Loan_Amount"].clip(upper=loan_cap)
    if n_loan_capped > 0:
        logger.info(f"Capped {n_loan_capped:,} extreme loan amounts at {loan_cap:,.0f}")
    
    # 6. Invalid ages -> treat as missing (outside 18-90 range)
    invalid_age_mask = (df["Age"] < 18) | (df["Age"] > 90)
    n_invalid_ages = invalid_age_mask.sum()
    df.loc[invalid_age_mask, "Age"] = np.nan
    if n_invalid_ages > 0:
        logger.info(f"Treated {n_invalid_ages:,} invalid ages as missing")
    
    # 7. Missing branch -> explicit "Unknown" category
    df["Branch_Location"] = df["Branch_Location"].fillna("Unknown")
    
    logger.info(f"Data cleaning complete. Shape: {df.shape}")
    
    return df


def get_feature_columns() -> Dict[str, List[str]]:
    """
    Return the column groupings for modeling.
    
    Returns:
        Dictionary with column groups:
        - id_cols: Identifier columns to exclude from modeling
        - numeric_cols: Numeric features
        - categorical_cols: Categorical features
        - excluded_cols: Columns to exclude (redundant or post-origination)
    """
    return {
        "id_cols": ["Customer_ID", "Loan_ID"],
        "numeric_cols": [
            "Age",
            "Monthly_Income",
            "Account_Tenure_Months",
            "Loan_Amount",
            "Loan_Term_Months",
            "Interest_Rate",
            "Existing_Loan_Count",
            "Existing_Debt",
            "Savings_Balance",
            "Average_Monthly_Transactions",
            "Average_Monthly_Balance",
            "Credit_History_Length_Months",
            "Previous_Loan_Count",
            "Previous_Default_Count",
            "Number_of_Late_Payments",
            "Days_Past_Due_History",
            "Debt_to_Income_Ratio",
            "Loan_to_Income_Ratio",
            "Credit_Score",
            "Monthly_Repayment_Ratio",
            "Savings_to_Income_Ratio",
            "Late_Payment_Frequency",
            "Previous_Default_Flag"
        ],
        "categorical_cols": [
            "Gender",
            "Marital_Status",
            "Education_Level",
            "Employment_Status",
            "Occupation",
            "Branch_Location",
            "Customer_Segment",
            "Loan_Purpose",
            "Has_Credit_Card",
            "Has_Mobile_Banking",
            "Collateral_Type",
            "Loan_Application_Channel",
            "Loan_Burden_Category",
            "Credit_Risk_Category"
        ],
        "excluded_cols": [
            "Monthly_Repayment",  # Redundant with Monthly_Repayment_Ratio
            "Collateral_Value",   # Often a downstream decision, not input
        ]
    }


def prepare_modeling_data(df: pd.DataFrame,
                         target_col: str = "Default") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling by separating features and target.
    
    Args:
        df: Cleaned DataFrame
        target_col: Name of target column
        
    Returns:
        Tuple of (X_features, y_target)
    """
    feature_cols = get_feature_columns()
    
    # Combine numeric and categorical columns
    model_cols = feature_cols["numeric_cols"] + feature_cols["categorical_cols"]
    
    # Filter to columns that exist in the dataframe
    available_cols = [col for col in model_cols if col in df.columns]
    
    X = df[available_cols].copy()
    y = df[target_col].copy()
    
    logger.info(f"Prepared modeling data: X shape {X.shape}, y shape {y.shape}")
    
    return X, y


def create_train_test_split(X: pd.DataFrame,
                           y: pd.Series,
                           test_size: float = 0.25,
                           random_state: int = 42,
                           stratify: Optional[pd.Series] = None) -> Tuple:
    """
    Create stratified train/test split.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        stratify: Column to stratify on (default: y)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    if stratify is None:
        stratify = y
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    logger.info(f"Train/test split: train {len(X_train):,}, test {len(X_test):,}")
    logger.info(f"Train default rate: {y_train.mean():.2%}, test default rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def create_preprocessing_pipeline(numeric_cols: List[str],
                                 categorical_cols: List[str]) -> any:
    """
    Create a scikit-learn ColumnTransformer for preprocessing.
    
    Pipeline includes:
    - Numeric: median imputation + standard scaling
    - Categorical: most-frequent imputation + one-hot encoding
    
    Args:
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        Fitted ColumnTransformer (must be fit on training data)
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    
    logger.info(f"Created preprocessing pipeline: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    
    return preprocessor


def get_modeling_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get the actual feature columns available in the dataframe for modeling.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with available numeric and categorical columns
    """
    feature_cols = get_feature_columns()
    
    available_numeric = [col for col in feature_cols["numeric_cols"] if col in df.columns]
    available_categorical = [col for col in feature_cols["categorical_cols"] if col in df.columns]
    
    return {
        "numeric": available_numeric,
        "categorical": available_categorical
    }

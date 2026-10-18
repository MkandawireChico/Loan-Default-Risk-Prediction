"""
Data validation functions for loan default risk prediction.

This module provides reusable functions to validate the loan applications dataset
for data quality issues, schema compliance, and potential leakage risks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def validate_schema(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, any]:
    """
    Validate that the DataFrame contains all required columns.
    
    Args:
        df: Input DataFrame
        required_columns: List of column names that must be present
        
    Returns:
        Dictionary with validation results
    """
    missing_cols = set(required_columns) - set(df.columns)
    result = {
        "valid": len(missing_cols) == 0,
        "missing_columns": list(missing_cols),
        "total_columns": len(df.columns),
        "required_count": len(required_columns)
    }
    return result


def validate_data_types(df: pd.DataFrame, type_schema: Dict[str, str]) -> Dict[str, any]:
    """
    Validate data types against expected schema.
    
    Args:
        df: Input DataFrame
        type_schema: Dictionary mapping column names to expected types
                     ('numeric', 'integer', 'categorical', 'boolean')
        
    Returns:
        Dictionary with validation results and issues
    """
    issues = []
    
    for col, expected_type in type_schema.items():
        if col not in df.columns:
            continue
            
        if expected_type == 'numeric':
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"{col}: expected numeric, got {df[col].dtype}")
        elif expected_type == 'integer':
            if not pd.api.types.is_integer_dtype(df[col]):
                issues.append(f"{col}: expected integer, got {df[col].dtype}")
        elif expected_type == 'categorical':
            if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])):
                issues.append(f"{col}: expected categorical, got {df[col].dtype}")
        elif expected_type == 'boolean':
            if not pd.api.types.is_bool_dtype(df[col]):
                issues.append(f"{col}: expected boolean, got {df[col].dtype}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


def validate_missing_values(df: pd.DataFrame, 
                           max_missing_pct: float = 0.05,
                           exclude_cols: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Check for missing values and report columns exceeding threshold.
    
    Args:
        df: Input DataFrame
        max_missing_pct: Maximum allowed missing percentage (0-1)
        exclude_cols: Columns to exclude from validation
        
    Returns:
        Dictionary with missing value statistics
    """
    if exclude_cols is None:
        exclude_cols = []
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    high_missing = missing_pct[missing_pct > (max_missing_pct * 100)]
    
    return {
        "total_rows": len(df),
        "total_missing": missing.sum(),
        "columns_with_missing": (missing > 0).sum(),
        "high_missing_columns": high_missing.to_dict(),
        "missing_by_column": missing[missing > 0].to_dict(),
        "missing_pct_by_column": missing_pct[missing_pct > 0].to_dict()
    }


def validate_duplicates(df: pd.DataFrame, 
                       id_columns: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Check for duplicate records.
    
    Args:
        df: Input DataFrame
        id_columns: Columns to use for duplicate detection (None = all columns)
        
    Returns:
        Dictionary with duplicate statistics
    """
    if id_columns is None:
        exact_duplicates = df.duplicated().sum()
    else:
        exact_duplicates = df.duplicated(subset=id_columns).sum()
    
    return {
        "total_rows": len(df),
        "duplicate_rows": int(exact_duplicates),
        "duplicate_pct": (exact_duplicates / len(df)) * 100,
        "unique_rows": len(df) - int(exact_duplicates)
    }


def validate_target_variable(df: pd.DataFrame, 
                            target_col: str = "Default",
                            expected_values: Optional[List[int]] = None) -> Dict[str, any]:
    """
    Validate the target variable for correct values and distribution.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        expected_values: Expected unique values (default: [0, 1])
        
    Returns:
        Dictionary with target validation results
    """
    if expected_values is None:
        expected_values = [0, 1]
    
    if target_col not in df.columns:
        return {
            "valid": False,
            "error": f"Target column '{target_col}' not found"
        }
    
    unique_values = df[target_col].unique()
    unexpected = set(unique_values) - set(expected_values)
    
    value_counts = df[target_col].value_counts()
    default_rate = df[target_col].mean()
    
    return {
        "valid": len(unexpected) == 0,
        "unique_values": list(unique_values),
        "unexpected_values": list(unexpected),
        "value_counts": value_counts.to_dict(),
        "default_rate": float(default_rate),
        "total_defaults": int(value_counts.get(1, 0)),
        "total_non_defaults": int(value_counts.get(0, 0))
    }


def validate_numeric_ranges(df: pd.DataFrame, 
                           range_schema: Dict[str, Tuple[float, float]]) -> Dict[str, any]:
    """
    Validate that numeric columns fall within expected ranges.
    
    Args:
        df: Input DataFrame
        range_schema: Dictionary mapping column names to (min, max) tuples
        
    Returns:
        Dictionary with range validation results
    """
    issues = []
    
    for col, (min_val, max_val) in range_schema.items():
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        
        below_min = (col_data < min_val).sum()
        above_max = (col_data > max_val).sum()
        
        if below_min > 0:
            issues.append(f"{col}: {below_min} values below {min_val}")
        if above_max > 0:
            issues.append(f"{col}: {above_max} values above {max_val}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


def validate_negative_values(df: pd.DataFrame, 
                             non_negative_cols: List[str]) -> Dict[str, any]:
    """
    Check for negative values in columns that should be non-negative.
    
    Args:
        df: Input DataFrame
        non_negative_cols: Columns that should not contain negative values
        
    Returns:
        Dictionary with negative value statistics
    """
    negative_counts = {}
    
    for col in non_negative_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                negative_counts[col] = int(neg_count)
    
    return {
        "valid": len(negative_counts) == 0,
        "columns_with_negatives": negative_counts
    }


def validate_infinite_values(df: pd.DataFrame) -> Dict[str, any]:
    """
    Check for infinite values in numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with infinite value statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_counts = {}
    
    for col in numeric_cols:
        pos_inf = (df[col] == np.inf).sum()
        neg_inf = (df[col] == -np.inf).sum()
        
        if pos_inf > 0 or neg_inf > 0:
            infinite_counts[col] = {
                "positive_infinity": int(pos_inf),
                "negative_infinity": int(neg_inf)
            }
    
    return {
        "valid": len(infinite_counts) == 0,
        "columns_with_infinite": infinite_counts
    }


def validate_categorical_consistency(df: pd.DataFrame,
                                    col: str,
                                    expected_categories: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Validate categorical columns for unexpected categories.
    
    Args:
        df: Input DataFrame
        col: Column name to validate
        expected_categories: Expected unique values (None = no validation)
        
    Returns:
        Dictionary with categorical validation results
    """
    if col not in df.columns:
        return {
            "valid": False,
            "error": f"Column '{col}' not found"
        }
    
    unique_values = df[col].dropna().unique()
    value_counts = df[col].value_counts(dropna=False).to_dict()
    
    if expected_categories is not None:
        unexpected = set(unique_values) - set(expected_categories)
        return {
            "valid": len(unexpected) == 0,
            "unique_values": list(unique_values),
            "unexpected_categories": list(unexpected),
            "value_counts": value_counts
        }
    
    return {
        "valid": True,
        "unique_values": list(unique_values),
        "value_counts": value_counts
    }


def check_leakage_risks(df: pd.DataFrame, 
                       target_col: str = "Default",
                       leakage_keywords: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Check for potential target leakage in feature names.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        leakage_keywords: Keywords that suggest leakage (default list)
        
    Returns:
        Dictionary with potential leakage risks
    """
    if leakage_keywords is None:
        leakage_keywords = [
            'default', 'defaulted', 'outcome', 'result', 'status',
            'paid', 'payment', 'repayment', 'recovery', 'loss',
            'write_off', 'charge_off', 'delinquent', 'collection'
        ]
    
    feature_cols = [col for col in df.columns if col != target_col]
    suspicious_cols = []
    
    for col in feature_cols:
        col_lower = col.lower()
        for keyword in leakage_keywords:
            if keyword in col_lower:
                suspicious_cols.append({
                    "column": col,
                    "keyword": keyword,
                    "reason": f"Column name contains '{keyword}' which may indicate target leakage"
                })
                break
    
    return {
        "suspicious_columns": suspicious_cols,
        "total_suspicious": len(suspicious_cols)
    }


def generate_data_quality_report(df: pd.DataFrame,
                                 target_col: str = "Default") -> Dict[str, any]:
    """
    Generate a comprehensive data quality report.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        Dictionary with comprehensive validation results
    """
    report = {
        "dataset_info": {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict()
        },
        "missing_values": validate_missing_values(df),
        "duplicates": validate_duplicates(df),
        "target_validation": validate_target_variable(df, target_col),
        "infinite_values": validate_infinite_values(df),
        "leakage_check": check_leakage_risks(df, target_col)
    }
    
    return report


def print_validation_report(report: Dict[str, any]) -> None:
    """
    Print a formatted validation report.
    
    Args:
        report: Validation report dictionary from generate_data_quality_report
    """
    print("=" * 60)
    print("DATA QUALITY VALIDATION REPORT")
    print("=" * 60)
    
    print("\n[Dataset Info]")
    info = report["dataset_info"]
    print(f"  Shape: {info['shape']}")
    print(f"  Columns: {info['shape'][1]}")
    
    print("\n[Missing Values]")
    missing = report["missing_values"]
    print(f"  Total missing: {missing['total_missing']}")
    print(f"  Columns with missing: {missing['columns_with_missing']}")
    if missing["high_missing_columns"]:
        print("  High missing columns:")
        for col, pct in missing["high_missing_columns"].items():
            print(f"    {col}: {pct:.1f}%")
    
    print("\n[Duplicates]")
    dup = report["duplicates"]
    print(f"  Duplicate rows: {dup['duplicate_rows']} ({dup['duplicate_pct']:.2f}%)")
    
    print("\n[Target Variable]")
    target = report["target_validation"]
    if target["valid"]:
        print(f"  Valid: Yes")
        print(f"  Default rate: {target['default_rate']:.2%}")
        print(f"  Defaults: {target['total_defaults']:,}")
        print(f"  Non-defaults: {target['total_non_defaults']:,}")
    else:
        print(f"  Valid: No - {target.get('error', 'Unknown error')}")
    
    print("\n[Infinite Values]")
    inf = report["infinite_values"]
    if inf["valid"]:
        print("  No infinite values found")
    else:
        print(f"  Columns with infinite values: {len(inf['columns_with_infinite'])}")
        for col, counts in inf["columns_with_infinite"].items():
            print(f"    {col}: +inf={counts['positive_infinity']}, -inf={counts['negative_infinity']}")
    
    print("\n[Leakage Check]")
    leakage = report["leakage_check"]
    if leakage["total_suspicious"] == 0:
        print("  No suspicious column names detected")
    else:
        print(f"  Suspicious columns: {leakage['total_suspicious']}")
        for item in leakage["suspicious_columns"]:
            print(f"    {item['column']}: {item['reason']}")
    
    print("\n" + "=" * 60)

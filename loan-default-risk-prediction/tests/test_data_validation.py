"""
Tests for data validation functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.validation import (
    validate_schema,
    validate_missing_values,
    validate_duplicates,
    validate_target_variable,
    validate_numeric_ranges,
    validate_negative_values,
    validate_infinite_values,
    generate_data_quality_report,
    print_validation_report
)


def test_validate_schema():
    """Test schema validation."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    
    # Valid schema
    result = validate_schema(df, ["col1", "col2"])
    assert result["valid"] is True
    assert result["missing_columns"] == []
    
    # Missing column
    result = validate_schema(df, ["col1", "col3"])
    assert result["valid"] is False
    assert "col3" in result["missing_columns"]


def test_validate_missing_values():
    """Test missing value validation."""
    df = pd.DataFrame({
        "col1": [1, 2, np.nan, 4],
        "col2": ["a", None, "c", "d"],
        "col3": [1, 2, 3, 4]
    })
    
    result = validate_missing_values(df, max_missing_pct=0.5)
    assert result["total_rows"] == 4
    assert result["columns_with_missing"] == 2
    assert result["total_missing"] == 2


def test_validate_duplicates():
    """Test duplicate detection."""
    df = pd.DataFrame({
        "col1": [1, 2, 2, 3],
        "col2": ["a", "b", "b", "c"]
    })
    
    result = validate_duplicates(df)
    assert result["total_rows"] == 4
    assert result["duplicate_rows"] == 1  # pandas marks second occurrence as duplicate
    assert result["unique_rows"] == 3


def test_validate_target_variable():
    """Test target variable validation."""
    df = pd.DataFrame({"Default": [0, 1, 0, 1, 1]})
    
    result = validate_target_variable(df, "Default")
    assert result["valid"] is True
    assert result["default_rate"] == 0.6
    assert result["total_defaults"] == 3
    assert result["total_non_defaults"] == 2
    
    # Invalid target values
    df_invalid = pd.DataFrame({"Default": [0, 1, 2, 1]})
    result = validate_target_variable(df_invalid, "Default")
    assert result["valid"] is False
    assert 2 in result["unexpected_values"]


def test_validate_numeric_ranges():
    """Test numeric range validation."""
    df = pd.DataFrame({
        "age": [25, 30, 150, -5],
        "income": [50000, 60000, 70000, 80000]
    })
    
    result = validate_numeric_ranges(df, {"age": (18, 90)})
    assert result["valid"] is False
    assert len(result["issues"]) == 2  # 150 and -5 both out of range


def test_validate_negative_values():
    """Test negative value validation."""
    df = pd.DataFrame({
        "income": [50000, -1000, 60000],
        "loan_amount": [100000, 200000, 300000]
    })
    
    result = validate_negative_values(df, ["income", "loan_amount"])
    assert result["valid"] is False
    assert "income" in result["columns_with_negatives"]


def test_validate_infinite_values():
    """Test infinite value validation."""
    df = pd.DataFrame({
        "col1": [1, 2, np.inf, 4],
        "col2": [1, -np.inf, 3, 4]
    })
    
    result = validate_infinite_values(df)
    assert result["valid"] is False
    assert len(result["columns_with_infinite"]) == 2


def test_generate_data_quality_report():
    """Test comprehensive data quality report."""
    df = pd.DataFrame({
        "Default": [0, 1, 0, 1],
        "Age": [25, 30, 35, 40],
        "Income": [50000, 60000, 70000, 80000]
    })
    
    report = generate_data_quality_report(df, "Default")
    
    assert "dataset_info" in report
    assert "missing_values" in report
    assert "duplicates" in report
    assert "target_validation" in report
    assert report["dataset_info"]["shape"] == (4, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

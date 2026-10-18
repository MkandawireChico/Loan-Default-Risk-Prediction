"""
Data loading utilities for loan default risk prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw synthetic loan applications dataset.
    
    Args:
        data_path: Path to the raw CSV file. If None, uses default path.
        
    Returns:
        DataFrame with raw loan application data
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "loan_applications_synthetic.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded raw data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    return df


def load_cleaned_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the cleaned and feature-engineered loan applications dataset.
    
    Args:
        data_path: Path to the cleaned CSV file. If None, uses default path.
        
    Returns:
        DataFrame with cleaned loan application data
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "loan_applications_clean.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded cleaned data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    return df


def save_cleaned_data(df: pd.DataFrame, data_path: Optional[Path] = None) -> None:
    """
    Save the cleaned and feature-engineered dataset.
    
    Args:
        df: DataFrame to save
        data_path: Path to save the CSV file. If None, uses default path.
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "loan_applications_clean.csv"
    
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    logger.info(f"Saved cleaned data to {data_path}")


def load_model(model_path: Path):
    """
    Load a saved model pipeline from disk.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model/pipeline object
    """
    import joblib
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    return model


def save_model(model, model_path: Path) -> None:
    """
    Save a model pipeline to disk.
    
    Args:
        model: Model/pipeline object to save
        model_path: Path to save the model file
    """
    import joblib
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

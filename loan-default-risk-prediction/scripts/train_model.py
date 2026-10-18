"""
Training script for loan default risk prediction models.

This script provides a command-line interface to train models with proper
data validation, preprocessing, and evaluation.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loading import load_raw_data, save_cleaned_data, load_model, save_model
from src.data.validation import generate_data_quality_report, print_validation_report
from src.data.preprocessing import (
    clean_raw_data,
    prepare_modeling_data,
    create_train_test_split,
    create_preprocessing_pipeline,
    get_modeling_features
)
from src.features.engineering import engineer_all_features
from src.models.training import (
    create_baseline_model,
    create_logistic_regression,
    create_random_forest,
    create_xgboost,
    create_model_pipeline,
    train_model,
    benchmark_models,
    calculate_scale_pos_weight
)
from src.evaluation.metrics import comprehensive_evaluation, print_evaluation_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train loan default risk prediction models")
    parser.add_argument("--model", type=str, default="random_forest",
                       choices=["baseline", "logistic", "random_forest", "xgboost", "benchmark"],
                       help="Model to train")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip data quality validation")
    parser.add_argument("--skip-cleaning", action="store_true",
                       help="Skip data cleaning (use existing cleaned data)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save trained models")
    
    args = parser.parse_args()
    
    np.random.seed(args.random_state)
    
    logger.info("=" * 60)
    logger.info("LOAN DEFAULT RISK PREDICTION - MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load raw data
    logger.info("\n[1] Loading raw data...")
    df = load_raw_data()
    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Data validation
    if not args.skip_validation:
        logger.info("\n[2] Running data quality validation...")
        report = generate_data_quality_report(df, target_col="Default")
        print_validation_report(report)
    
    # Clean data
    if args.skip_cleaning:
        logger.info("\n[3] Skipping cleaning, loading existing cleaned data...")
        from src.data.loading import load_cleaned_data
        df = load_cleaned_data()
    else:
        logger.info("\n[3] Cleaning data...")
        df = clean_raw_data(df)
        
        # Feature engineering
        logger.info("\n[4] Engineering features...")
        df = engineer_all_features(df)
        
        # Save cleaned data
        save_cleaned_data(df)
    
    # Prepare modeling data
    logger.info("\n[5] Preparing modeling data...")
    X, y = prepare_modeling_data(df, target_col="Default")
    
    # Train/test split
    logger.info("\n[6] Creating train/test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, test_size=0.25, random_state=args.random_state
    )
    
    # Get available features
    available_features = get_modeling_features(X_train)
    logger.info(f"Numeric features: {len(available_features['numeric'])}")
    logger.info(f"Categorical features: {len(available_features['categorical'])}")
    
    # Create preprocessing pipeline
    logger.info("\n[7] Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(
        available_features["numeric"],
        available_features["categorical"]
    )
    
    # Train or benchmark models
    if args.model == "benchmark":
        logger.info("\n[8] Benchmarking all models...")
        results_df = benchmark_models(X_train, y_train, preprocessor)
        
        # Save benchmark results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "benchmark_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Benchmark results saved to {results_path}")
        
    else:
        logger.info(f"\n[8] Training {args.model} model...")
        
        # Create model
        if args.model == "baseline":
            model = create_baseline_model(strategy="prevalence")
        elif args.model == "logistic":
            model = create_logistic_regression()
        elif args.model == "random_forest":
            model = create_random_forest()
        elif args.model == "xgboost":
            scale_weight = calculate_scale_pos_weight(y_train)
            model = create_xgboost(scale_pos_weight=scale_weight)
        else:
            raise ValueError(f"Unknown model: {args.model}")
        
        # Create pipeline
        pipeline = create_model_pipeline(preprocessor, model)
        
        # Train
        pipeline = train_model(pipeline, X_train, y_train)
        
        # Evaluate
        logger.info("\n[9] Evaluating model on test set...")
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)
        
        eval_results = comprehensive_evaluation(y_test.values, y_prob, y_pred)
        print_evaluation_report(eval_results)
        
        # Save model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{args.model}_pipeline.joblib"
        save_model(pipeline, model_path)
        
        # Save evaluation results
        import json
        eval_path = output_dir / f"{args.model}_evaluation.json"
        with open(eval_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                return obj
            
            json.dump(convert_types(eval_results), f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Evaluation results saved to {eval_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

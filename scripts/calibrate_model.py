"""
Calibration script for loan default risk prediction models.

This script generates calibration plots, threshold analysis, and
comprehensive calibration reports for trained models.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loading import load_cleaned_data, load_model
from src.data.preprocessing import prepare_modeling_data, create_train_test_split
from src.evaluation.metrics import comprehensive_evaluation, print_evaluation_report
from src.evaluation.calibration import (
    calibrate_model,
    plot_calibration_curve,
    plot_threshold_analysis,
    plot_roc_curve,
    find_optimal_threshold,
    generate_calibration_report
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Calibrate loan default risk prediction models")
    parser.add_argument("--model", type=str, default="logistic",
                       choices=["logistic", "random_forest", "xgboost"],
                       help="Model to calibrate")
    parser.add_argument("--calibration-method", type=str, default="sigmoid",
                       choices=["sigmoid", "isotonic"],
                       help="Calibration method")
    parser.add_argument("--output-dir", type=str, default="visuals",
                       help="Directory to save calibration plots")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("LOAN DEFAULT RISK PREDICTION - MODEL CALIBRATION")
    logger.info("=" * 60)
    
    # Load cleaned data
    logger.info("\n[1] Loading cleaned data...")
    df = load_cleaned_data()
    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Prepare modeling data
    logger.info("\n[2] Preparing modeling data...")
    X, y = prepare_modeling_data(df, target_col="Default")
    
    # Train/test split (use same split as training for consistency)
    logger.info("\n[3] Creating train/test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Load trained model
    logger.info(f"\n[4] Loading trained {args.model} model...")
    model_path = Path("models") / f"{args.model}_pipeline.joblib"
    pipeline = load_model(model_path)
    
    # Get predictions from uncalibrated model
    logger.info("\n[5] Generating predictions from uncalibrated model...")
    y_prob_uncal = pipeline.predict_proba(X_test)[:, 1]
    y_pred_uncal = pipeline.predict(X_test)
    
    # Evaluate uncalibrated model
    logger.info("\n[6] Evaluating uncalibrated model...")
    eval_uncal = comprehensive_evaluation(y_test.values, y_prob_uncal, y_pred_uncal)
    print_evaluation_report(eval_uncal)
    
    # Calibrate model
    logger.info(f"\n[7] Calibrating model using {args.calibration_method} method...")
    calibrated_pipeline = calibrate_model(
        pipeline, X_train, y_train, 
        method=args.calibration_method, 
        cv=5
    )
    
    # Get predictions from calibrated model
    logger.info("\n[8] Generating predictions from calibrated model...")
    y_prob_cal = calibrated_pipeline.predict_proba(X_test)[:, 1]
    y_pred_cal = calibrated_pipeline.predict(X_test)
    
    # Evaluate calibrated model
    logger.info("\n[9] Evaluating calibrated model...")
    eval_cal = comprehensive_evaluation(y_test.values, y_prob_cal, y_pred_cal)
    print_evaluation_report(eval_cal)
    
    # Generate calibration report
    logger.info("\n[10] Generating calibration report...")
    cal_report = generate_calibration_report(y_test.values, y_prob_uncal, y_prob_cal)
    
    print("\n" + "=" * 60)
    print("CALIBRATION COMPARISON")
    print("=" * 60)
    print(f"\nUncalibrated:")
    print(f"  Brier Score: {cal_report['uncalibrated']['calibration']['brier_score']:.4f}")
    print(f"  Log Loss: {cal_report['uncalibrated']['calibration']['log_loss']:.4f}")
    print(f"  ROC-AUC: {cal_report['uncalibrated']['roc_auc']:.4f}")
    print(f"  Gini: {cal_report['uncalibrated']['gini']:.4f}")
    print(f"  KS Statistic: {cal_report['uncalibrated']['ks_statistic']:.4f}")
    
    print(f"\nCalibrated ({args.calibration_method}):")
    print(f"  Brier Score: {cal_report['calibrated']['calibration']['brier_score']:.4f}")
    print(f"  Log Loss: {cal_report['calibrated']['calibration']['log_loss']:.4f}")
    print(f"  ROC-AUC: {cal_report['calibrated']['roc_auc']:.4f}")
    print(f"  Gini: {cal_report['calibrated']['gini']:.4f}")
    print(f"  KS Statistic: {cal_report['calibrated']['ks_statistic']:.4f}")
    
    # Find optimal thresholds
    logger.info("\n[11] Finding optimal thresholds...")
    for metric in ["f1", "precision", "recall"]:
        thresh, metrics = find_optimal_threshold(y_test.values, y_prob_cal, metric)
        print(f"  Optimal threshold for {metric}: {thresh:.2f} ({metric} = {metrics[metric]:.4f})")
    
    # Generate plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n[12] Generating calibration plots...")
    
    # Calibration curve
    fig_cal = plot_calibration_curve(
        y_test.values, y_prob_uncal, y_prob_cal,
        n_bins=10,
        save_path=output_dir / f"{args.model}_calibration_curve.png"
    )
    plt.close(fig_cal)
    
    # Threshold analysis
    fig_thresh = plot_threshold_analysis(
        y_test.values, y_prob_cal,
        save_path=output_dir / f"{args.model}_threshold_analysis.png"
    )
    plt.close(fig_thresh)
    
    # ROC curve
    fig_roc = plot_roc_curve(
        y_test.values, y_prob_uncal, y_prob_cal,
        save_path=output_dir / f"{args.model}_roc_curve.png"
    )
    plt.close(fig_roc)
    
    # Save calibration report
    import json
    cal_report_path = Path("models") / f"{args.model}_calibration_report.json"
    with open(cal_report_path, "w") as f:
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
        
        json.dump(convert_types(cal_report), f, indent=2)
    
    logger.info(f"Calibration report saved to {cal_report_path}")
    
    # Save calibrated model
    cal_model_path = Path("models") / f"{args.model}_calibrated_pipeline.joblib"
    from src.data.loading import save_model
    save_model(calibrated_pipeline, cal_model_path)
    logger.info(f"Calibrated model saved to {cal_model_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""
Explainability script for loan default risk prediction models.

This script generates SHAP-based explanations and fairness analysis
for trained models.
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

from src.data.loading import load_cleaned_data, load_model
from src.data.preprocessing import prepare_modeling_data, create_train_test_split
from src.explainability.shap_analysis import (
    compute_shap_values,
    plot_feature_importance,
    plot_shap_summary,
    generate_shap_report
)
from src.explainability.fairness import (
    fairness_by_attribute,
    generate_fairness_report,
    print_fairness_report
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Explain loan default risk prediction models")
    parser.add_argument("--model", type=str, default="logistic",
                       choices=["logistic", "random_forest", "xgboost"],
                       help="Model to explain")
    parser.add_argument("--skip-shap", action="store_true",
                       help="Skip SHAP analysis")
    parser.add_argument("--skip-fairness", action="store_true",
                       help="Skip fairness analysis")
    parser.add_argument("--output-dir", type=str, default="visuals",
                       help="Directory to save explanation plots")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("LOAN DEFAULT RISK PREDICTION - MODEL EXPLAINABILITY")
    logger.info("=" * 60)
    
    # Load cleaned data
    logger.info("\n[1] Loading cleaned data...")
    df = load_cleaned_data()
    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Prepare modeling data
    logger.info("\n[2] Preparing modeling data...")
    X, y = prepare_modeling_data(df, target_col="Default")
    
    # Train/test split
    logger.info("\n[3] Creating train/test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Load trained model
    logger.info(f"\n[4] Loading trained {args.model} model...")
    model_path = Path("models") / f"{args.model}_pipeline.joblib"
    pipeline = load_model(model_path)
    
    # Get predictions
    logger.info("\n[5] Generating predictions...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # SHAP Analysis
    if not args.skip_shap:
        logger.info("\n[6] Computing SHAP values...")
        try:
            explainer, shap_values, feature_names = compute_shap_values(
                pipeline, X_test, background_samples=100
            )
            
            # Feature importance plot
            logger.info("\n[7] Generating feature importance plot...")
            fig_importance = plot_feature_importance(
                shap_values, feature_names, top_n=20,
                save_path=output_dir / f"{args.model}_shap_feature_importance.png"
            )
            
            # SHAP summary plot
            logger.info("\n[8] Generating SHAP summary plot...")
            fig_summary = plot_shap_summary(
                shap_values, feature_names, max_display=20,
                save_path=output_dir / f"{args.model}_shap_summary.png"
            )
            
            # Generate SHAP report
            logger.info("\n[9] Generating SHAP report...")
            shap_report = generate_shap_report(shap_values, feature_names, X_test, y_test)
            
            # Save SHAP report
            import json
            shap_report_path = Path("models") / f"{args.model}_shap_report.json"
            with open(shap_report_path, "w") as f:
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
                
                json.dump(convert_types(shap_report), f, indent=2)
            logger.info(f"SHAP report saved to {shap_report_path}")
            
            # Print top features
            print("\n" + "=" * 60)
            print("TOP 10 FEATURES (SHAP)")
            print("=" * 60)
            for i, feat in enumerate(shap_report["top_features"], 1):
                print(f"{i:2d}. {feat['feature']}: {feat['mean_abs_shap']:.4f}")
            
        except ImportError:
            logger.warning("SHAP package not installed. Skipping SHAP analysis.")
            logger.warning("Install with: pip install shap")
    
    # Fairness Analysis
    if not args.skip_fairness:
        logger.info("\n[10] Running fairness analysis...")
        
        # Define protected attributes to analyze
        protected_attrs = ["Gender", "Employment_Status", "Customer_Segment", "Education_Level"]
        
        fairness_report = generate_fairness_report(
            X_test, y_test.values, y_prob, y_pred,
            protected_attrs, min_group_size=100
        )
        
        print_fairness_report(fairness_report)
        
        # Save fairness report
        fairness_report_path = Path("models") / f"{args.model}_fairness_report.json"
        import json
        with open(fairness_report_path, "w") as f:
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
            
            json.dump(convert_types(fairness_report), f, indent=2)
        logger.info(f"Fairness report saved to {fairness_report_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("EXPLAINABILITY ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

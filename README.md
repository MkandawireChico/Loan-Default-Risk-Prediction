# Loan Default Risk Prediction — Nyanja Commercial Bank  
## Malawi Commercial Banking — Synthetic Portfolio Dataset

## Project Overview

This project builds a rigorous, reproducible credit-risk prediction workflow for a fictional commercial bank in Malawi: from synthetic data generation through data validation, cleaning, exploratory analysis, feature engineering, classification modeling (Logistic Regression, Random Forest, XGBoost), probability calibration, threshold analysis, model interpretation, fairness assessment, and risk segmentation — ending in an interactive Streamlit dashboard and comprehensive model documentation.
 
**Recent Enhancements:** The project has been upgraded to a professional, modular codebase with reusable data validation, preprocessing pipelines, model benchmarking with cross-validation, probability calibration, fairness analysis, automated testing, and an interactive dashboard.

## Problem Statement

A fictional commercial bank provides personal and SME loans to customers across different income groups and regions. The bank wants to improve its credit-risk assessment process by identifying borrowers with a higher probability of default, using historical borrower and loan characteristics rather than relying on manual judgment alone.

## Malawi Banking Context

The synthetic data is shaped to be reasonably representative of a Malawian commercial banking environment: a mix of salaried (government and private), SME, self-employed, and youth/student customer segments; loan purposes spanning personal, business, agriculture, education, and asset finance; and realistic commercial lending interest rates (18–35% annually). **Branch names used (Blantyre, Lilongwe, Mzuzu, Zomba, Mangochi, Kasungu, Salima, Dedza, Rumphi, Mulanje) are real Malawian towns used as plausible location labels only — the notebook found no meaningful default-rate difference between branches (all cluster within 16–18%), which is expected since branch was not built into the synthetic default-generating process, and is stated here rather than implied as a finding.**

## Project Objectives

1. Analyse borrower and loan characteristics associated with default
2. Identify the factors most strongly influencing default risk
3. Build a baseline Logistic Regression model and a stronger tree-based model
4. Compare model performance using metrics appropriate for imbalanced classification
5. Generate predicted probability of default for every borrower
6. Segment borrowers into transparent, validated risk categories
7. Translate technical findings into credit-risk recommendations
8. Specify a Power BI dashboard for credit-risk analysts and management
9. Demonstrate the complete workflow from raw synthetic data to business decision support

## Dataset

**20,007 loan applications** (20,000 generated + 1% duplicate rows deliberately injected, then handled in cleaning — see below), covering demographics, employment, income, loan terms, credit history, and financial-behavior fields, plus the `Default` target (1 = defaulted, 0 = did not).

## Data-Generation Methodology

**The target variable is generated from a genuine logistic risk model, not random assignment.** `Default` is produced by combining the risk factors a real credit officer would weigh — debt-to-income ratio, loan-to-income ratio, previous defaults, late-payment frequency, days past due, repayment burden, existing loan count, savings buffer, account tenure, employment stability, and customer segment — into a weighted logistic function, with random noise added so outcomes are probabilistic rather than deterministic (two borrowers with identical characteristics can have different outcomes, exactly as in reality).

This was verified, not assumed, before any modeling began (`notebooks/loan_default_risk_analysis.ipynb`, Section 1.4):

| Check | Result |
|---|---|
| Default rate by DTI quartile | 6.3% → 12.4% → 17.6% → **32.7%** (clean monotonic increase) |
| Default rate by previous defaults | 14.7% (0) → 32.0% (1) → 57.3% (2) → 83.3% (3) |
| Default rate by credit score category | Very Good 11.4% → Good 21.0% → Fair 41.9% → Poor **75.8%** |
| Default rate by employment status | Unemployed highest (28.1%), Business Owner lowest (15.6%) |

**Overall default rate: 17.24%** — within the intended 10–25% range.

## Data-Quality Issues (Deliberately Injected)

Ten categories, each reproducible via its own random seed: missing income (~2%), missing credit scores (~2%), missing employment status (~1.5%), extreme income outliers (10–20x, ~0.3%), extreme loan amounts (5–10x, ~0.3%), inconsistent employment-category spelling/casing (~1.5%), duplicate records (~1%), invalid ages (values like 12, 130, -5; ~0.2%), missing branch names (~1%), and inconsistent loan-purpose labels (~1.5%). None were removed silently — each is documented and handled explicitly in Section 4 of the notebook.

## Data-Cleaning Methodology

- **Missing income, credit score, employment status:** left as missing in the cleaned CSV; imputed *inside* the modeling pipeline using training-set-only statistics (median/mode), specifically to avoid leaking test-set information into how missing values are filled.
- **Extreme income/loan outliers:** winsorized (capped) at the 99.5th percentile rather than deleted, preserving the row while bounding the distortion.
- **Inconsistent employment/loan-purpose labels:** mapped back to their canonical categories via an explicit lookup.
- **Duplicate records:** exact duplicates dropped.
- **Invalid ages:** values outside a plausible 18–90 adult borrowing range treated as missing (not guessed at) and imputed downstream, same as other missing numerics.
- **Missing branch names:** given an explicit `"Unknown"` category rather than dropped or guessed — the missingness itself is informative.

## Exploratory Data Analysis

12+ cuts of the data across employment status, customer segment, loan purpose, branch, income quintile, loan amount quintile, DTI quartile, previous defaults, late-payment frequency, and credit score category, plus loan-amount-vs-income and repayment-burden relationships — 7 exported charts in `visuals/`. Full detail in the notebook, Section 6.

## Feature Engineering

| Feature | Definition | Business meaning |
|---|---|---|
| `Monthly_Repayment_Ratio` | Repayment ÷ income | direct affordability measure |
| `Savings_to_Income_Ratio` | Savings ÷ annual income | financial cushion against income shocks |
| `Previous_Default_Flag` | Binary version of prior default history | "has this happened before at all," simpler than the count |
| `Late_Payment_Frequency` | Late payments ÷ credit history length | normalizes for how long the credit history actually is |
| `Loan_Burden_Category` | Banded repayment ratio (Low/Moderate/High/Severe) | a label a loan officer can act on directly |
| `Credit_Risk_Category` | Banded credit score (Poor/Fair/Good/Very Good) | mirrors standard credit bureau bands |

## Machine-Learning Methodology

- **Stratified train/test split** (75/25) — preserves the ~17% default rate in both sets, since a plain random split could otherwise produce meaningfully different class balances by chance.
- **Leakage-aware preprocessing pipeline:** median imputation + scaling for numeric features, most-frequent imputation + one-hot encoding for categorical features, fit on the training set only.
- **Excluded from modeling, with reasons stated:** `Customer_ID`/`Loan_ID` (no predictive meaning), `Monthly_Repayment` (redundant with `Monthly_Repayment_Ratio`), `Loan_Burden_Category`/`Credit_Risk_Category` (double-count their own source ratio/score), `Collateral_Value` (mostly a proxy for a downstream business decision, not an input a first-pass model should lean on). No variable was excluded for being "too predictive" — nothing in this dataset is generated *after* or *because of* the outcome, so there is no leaked information to guard against beyond standard pipeline discipline.
- **Three classifiers trained:** Logistic Regression (`class_weight="balanced"`), Random Forest (depth/leaf-constrained to reduce overfitting), and XGBoost (`scale_pos_weight` rebalancing).

## Model Evaluation

**Why accuracy alone is insufficient:** with a 17% default rate, a trivial "predict no-default for everyone" model would score ~83% accuracy while catching zero real defaulters — the worst possible outcome for credit risk management. Precision, recall, F1, and ROC-AUC are evaluated together instead.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.752 | 0.371 | 0.629 | 0.467 | **0.776** |
| Logistic Regression | 0.719 | 0.340 | **0.672** | 0.452 | 0.773 |
| XGBoost | 0.735 | 0.353 | 0.649 | 0.458 | 0.770 |

*(Full classification reports and confusion matrices in the notebook, Section 9; `visuals/08_confusion_matrices.png`, `visuals/09_roc_curves.png`.)*

## Model Comparison

Random Forest has the best ROC-AUC; Logistic Regression has the best recall on the Default class. **Model choice is not simply "pick the highest AUC"** — Logistic Regression's coefficients are directly explainable to a credit committee, while the tree-based models capture nonlinear interactions (e.g. high DTI *combined with* prior defaults) better but are harder to explain. A real deployment decision would weigh both the metrics and this interpretability trade-off.

## Threshold Analysis

The default 0.50 threshold is not automatically optimal. Since a missed defaulter (false negative) typically costs a bank more than an unnecessarily-scrutinized good borrower (false positive), a threshold favoring recall was investigated using Random Forest's probabilities (notebook Section 10):

| Threshold | Precision | Recall | F1 | Borrowers Flagged High-Risk |
|---|---|---|---|---|
| 0.50 (default) | *(see notebook)* | *(see notebook)* | — | — |
| **0.40 (recommended)** | 0.294 | **0.782** | 0.427 | 2,297 of 5,002 (test set) |

**Recommended threshold: 0.40** — raises recall to 78.2% (catching substantially more real defaulters) at the cost of precision. **This is a starting point for discussion, not a final answer:** the actual threshold a bank should use requires validating false-positive cost (lost good business, manual review capacity) against false-negative cost (bad debt written off) with the credit-risk team's real figures, which this synthetic project has no way to know.

## Feature Importance and Model Interpretation

**Logistic Regression** (top risk-increasing factors): `Existing_Loan_Count`, `Debt_to_Income_Ratio`, `Previous_Default_Count`, `Monthly_Repayment_Ratio`, being `Unemployed`, `Loan_to_Income_Ratio`. Top risk-*decreasing* factors: `Savings_to_Income_Ratio`, `Savings_Balance`, `Account_Tenure_Months`, being in the `Salaried Premium` segment.

**Random Forest** (top features by importance): `Debt_to_Income_Ratio` (16.2%), `Existing_Loan_Count` (10.3%), `Existing_Debt` (9.2%), `Credit_Score` (5.7%), `Late_Payment_Frequency` (5.2%).

**Both models independently rank `Debt_to_Income_Ratio` and `Existing_Loan_Count` among the top 2–3 predictors**, despite working in completely different ways (linear coefficients vs. tree splits) — agreement between structurally different models is stronger evidence these are genuine signal, not an artifact of one algorithm.

## Key Findings

1. **Debt-to-income ratio is the single strongest and most consistent default predictor**, confirmed independently by both Logistic Regression and Random Forest, with a default rate that rises more than 5x from the lowest to highest DTI quartile (6.3% → 32.7%).
2. **Prior default history is the sharpest behavioral signal**: one previous default more than doubles the default rate (14.7% → 32.0%), and it keeps climbing with each additional prior default.
3. **Credit score bands cleanly separate risk**: Poor-band borrowers default at 75.8% versus 11.4% for Very Good — a gap wide enough to be genuinely decision-useful on its own.
4. **Employment stability matters, but less than debt burden or history**: Unemployed applicants default at 28.1% versus ~15–18% for every employed category — a real but comparatively modest gap next to the DTI and prior-default effects.
5. **Branch location shows no meaningful effect** (16.0–18.2% across all 10 branches) — reported here as a genuine (negative) finding rather than omitted, and expected given branch was not part of the synthetic default-generating process.
6. **The risk segmentation validates cleanly against actual outcomes**: Low/Medium/High risk categories (by predicted probability) show actual test-set default rates of **3.0%, 11.0%, and 37.0%** respectively — a clean monotonic separation confirming the categories mean what they claim.
7. **No single model dominates on every metric** — Random Forest wins on ROC-AUC, Logistic Regression on recall — reinforcing that model selection for credit risk is a business trade-off, not a single-number leaderboard exercise.

## Risk Segmentation

| Risk Category | Borrowers (test set) | Actual Default Rate | Avg Loan Amount | Avg Income | Avg DTI | Avg Previous Defaults |
|---|---|---|---|---|---|---|
| Low Risk (p < 0.20) | 862 | 3.0% | 1,881,180 | 504,619 | 0.03 | 0.01 |
| Medium Risk (0.20–0.50) | 2,678 | 11.0% | 2,254,443 | 469,096 | 0.12 | 0.08 |
| High Risk (p > 0.50) | 1,462 | 37.0% | 2,089,558 | 392,863 | 0.33 | 0.33 |

(Figures in MWK-equivalent synthetic currency units. Boundaries are fixed probability thresholds, not percentiles, so they retain the same meaning even if the overall applicant risk profile shifts over time.)

## Business Recommendations

- **Enhanced review for High Risk applications** (37% actual default rate) — not automatic rejection; a human affordability assessment given the segment's average DTI of 0.33 and near-universal prior default history.
- **Additional affordability assessment weighted toward DTI and existing loan count specifically** (Finding #1), since these are the two factors both models agree matter most — a streamlined affordability check could focus disproportionately here rather than treating all fields as equally important.
- **Early-warning monitoring for existing borrowers who cross into Medium/High risk territory** after loan approval — since prior defaults and late payments are such strong signals (Finding #2), a borrower's *behavior after* approval is itself a leading indicator worth monitoring, not just the application-time snapshot.
- **Manual review of borderline applications** — the 0.40 threshold recommendation deliberately trades precision for recall, meaning a meaningful share of flagged applicants will not actually default; a manual review layer prevents that trade-off from becoming automatic rejections of creditworthy borrowers.
- **Do not treat unemployment as disqualifying on its own** (Finding #4) — its effect is real but modest next to DTI and credit history, so a blanket employment-status rule would be less accurate than the model's combined assessment.
- **Regular model performance monitoring** — this model is trained on a synthetic snapshot; a real deployment would need ongoing recalibration as the bank's actual portfolio and macroeconomic conditions evolve.
- **Responsible lending and human oversight**: this model is a decision-*support* tool, not a decision-*maker* — every recommendation above assumes a human credit officer remains in the loop, particularly for Medium and High risk cases near the threshold boundary.

## Dashboard Description

A 4-page Power BI dashboard specification (`dashboard/power_bi_dashboard_specification.md`, not yet built): Credit Risk Overview, Borrower and Loan Profile, Default Drivers, and Predicted Credit Risk — including a specific plan to show the risk-segmentation validation (Finding #6) directly on the dashboard rather than leaving it in a report someone has to already trust.

## Limitations

- This is a cross-sectional synthetic snapshot, not a real historical loan book — there is no genuine time dimension, so trend analysis (e.g. "default rate over the last 12 months") is not available without extending the simulation.
- The specific numbers throughout this README (e.g. "17.24% default rate," "0.776 ROC-AUC") are properties of this simulation's parameters and random seed, not measurements of any real bank's portfolio.
- The recommended 0.40 threshold is a reasoned starting point, not a validated business decision — it requires real cost figures from an actual credit-risk team.
- Collateral information exists in the dataset but was deliberately excluded from modeling (see Machine-Learning Methodology) — a production model might handle collateral differently depending on the bank's actual underwriting policy.
- The model does not account for macroeconomic factors (currency depreciation, inflation, sector-specific shocks) that materially affect real-world default risk in Malawi.

## Ethical Considerations

**This project uses entirely synthetic data created for educational and portfolio purposes. The bank, customers, loan records, branches, financial figures, and model results do not represent any real commercial bank or actual banking data in Malawi.** "Nyanja Commercial Bank" is fictional. No real customer or banking data was used at any point. Credit-risk models in real deployments carry fairness and discrimination risks (e.g. proxy variables correlating with protected characteristics) that a production system would need to audit for — this portfolio project does not claim to have done that audit, since it is not built on real data or a real lending policy.

## Tools Used

Python (Pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-learn, XGBoost, joblib) · Jupyter Notebook · Power BI (specification)

## Project Structure

```
loan-default-risk-prediction/
├── data/
│   ├── raw/
│   │   └── loan_applications_synthetic.csv     # As generated + injected issues
│   └── processed/
│       └── loan_applications_clean.csv          # Cleaned + feature-engineered
├── notebooks/
│   └── loan_default_risk_analysis.ipynb         # Full workflow, executed with real outputs
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loading.py                          # Data loading utilities
│   │   ├── validation.py                       # Data validation functions
│   │   └── preprocessing.py                    # Data cleaning and preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py                      # Feature engineering functions
│   ├── models/
│   │   ├── __init__.py
│   │   └── training.py                         # Model training functions
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                          # Credit-risk evaluation metrics
│   │   └── calibration.py                      # Probability calibration
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_analysis.py                    # SHAP-based explanations
│   │   └── fairness.py                         # Fairness analysis
│   ├── analytics/
│   │   ├── __init__.py
│   │   └── sql_queries.py                      # SQL analytics queries
│   └── generate_data.py                        # Standalone, reproducible data generator
├── scripts/
│   ├── train_model.py                          # Model training script
│   ├── calibrate_model.py                      # Model calibration script
│   └── explain_model.py                        # Model explainability script
├── tests/
│   ├── __init__.py
│   └── test_data_validation.py                 # Automated tests
├── models/
│   ├── baseline_pipeline.joblib
│   ├── logistic_pipeline.joblib
│   ├── logistic_calibrated_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   ├── xgboost_pipeline.joblib
│   └── benchmark_results.csv
├── visuals/                                    # Chart exports from notebook and scripts
├── dashboard/
│   └── power_bi_dashboard_specification.md     # Power BI specification
├── .github/
│   └── workflows/
│       └── ci.yml                             # GitHub Actions CI
├── app.py                                     # Streamlit dashboard
├── requirements.txt
├── pyproject.toml                             # Project configuration (pytest, ruff)
├── .gitignore
└── README.md
```

## Reproducibility Instructions

```bash
git clone <this-repo>
cd loan-default-risk-prediction
pip install -r requirements.txt

# Option A: Run the full notebook (generates data + trains models + exports everything)
jupyter notebook notebooks/loan_default_risk_analysis.ipynb

# Option B: Use the modular scripts (recommended for production workflows)
# Generate synthetic data
python src/generate_data.py

# Train models with data validation
python scripts/train_model.py --model benchmark  # Benchmark all models
python scripts/train_model.py --model logistic    # Train specific model

# Calibrate model probabilities
python scripts/calibrate_model.py --model logistic

# Generate explainability and fairness reports
python scripts/explain_model.py --model logistic

# Run automated tests
pytest tests/ -v

# Lint code
ruff check src/ tests/ scripts/

# Launch interactive dashboard
streamlit run app.py
```

All paths use `np.random.seed(42)` and reproduce the same dataset described in this README.

## Synthetic-Data Disclaimer

**This project uses entirely synthetic data created for educational and portfolio purposes. The bank, customers, loan records, branches, financial figures, and model results do not represent any real commercial bank or actual banking data in Malawi.**

## Author

Chico Mkandawire
MSc Data Science, Malawi University of Science and Technology (in progress)
BSc Management Information Systems, Malawi University of Business and Applied Sciences

---


 

 

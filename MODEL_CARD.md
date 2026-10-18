# Model Card: Loan Default Risk Prediction

## Model Details

- **Model Name:** Logistic Regression Pipeline (Calibrated)
- **Model Type:** Binary Classification
- **Target Variable:** Default (1 = defaulted, 0 = did not default)
- **Prediction Task:** Probability of default for loan applicants
- **Date Developed:** October 2026
- **Model Version:** 1.0

## Intended Use

- **Primary Use:** Decision-support tool for credit risk assessment at loan application time
- **Intended Users:** Credit officers, risk analysts, lending managers
- **Use Cases:**
  - Estimate probability of default for new loan applications
  - Segment applicants into risk categories for manual review prioritization
  - Monitor portfolio risk distribution
  - Support underwriting decisions (not replace human judgment)

## Training Data

- **Dataset Size:** 20,007 loan applications (synthetic)
- **Time Period:** Cross-sectional snapshot (no temporal dimension)
- **Data Source:** Synthetic data generated via logistic risk model
- **Default Rate:** 17.24%
- **Features:** 37 features including demographics, employment, income, loan terms, credit history, and financial ratios

### Key Features

**Demographic:**
- Age, Gender, Marital Status, Education Level

**Employment & Income:**
- Employment Status, Occupation, Monthly Income, Account Tenure

**Loan Characteristics:**
- Loan Amount, Loan Term, Interest Rate, Loan Purpose, Monthly Repayment

**Credit History:**
- Credit Score, Previous Defaults, Late Payments, Days Past Due

**Financial Ratios:**
- Debt-to-Income Ratio, Loan-to-Income Ratio, Monthly Repayment Ratio
- Savings-to-Income Ratio, Late Payment Frequency

**Categorical:**
- Customer Segment, Branch Location, Loan Application Channel, Collateral Type

## Model Performance

### Test Set Metrics (n=5,002)

**Discrimination:**
- ROC-AUC: 0.772
- Gini Coefficient: 0.544
- KS Statistic: 0.403

**Calibration (after Platt scaling):**
- Brier Score: 0.118
- Log Loss: 0.384

**Classification at 0.50 threshold:**
- Accuracy: 84.1%
- Precision: 62.1%
- Recall: 19.6%
- Specificity: 97.5%
- F1 Score: 29.8%

**Optimal Threshold (F1-maximizing):**
- Threshold: 0.26
- F1 Score: 0.471

### Cross-Validation Results (5-fold)

| Metric | Mean | Std |
|--------|------|-----|
| ROC-AUC | 0.783 | 0.008 |
| Accuracy | 0.732 | 0.012 |
| Precision | 0.355 | 0.021 |
| Recall | 0.680 | 0.018 |
| F1 | 0.466 | 0.015 |

## Model Interpretation

### Top Risk-Increasing Factors (Logistic Regression Coefficients)

1. **Existing Loan Count** (+0.82)
2. **Debt-to-Income Ratio** (+0.76)
3. **Previous Default Count** (+0.71)
4. **Monthly Repayment Ratio** (+0.68)
5. **Unemployed Status** (+0.54)

### Top Risk-Decreasing Factors

1. **Savings-to-Income Ratio** (-0.89)
2. **Savings Balance** (-0.67)
3. **Account Tenure Months** (-0.52)
4. **Salaried Premium Segment** (-0.41)

## Risk Segmentation

| Risk Category | Probability Range | Test Set Default Rate | Borrower Count |
|---------------|-------------------|----------------------|----------------|
| Low Risk | < 0.20 | 3.0% | 862 |
| Medium Risk | 0.20 - 0.50 | 11.0% | 2,678 |
| High Risk | > 0.50 | 37.0% | 1,462 |

## Fairness Analysis

### Disparities Across Protected Attributes

**Gender:**
- Default Rate Disparity: 1.26% (Male: 17.83%, Female: 16.57%)
- Prediction Disparity: 0.20%

**Employment Status:**
- Default Rate Disparity: 12.74% (Unemployed: 27.75%, Business Owner: 15.01%)
- Prediction Disparity: 17.69%

**Customer Segment:**
- Default Rate Disparity: 14.35% (Youth/Student: 25.19%, Salaried Premium: 10.85%)
- Prediction Disparity: 17.99%

**Note:** These disparities reflect underlying risk differences in the synthetic data. A real deployment would require additional fairness auditing with actual demographic data and regulatory considerations.

## Limitations

1. **Synthetic Data:** Model trained on entirely synthetic data; performance on real data may differ significantly
2. **Cross-Sectional:** No temporal dimension; cannot capture time-varying risk factors or macroeconomic trends
3. **No Macroeconomic Factors:** Does not account for inflation, currency depreciation, or sector-specific shocks
4. **Limited Geographic Scope:** Synthetic Malawian context only; not generalizable to other regions
5. **No Recovery Modeling:** Does not estimate loss given default (LGD) or exposure at default (EAD)
6. **Collateral Excluded:** Collateral value deliberately excluded from modeling; real underwriting may weight this differently

## Ethical Considerations

- **Decision Support, Not Decision Maker:** Model outputs should inform human credit officers, not automatically approve/deny loans
- **Human Oversight Required:** Manual review recommended for Medium and High risk applications
- **Fairness Monitoring:** Regular fairness audits required in production to detect and mitigate bias
- **Transparency:** Model coefficients and feature importance available for explanation to stakeholders
- **No Protected Attributes Used:** Gender, ethnicity, and other protected characteristics not used as direct features (though correlated features may exist)

## Maintenance & Monitoring

### Recommended Monitoring

- **Performance Drift:** Track ROC-AUC, calibration, and default rate by risk segment monthly
- **Feature Drift:** Monitor distribution changes in key features (DTI, credit score, income)
- **Fairness Drift:** Regular fairness audits across protected attributes
- **Calibration:** Recalibrate annually or when significant drift detected

### Retraining Schedule

- **Recommended:** Retrain annually with new loan performance data
- **Trigger Conditions:** Significant performance drop (>5% ROC-AUC), feature drift, or regulatory changes

## Technical Details

### Preprocessing Pipeline

- **Numeric Features:** Median imputation + StandardScaler
- **Categorical Features:** Most-frequent imputation + OneHotEncoder
- **Leakage Prevention:** Preprocessing fit on training data only

### Model Configuration

```python
LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)
```

### Calibration

- **Method:** Platt Scaling (sigmoid)
- **Cross-Validation:** 5-fold
- **Calibration Improvement:** Brier Score reduced from 0.189 to 0.118

## Model Files

- **Uncalibrated Model:** `models/logistic_pipeline.joblib`
- **Calibrated Model:** `models/logistic_calibrated_pipeline.joblib`
- **Evaluation Report:** `models/logistic_evaluation.json`
- **Calibration Report:** `models/logistic_calibration_report.json`
- **Fairness Report:** `models/logistic_fairness_report.json`

## Contact & Governance

- **Model Owner:** Credit Risk Team
- **Developer:** Chico Mkandawire
- **Review Date:** October 2026
- **Next Review:** October 2027

## Disclaimer

**This model is trained on entirely synthetic data created for educational purposes. The bank, customers, loan records, branches, financial figures, and model results do not represent any real commercial bank or actual banking data in Malawi. This model should not be used for real lending decisions without validation on actual historical data and approval from relevant regulatory authorities.**

# Model card: Nyanja Credit Risk Intelligence v2

## Model identity

- Model version: `credit-risk-calibrated-v2`
- Model family: Logistic Regression with sigmoid probability calibration
- Target: `Default` (simulated binary outcome)
- Training data: committed synthetic loan-application dataset
- Model artifact: generated at `models/v2/credit_risk_calibrated_v2.joblib`
- Evaluation report: generated at `models/v2/evaluation_v2.json`
- Metadata: generated at `models/v2/metadata.json`

## Intended use

The model exists for:
- Demonstrating an end-to-end credit-risk ML workflow.
- Teaching probability calibration, threshold selection and evaluation.
- Demonstrating API inference and model-governance interfaces.
- Exploring subgroup error metrics on simulated data.

The model is not intended for real credit decisions, underwriting, eligibility, pricing, collections, customer ranking or other consequential decisions.

## Data and population

The fictional dataset is generated for a Malawi-oriented commercial-bank scenario. All applicants, financial values, branch labels and outcomes are synthetic. The target is sampled from a probabilistic risk function defined in the data generator.

The synthetic distribution is not a representative sample of Malawi's population, borrowers, banking customers or loan performance.

## Features and governance

The model uses 33 application and credit-history features. Direct demographic fields are excluded from predictive inputs:

- Age
- Gender
- Marital status
- Education level

These fields remain available for post-hoc subgroup diagnostics. Excluding direct demographic fields does not guarantee fairness. Other model inputs may act as proxies, and the data generator itself may encode structural relationships.

## Training and evaluation protocol

1. Exact duplicate rows are removed before partitioning.
2. Stratified train, validation and final test partitions are created.
3. Winsorisation caps are estimated on the training partition only.
4. The same caps are applied to validation and test data.
5. Imputation, scaling and one-hot encoding are fitted within the model pipeline.
6. Sigmoid calibration uses cross-validation inside the training partition.
7. A review threshold is selected on validation data under a stated review-capacity constraint.
8. Final metrics are computed once on the held-out test partition.

The generated report records partition sizes, preprocessing caps, feature lists, threshold selection, final metrics and subgroup diagnostics. Do not copy metrics from earlier model versions into a v2 report.

## Metrics

The evaluation report includes:
- ROC-AUC and PR-AUC.
- Brier score and log loss.
- Precision, recall, specificity and F1 at the selected threshold.
- Confusion matrix and review rate.
- Subgroup observed default rates, review rates, recall, false-positive rates and precision.
- 95% Wilson intervals for reported subgroup proportions.

The threshold is a simulated operational setting. It is not a lending policy or a recommendation for real institutions.

## Explainability

The application reports mean fold-wise logistic coefficient contributions on the uncalibrated log-odds scale. Contributions are model associations, not causal effects or probability-point changes. They do not explain the subsequent calibration mapping.

## Limitations

- Synthetic, cross-sectional dataset only.
- No external validation against real institutional data.
- No prospective or temporal validation.
- No evidence of real-world calibration or discrimination.
- No causal interpretation of model features.
- No guarantee of fairness or legal compliance.
- Threshold selection depends on simulated review capacity and error trade-offs.
- No automated lending decision is produced.

## Safety and privacy

The API stores metadata-only assessment audit events. Applicant feature values, names, contact details and identifiers are not persisted by the audit layer. Do not upload real customer data.

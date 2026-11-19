# Model card: Nyanja Credit Risk Intelligence v2

## Model identity

- Model version prefix: `credit-risk-calibrated-v2-<UTC timestamp>-<dataset hash prefix>` (the exact generated version is recorded in metadata.json)
- Model family: Logistic Regression with sigmoid probability calibration
- Target: `Default` (simulated binary outcome)
- Training data: committed synthetic loan-application dataset
- Model artifact: generated at `models/v2/credit_risk_calibrated_v2.joblib`
- Evaluation report: generated at `models/v2/evaluation_v2.json`
- Metadata: generated at `models/v2/metadata.json`
- Registry: model version, training provenance, artifact SHA-256 and dataset SHA-256 are recorded in PostgreSQL on API startup

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
- Stratified bootstrap 95% intervals for held-out ROC-AUC, PR-AUC, Brier score and log loss.
- Fixed-width calibration bins with expected calibration error and Wilson intervals for observed default rates.

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


## Separate synthetic chronological experiment

The repository also includes a separate chronological stress-test in scripts/run_temporal_validation.py. It assigns 36 synthetic months, introduces a deliberately simulated macro-stress shift in the final eight months, regenerates the synthetic outcome, and compares a prior-only baseline, calibrated Logistic Regression and calibrated Random Forest on a future time window. Whole-month partitions are used. Preprocessing caps are learned from the historical training window, calibration uses expanding-window TimeSeriesSplit folds within training, and the review threshold is selected on validation data.

This experiment does not change the v2 model card identity or its cross-sectional evaluation report. It is not temporal validation against observed loan outcomes. The generated dates, stress index and labels are simulation constructs. Results must not be presented as evidence of predictive performance in Malawi or any real lending institution.


## Monitoring and drift

The training workflow stores a reference profile from the training partition. The profile includes quantile distributions for numeric model inputs, top-category distributions for categorical inputs, missingness rates, and a prediction-probability reference derived from validation predictions.

The batch monitoring endpoint compares unlabeled batches with this reference using Population Stability Index (PSI). It reports per-feature PSI, missing-rate changes, unseen-category rates, and prediction-probability PSI. Conventional bands are below 0.10 (low), 0.10 to below 0.20 (moderate), and 0.20 or higher (high). These values are triage conventions, not universal thresholds.

PSI detects distribution differences. It does not establish model degradation, predictive accuracy, causality, fairness or the need for retraining. Outcome-based monitoring requires reliable labels after an appropriate outcome-maturation period. The current implementation uses synthetic records only.

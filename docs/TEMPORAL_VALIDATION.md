# Synthetic chronological validation experiment

## Why this exists

The existing v2 model uses a stratified random split. That is useful for a reproducible cross-sectional baseline, but it does not test how a model trained on earlier applications behaves on later applications.

This experiment adds a separate, explicitly synthetic time dimension. It does not replace the v2 baseline and does not claim temporal validation against real lending outcomes.

## Scenario design

- The script starts from the committed synthetic application dataset and removes exact duplicate rows.
- It assigns application dates across 36 synthetic months beginning in January 2023.
- It creates a monthly macro-stress index with seasonal variation and a deliberately elevated stress regime in the final eight months.
- It regenerates the binary target from a documented synthetic risk function. Macro stress also affects the simulated outcome, while the stress index and date fields are excluded from model inputs.
- A fixed random seed makes the scenario reproducible.

The date and macro variables are simulation controls. They are not observations of Malawi's economy, bank lending, or actual borrower behaviour.

## Evaluation protocol

| Window | Months | Purpose |
|---|---:|---|
| Training | 0–23 | Fit preprocessing and model |
| Validation | 24–29 | Select review threshold under a stated review-capacity limit |
| Future test | 30–35 | Evaluate once on later synthetic applications |

The partitions use whole months and never shuffle records across time boundaries. Winsorisation caps are learned from the training window only. Imputation, scaling, and category encoding remain inside the model pipeline. Probability calibration uses expanding-window TimeSeriesSplit folds within the training window. The validation set selects the operating threshold. The future test set is reserved for final evaluation.

The experiment compares a prior-only baseline, calibrated Logistic Regression, and calibrated Random Forest under identical chronological partitions. Each candidate selects its own review threshold on validation data under the same review-capacity constraint. The future test window reports ROC-AUC, PR-AUC, Brier score, log loss, threshold metrics, and month-by-month observed default rates, mean predicted probabilities, ROC-AUC where both classes are present, and Brier scores. The future test results are descriptive comparisons only. They do not select a winning model.

## Run

    python scripts/run_temporal_validation.py

Outputs are written to models/temporal/:
- temporal_synthetic_dataset.csv
- temporal_synthetic_model.joblib
- temporal_validation_report.json

## Interpretation limits

This is a controlled software and methodology exercise. The stress regime and outcomes are simulated by code. Good performance on this scenario would only show how the implementation behaves under the chosen simulation assumptions. It would not establish real-world predictive accuracy, fairness, calibration, stability, or suitability for lending.

Do not use the outputs for real applicants, lending decisions, pricing, eligibility, collections, or credit policy.

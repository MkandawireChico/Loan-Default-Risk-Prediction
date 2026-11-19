# Technical presentation guide

## Suggested presentation title

Nyanja Credit Risk Intelligence: Building a Responsible, End-to-End Credit Risk Analytics Platform

## Suggested 12-slide structure

1. **Problem framing**
   - Explain the fictional bank scenario.
   - Describe the difference between portfolio monitoring and individual application assessment.
   - State the synthetic-data boundary.

2. **Project objectives**
   - Demonstrate applied ML, BI, API engineering, model governance and deployment.
   - Define the intended users: portfolio analysts, model-risk reviewers and technical maintainers.

3. **Synthetic data design**
   - Explain the generator, application fields and probabilistic target.
   - Show why synthetic data is useful for reproducibility.
   - Explain why results are not evidence about real Malawi borrowers.

4. **Data quality and feature engineering**
   - Show missingness, category inconsistencies, duplicates and outlier handling.
   - Explain affordability ratios and credit-history features.
   - Distinguish application-time features from target outcomes.

5. **Model development protocol**
   - Explain stratified train, validation and test partitions.
   - Show how preprocessing is fitted on training data.
   - Explain why the final test set is not used for threshold selection.

6. **Calibration and operating threshold**
   - Explain the difference between ranking and probability calibration.
   - Show Brier score, log loss, fixed-width calibration bins and expected calibration error.
   - Explain stratified bootstrap 95% intervals for held-out metrics and Wilson intervals for bin rates.
   - Explain the review-capacity constraint and its trade-offs.
   - Compare the prior-only baseline, calibrated Logistic Regression and calibrated Random Forest on a later synthetic time window.

7. **Fairness and model governance**
   - Explain demographic feature exclusion.
   - Show subgroup diagnostics and Wilson intervals.
   - Explain why removing protected attributes does not prove fairness.

8. **System architecture**
   - Walk through React, FastAPI, the ML pipeline and PostgreSQL.
   - Explain the model artifact and metadata generated during the image build.
   - Explain metadata-only audit persistence.
   - Demonstrate model registry versioning, active/archived status and artifact/dataset fingerprints.

9. **Live product demonstration**
   - Open Portfolio Intelligence.
   - Inspect branch and customer-segment breakdowns.
   - Load a synthetic application, change loan and affordability inputs, and run an assessment.
   - Inspect feature contributions.
   - Upload a synthetic batch and download results.
   - Open Model Governance and inspect held-out metrics, bootstrap intervals, calibration bins and subgroup diagnostics.
   - Compare the prior baseline, calibrated Logistic Regression and calibrated Random Forest across the future synthetic test window.

10. **Testing and deployment**
    - Show GitHub Actions quality checks, including execution of the synthetic chronological experiment.
    - Explain Docker Compose services and health checks.
    - Show the deployment guide and environment configuration.

11. **Limitations and responsible use**
    - Emphasise synthetic data and lack of external validation.
    - Explain model and fairness limitations.
    - State clearly that the model does not make lending decisions.

12. **Next steps**
    - Extend the chronological benchmark to gradient boosting and additional baselines under a shared evaluation protocol.
    - Add durable model artifact storage, integrity verification on load and controlled rollback.
    - Add deeper monitoring, explanation stability and model-risk reporting.
    - Improve monitoring and model documentation.
    - Deploy a public demo and prepare a reproducible technical report.

## Recommended live demonstration order

1. Portfolio overview.
2. Application assessment with a synthetic example.
3. Change monthly income, loan amount and term. Explain how affordability features update.
4. Run assessment and inspect the probability estimate and feature contributions.
5. Batch analysis with a small synthetic CSV.
6. Model governance and subgroup diagnostics.
7. Open the GitHub repository and show tests, training workflow, Docker files and model card.

## Presentation discipline

- Do not quote metrics from old model artifacts as v2 results.
- Read final metrics from the generated v2 evaluation report.
- State the sample size and partition whenever presenting a metric.
- Do not describe synthetic patterns as empirical findings about Malawi.
- Do not claim fairness, regulatory approval or production readiness from these experiments.

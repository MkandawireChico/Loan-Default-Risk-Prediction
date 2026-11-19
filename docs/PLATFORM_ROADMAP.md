# Credit Risk Intelligence Platform: Delivery Roadmap

## Product direction

Evolve the existing Loan Default Risk Prediction repository into a complete, deployable credit-risk analytics and decision-support platform for a fictional Malawian commercial bank, Nyanja Commercial Bank.

The project is a portfolio and educational simulation. All borrowers, applications, branches, financial values, outcomes, and institutions are synthetic. It must never be presented as evidence about a real bank or used to make real lending decisions.

## What already exists

- A reproducible synthetic dataset of approximately 20,000 loan applications with deliberately injected data-quality issues.
- A documented cleaning and feature-engineering workflow.
- Logistic Regression, Random Forest, and XGBoost pipelines.
- Model evaluation, threshold analysis, probability calibration, SHAP utilities, and initial fairness analysis.
- A Streamlit dashboard, a model card, a notebook, command-line training scripts, and basic CI tests.

These assets are the starting point. We will audit and improve them rather than discard working research.

## Target product

A user-facing platform with four connected workspaces:

1. Portfolio Intelligence
   - Portfolio size, exposure, default rates, risk mix, and trends.
   - Drill-downs by branch, product, loan purpose, customer segment, and time.
   - Cohort and vintage analysis once the synthetic data generator supports meaningful dates.

2. Application Assessment
   - Structured applicant and loan inputs.
   - Probability of default, risk band, uncertainty/context, and human-readable explanation.
   - What-if analysis for affordability and loan terms.
   - No automatic approve/reject decisions. Every result is decision support only.

3. Model Risk and Governance
   - Model performance and calibration.
   - Threshold and cost trade-off analysis.
   - Segment-level performance and fairness diagnostics.
   - Data quality, drift, model version, training date, and limitations.
   - Audit trail for assessment requests, without storing unnecessary personal data.

4. Batch Analysis and Reporting
   - Upload a synthetic or user-provided demonstration CSV.
   - Validate schema and data quality before scoring.
   - Download scored results and a concise portfolio report.
   - Clear warnings about the synthetic model's lack of real-world validity.

## Target technical architecture

- Frontend: React + TypeScript, responsive and accessible.
- API: FastAPI with typed request/response schemas and OpenAPI documentation.
- ML: scikit-learn pipelines, calibrated probabilities, model evaluation and explainability modules.
- Storage: PostgreSQL for demonstration application records, model metadata, and audit events.
- Data science: Python, Pandas, NumPy, notebooks, reproducible generator and experiment outputs.
- Delivery: Docker Compose for local development; automated tests and linting in GitHub Actions; public deployment after security, reliability, and data-disclaimer checks.

The existing Streamlit dashboard remains available during the transition as a research/analysis interface. We will not remove it until the replacement workspaces have equivalent functionality.

## Research and modelling standards

- Keep the existing cross-sectional dataset as a reproducible baseline.
- [x] Add a separate synthetic chronological scenario with application dates and a documented macro-stress mechanism. This is a methodological simulation, not observed temporal validation.
- Separate training, validation, and final test data. Keep threshold selection and calibration away from the final test set.
- [x] Compare a prior-only baseline, Logistic Regression and Random Forest under the same synthetic chronological protocol.
- [ ] Add gradient boosting to the shared chronological benchmark.
- Report ROC-AUC and PR-AUC alongside default-class precision, recall, F1, specificity, confusion matrices, Brier score, log loss, and calibration plots.
- Select thresholds through explicit false-positive/false-negative cost scenarios and operational review capacity, not an arbitrary universal cutoff.
- Assess subgroup sample sizes, uncertainty intervals, calibration, error rates, and proxy-variable risks. Do not treat one fairness statistic as proof of fairness.
- Add data and prediction drift monitoring only where the simulated data supports a defensible reference distribution.
- Record seeds, package versions, dataset version, model version, and experiment configuration.

## Delivery phases

### Phase 0: Repository audit and foundations
- Verify current code, dependencies, tests, data-generation reproducibility, and model artifacts.
- Establish the platform architecture, coding conventions, and development branch.
- Identify existing defects before adding features.

### Phase 1: Reliable ML core
- Refactor training and inference into stable reusable interfaces.
- Add strict input validation and schema contracts.
- [x] Add temporal synthetic data only with a documented generation mechanism.
- [x] Add robust evaluation, calibration, threshold selection, bootstrap metric intervals, calibration-bin diagnostics and explainability tests.
- Produce versioned model metadata and reproducible reports.

### Phase 2: API and persistence
- [x] Build FastAPI endpoints for health, model metadata, single assessment, batch scoring, portfolio analytics and explanations.
- [x] Add PostgreSQL persistence for metadata-only assessment audit events.
- [x] Add request validation, structured errors and API tests.
- [x] Add formal Alembic migrations with upgrade/downgrade coverage.\n- [x] Add a lightweight model metadata registry with artifact/dataset fingerprints and version status.
- [ ] Add durable artifact archival, integrity verification on load, and controlled rollback.
- [x] Add a separate synthetic chronological stress-test experiment with whole-month splits, expanding-window calibration, validation-only threshold selection, and month-level future-test diagnostics.
- [x] Expose the temporal comparison report through the API and Model Governance workspace.

### Phase 3: Product interface
- [x] Build the React application with Portfolio Intelligence, Application Assessment, Model Governance, and Batch Analysis.
- [x] Add responsive layouts, loading/error states and accessible interactions.
- [x] Connect dashboard metrics and charts to API responses.

### Phase 4: Deployment and operations
- [x] Containerize frontend, API and PostgreSQL for local development.
- [x] Add CI checks, environment configuration, health checks and deployment documentation.
- [x] Verify the full Docker Compose stack through automated end-to-end smoke tests.
- [x] Add a Render Blueprint for API, static frontend and managed PostgreSQL.
- [ ] Provision hosting after reviewing service/database charges and account access.
- [ ] Verify the live application end to end.
- [ ] Publish a presentation-ready case study, architecture diagram, model card, limitations and demo guide.

## Definition of done

The project is complete when a reviewer can open the live application, inspect a clearly labelled synthetic portfolio, assess a sample application, run a batch demonstration, inspect model performance and governance information, reproduce the ML workflow from the repository, run automated tests, and understand every material limitation.

## Non-negotiable safeguards

- No real banking/customer data is required or implied.
- No claims of real-world predictive performance.
- No automated lending decisions or approval recommendations.
- No hidden or fabricated performance metrics.
- No deployment of secrets, credentials, or private data.
- Every public page and exported report must disclose the synthetic nature of the project.

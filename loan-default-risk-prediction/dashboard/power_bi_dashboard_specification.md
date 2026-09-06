# Power BI Dashboard Specification

**Status: not yet built.** This is a structured specification for building the dashboard in Power BI Desktop, written from the actual fields and results in `data/processed/loan_applications_clean.csv` and the model outputs in `notebooks/loan_default_risk_analysis.ipynb`. All data is synthetic — see the project README's disclaimer.

---

## 1. Data sources

- `data/processed/loan_applications_clean.csv` — the cleaned application-level dataset
- Model outputs (predicted probability, risk category) — export the test-set predictions from the notebook (`test_results` DataFrame, Section 12) to a CSV (`data/processed/scored_applications.csv`) for Page 4, since a fixed Power BI report can't retrain scikit-learn models itself

## 2. Data model

A single flat table is sufficient given this dataset's structure — one row per loan application, already denormalized. Add a **Risk_Category** lookup table (Low / Medium / High, with the 0.20/0.50 boundary values) so the boundary definitions are documented in the model itself, not just hard-coded into a filter.

## 3. Measures (DAX)

```dax
Total Applications = COUNTROWS(applications)

Default Rate = DIVIDE(SUM(applications[Default]), COUNTROWS(applications))

Average Loan Amount = AVERAGE(applications[Loan_Amount])

Average Income = AVERAGE(applications[Monthly_Income])

High Risk Borrowers = CALCULATE(COUNTROWS(scored), scored[Risk_Category] = "High Risk")

Average Probability of Default = AVERAGE(scored[Predicted_Probability])
```

**Why `DIVIDE()` instead of `/`:** returns `BLANK()` rather than erroring when a slicer selection produces a zero-row context — common once Segment/Branch/Purpose/Risk-Category slicers are added.

## 4. Pages

### Page 1 — Credit Risk Overview

KPI cards: Total Applications, Default Rate, Average Loan Amount, Average Income, High-Risk Borrower Count.
Visuals: default rate trend (if a date field is added in a future version — this synthetic dataset is cross-sectional, not time-stamped, so a genuine trend isn't available yet; see Limitations), default rate by customer segment (bar), default rate by branch (bar — with a note that branch names are real Malawian towns used as labels only, and the notebook found no meaningful branch effect in this data).

### Page 2 — Borrower and Loan Profile

Income distribution (histogram), loan amount distribution (histogram), loan purpose (bar), employment status (bar), customer segment (donut), loan term distribution (bar), credit score distribution (histogram), DTI distribution (histogram).

### Page 3 — Default Drivers

Default rate by DTI quartile/category (bar — should show the strong monotonic pattern found in the notebook: 6.3% → 12.4% → 17.6% → 32.7% across quartiles), default rate by credit score category (bar — Poor/Fair/Good/Very Good), default rate by previous-default count (bar), default rate by late-payment frequency band, a table of the top model features (from `coef_df` / `importance_df` in the notebook, exported to CSV), high-risk segment summary.

### Page 4 — Predicted Credit Risk

Low/Medium/High risk distribution (donut), predicted probability distribution (histogram, matching the notebook's Section 12 chart), a high-risk borrower table (filterable), average probability of default (card), risk by branch (bar), risk by customer segment (bar). Filters: Customer Segment, Branch, Loan Purpose, Risk Category.

## 5. A finding that belongs directly on the dashboard, not just the README

**The risk segmentation actually works — and the dashboard should prove it, not just assert it.** The notebook found actual default rates of 3.0% (Low Risk), 11.0% (Medium Risk), and 37.0% (High Risk) in the test set — a clean, monotonic validation that the model's risk categories mean what they claim. Page 4 should show this validation explicitly (e.g. a bar chart of actual default rate by predicted risk category) rather than only showing the *predicted* distribution — a credit team evaluating whether to trust this scoring system needs to see it checked against real outcomes, not just presented.

## 6. Build order

1. Import both data sources, build the Risk_Category lookup table
2. Write the core measures (Section 3)
3. Build Page 1's KPI cards and segment/branch charts — sense-check every number against the notebook's printed output
4. Build Pages 2–3
5. Build Page 4, including the Section 5 validation chart
6. Add slicers last
7. Screenshot into a `visuals/dashboard/` folder and link from the main README

"""
Generates the synthetic loan application dataset for the Loan Default Risk
Prediction portfolio project.

ALL DATA IS ENTIRELY SYNTHETIC. This script generates a fictional dataset
for a fictional bank ("Nyanja Commercial Bank" — verified not to match any
real Malawian commercial bank) for educational and portfolio purposes only.
No real customers, loans, or banking data are used or represented.

Run:
    python generate_data.py
Outputs:
    ../data/raw/loan_applications_synthetic.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 20000

BRANCHES = ["Blantyre", "Lilongwe", "Mzuzu", "Zomba", "Mangochi",
            "Kasungu", "Salima", "Dedza", "Rumphi", "Mulanje"]
BRANCH_WEIGHTS = [0.22, 0.22, 0.12, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]

SEGMENTS = ["Salaried Mass Market", "Salaried Premium", "SME", "Self-Employed", "Youth/Student"]
SEGMENT_WEIGHTS = [0.35, 0.12, 0.20, 0.25, 0.08]

LOAN_PURPOSES = ["Personal", "Business/SME", "Agriculture", "Education",
                  "Home Improvement", "Asset Finance", "Emergency"]
LOAN_PURPOSE_WEIGHTS = [0.30, 0.20, 0.12, 0.08, 0.12, 0.10, 0.08]
EDUCATION = ["Primary", "Secondary", "Tertiary/College", "University"]
COLLATERAL_TYPES = ["None", "Vehicle", "Land/Title Deed", "Household Goods", "Salary Guarantee"]
CHANNELS = ["Branch", "Mobile Banking App", "Agent Banking", "Online Portal"]

OCCUPATIONS_BY_SEGMENT = {
    "Salaried Mass Market": ["Teacher", "Nurse", "Civil Servant", "Clerk", "Police Officer"],
    "Salaried Premium": ["Bank Officer", "Engineer", "Accountant", "Manager", "IT Professional"],
    "SME": ["Shop Owner", "Wholesaler", "Trader", "Restaurant Owner", "Hardware Dealer"],
    "Self-Employed": ["Farmer", "Tailor", "Mechanic", "Driver", "Freelance Trader"],
    "Youth/Student": ["Student", "Intern", "Part-time Worker", "Recent Graduate"],
}

def employment_for_segment(seg):
    if seg == "Salaried Mass Market":
        return np.random.choice(["Employed - Government", "Employed - Private"], p=[0.5, 0.5])
    if seg == "Salaried Premium":
        return np.random.choice(["Employed - Government", "Employed - Private"], p=[0.35, 0.65])
    if seg == "SME":
        return "Business Owner"
    if seg == "Self-Employed":
        return "Self-Employed"
    return np.random.choice(["Employed - Private", "Unemployed"], p=[0.55, 0.45])


def generate_clean_dataset():
    # --- Demographics ---
    age = np.clip(np.random.normal(38, 11, N), 20, 70).round().astype(int)
    gender = np.random.choice(["Male", "Female"], N, p=[0.54, 0.46])
    marital_status = np.random.choice(["Single", "Married", "Divorced", "Widowed"], N, p=[0.30, 0.55, 0.10, 0.05])
    education = np.random.choice(EDUCATION, N, p=[0.10, 0.35, 0.30, 0.25])
    segment = np.random.choice(SEGMENTS, N, p=SEGMENT_WEIGHTS)
    employment_status = np.array([employment_for_segment(s) for s in segment])
    occupation = np.array([np.random.choice(OCCUPATIONS_BY_SEGMENT[s]) for s in segment])
    branch = np.random.choice(BRANCHES, N, p=BRANCH_WEIGHTS)
    loan_purpose = np.random.choice(LOAN_PURPOSES, N, p=LOAN_PURPOSE_WEIGHTS)

    # --- Income ---
    income_base = {
        "Salaried Mass Market": 280000, "Salaried Premium": 950000, "SME": 550000,
        "Self-Employed": 320000, "Youth/Student": 150000,
    }
    monthly_income = np.array([
        max(60000, np.random.lognormal(np.log(income_base[s]), 0.45)) for s in segment
    ]).round(-2)

    account_tenure_months = np.clip(np.random.exponential(36, N), 1, 240).round().astype(int)

    # --- Credit history ---
    credit_history_length_months = np.clip(
        account_tenure_months * np.random.uniform(0.6, 1.3, N), 0, 300
    ).round().astype(int)
    previous_loan_count = np.random.poisson(1.8, N)
    previous_default_count = np.random.binomial(np.clip(previous_loan_count, 0, 5), 0.08, N)
    number_of_late_payments = np.random.poisson(1.2 + previous_default_count * 1.5, N)
    days_past_due_history = (number_of_late_payments * np.random.uniform(3, 15, N)).round().astype(int)

    # --- Loan characteristics ---
    loan_amount = np.clip(monthly_income * np.random.uniform(1.5, 8, N), 50000, 15_000_000).round(-3)
    loan_term_months = np.random.choice([6, 12, 18, 24, 36, 48, 60], N,
                                         p=[0.10, 0.25, 0.15, 0.20, 0.15, 0.10, 0.05])
    interest_rate = np.round(np.random.uniform(18, 35, N), 2)
    monthly_rate = interest_rate / 100 / 12
    monthly_repayment = (loan_amount * monthly_rate * (1 + monthly_rate) ** loan_term_months) / \
                         ((1 + monthly_rate) ** loan_term_months - 1)

    existing_loan_count = np.random.poisson(0.7, N)
    existing_debt = (existing_loan_count * monthly_income * np.random.uniform(1.0, 3.5, N)).round(-2)

    savings_balance = np.clip(
        monthly_income * np.random.lognormal(-0.3, 0.9, N) * np.clip(1 - previous_default_count * 0.15, 0.2, 1),
        0, None
    ).round(-2)
    average_monthly_balance = (savings_balance * np.random.uniform(0.3, 0.9, N)).round(-2)
    average_monthly_transactions = np.clip(np.random.poisson(12, N), 0, 60)

    has_credit_card = np.random.binomial(1, np.where(segment == "Salaried Premium", 0.6, 0.15))
    has_mobile_banking = np.random.binomial(1, 0.72, N)

    collateral_type = np.array([
        np.random.choice(COLLATERAL_TYPES, p=[0.35, 0.20, 0.15, 0.15, 0.15]) if la > 500000 else "None"
        for la in loan_amount
    ])
    collateral_value = np.where(
        collateral_type == "None", 0,
        (loan_amount * np.random.uniform(0.8, 1.5, N)).round(-3)
    )
    loan_application_channel = np.random.choice(CHANNELS, N, p=[0.40, 0.35, 0.15, 0.10])

    # --- Ratios ---
    debt_to_income_ratio = np.round((existing_debt + loan_amount * 0.1) / (monthly_income * 12), 3)
    loan_to_income_ratio = np.round(loan_amount / (monthly_income * 12), 3)
    monthly_repayment_ratio = np.round(monthly_repayment / monthly_income, 3)

    # --- Credit score ---
    credit_score_raw = (
        650
        - previous_default_count * 45
        - number_of_late_payments * 8
        - debt_to_income_ratio * 60
        + np.log1p(savings_balance) * 8
        + np.log1p(account_tenure_months) * 6
        + np.random.normal(0, 35, N)
    )
    credit_score = np.clip(credit_score_raw, 300, 850).round().astype(int)

    # --- Target: Default, via logistic risk score + noise ---
    z = (
        -2.1
        + 2.6 * debt_to_income_ratio
        + 1.4 * loan_to_income_ratio
        + 0.9 * previous_default_count
        + 0.18 * number_of_late_payments
        + 0.010 * days_past_due_history
        + 1.1 * monthly_repayment_ratio
        + 0.5 * existing_loan_count
        - 0.35 * np.log1p(savings_balance / 10000)
        - 0.30 * np.log1p(account_tenure_months)
        - 0.004 * (credit_score - 650) / 10
        - 0.15 * (employment_status == "Employed - Government").astype(int)
        - 0.10 * (employment_status == "Employed - Private").astype(int)
        + 0.35 * (employment_status == "Unemployed").astype(int)
        + 0.15 * (segment == "Youth/Student").astype(int)
        - 0.10 * (segment == "Salaried Premium").astype(int)
        + np.random.normal(0, 0.55, N)
    )
    prob_default = 1 / (1 + np.exp(-z))
    default = np.random.binomial(1, prob_default)

    df = pd.DataFrame({
        "Customer_ID": [f"CUST-{i:07d}" for i in range(1, N + 1)],
        "Loan_ID": [f"LN-{i:07d}" for i in range(1, N + 1)],
        "Age": age, "Gender": gender, "Marital_Status": marital_status,
        "Education_Level": education, "Employment_Status": employment_status,
        "Occupation": occupation, "Monthly_Income": monthly_income,
        "Account_Tenure_Months": account_tenure_months, "Branch_Location": branch,
        "Customer_Segment": segment, "Loan_Purpose": loan_purpose,
        "Loan_Amount": loan_amount, "Loan_Term_Months": loan_term_months,
        "Interest_Rate": interest_rate, "Monthly_Repayment": monthly_repayment.round(2),
        "Existing_Loan_Count": existing_loan_count, "Existing_Debt": existing_debt,
        "Savings_Balance": savings_balance, "Average_Monthly_Transactions": average_monthly_transactions,
        "Average_Monthly_Balance": average_monthly_balance,
        "Credit_History_Length_Months": credit_history_length_months,
        "Previous_Loan_Count": previous_loan_count, "Previous_Default_Count": previous_default_count,
        "Number_of_Late_Payments": number_of_late_payments, "Days_Past_Due_History": days_past_due_history,
        "Debt_to_Income_Ratio": debt_to_income_ratio, "Loan_to_Income_Ratio": loan_to_income_ratio,
        "Has_Credit_Card": has_credit_card, "Has_Mobile_Banking": has_mobile_banking,
        "Collateral_Value": collateral_value, "Collateral_Type": collateral_type,
        "Loan_Application_Channel": loan_application_channel, "Credit_Score": credit_score,
        "Default": default,
    })
    return df


def inject_data_quality_issues(df):
    df = df.copy()

    # 1. Missing income values (~2%)
    idx = df.sample(frac=0.02, random_state=10).index
    df.loc[idx, "Monthly_Income"] = np.nan

    # 2. Missing credit scores (~2%)
    idx = df.sample(frac=0.02, random_state=11).index
    df.loc[idx, "Credit_Score"] = np.nan

    # 3. Missing employment status (~1.5%)
    idx = df.sample(frac=0.015, random_state=12).index
    df.loc[idx, "Employment_Status"] = np.nan

    # 4. Extreme income outliers (~0.3%)
    idx = df.sample(frac=0.003, random_state=20).index
    df.loc[idx, "Monthly_Income"] = df.loc[idx, "Monthly_Income"] * np.random.uniform(10, 20, len(idx))

    # 5. Extreme loan amounts (~0.3%)
    idx = df.sample(frac=0.003, random_state=21).index
    df.loc[idx, "Loan_Amount"] = df.loc[idx, "Loan_Amount"] * np.random.uniform(5, 10, len(idx))

    # 6. Inconsistent employment category names (~1.5%)
    idx = df.sample(frac=0.015, random_state=30).index
    variant_map = {
        "Employed - Government": "employed - government",
        "Employed - Private": "Employed-Private",
        "Self-Employed": "SELF EMPLOYED",
        "Business Owner": "Buisness Owner",
        "Unemployed": "unemployed ",
    }
    for i in idx:
        original = df.loc[i, "Employment_Status"]
        if pd.notna(original) and original in variant_map:
            df.loc[i, "Employment_Status"] = variant_map[original]

    # 7. Duplicate records (~1%)
    dup_idx = df.sample(frac=0.01, random_state=40).index
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # 8. Invalid / impossible ages (~0.2%)
    idx = df.sample(frac=0.002, random_state=50).index
    df.loc[idx, "Age"] = np.random.choice([12, 14, 130, 145, -5], len(idx))

    # 9. Missing branch names (~1%)
    idx = df.sample(frac=0.01, random_state=60).index
    df.loc[idx, "Branch_Location"] = np.nan

    # 10. Inconsistent loan-purpose labels (~1.5%)
    idx = df.sample(frac=0.015, random_state=70).index
    purpose_variant_map = {
        "Personal": "personal ",
        "Business/SME": "biz/sme",
        "Home Improvement": "home improvement",
        "Agriculture": "AGRICULTURE",
    }
    for i in idx:
        original = df.loc[i, "Loan_Purpose"]
        if original in purpose_variant_map:
            df.loc[i, "Loan_Purpose"] = purpose_variant_map[original]

    return df


def main():
    df_clean = generate_clean_dataset()
    print(f"Clean dataset generated: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")
    print(f"Overall default rate (before quality-issue injection): {df_clean['Default'].mean():.2%}")

    df_raw = inject_data_quality_issues(df_clean)
    print(f"Raw dataset after injecting quality issues: {df_raw.shape[0]:,} rows")

    out_dir = Path("../data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loan_applications_synthetic.csv"
    df_raw.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

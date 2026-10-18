"""
Streamlit app for loan default risk prediction dashboard.

This interactive dashboard provides model predictions, risk analysis,
and portfolio insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.loading import load_cleaned_data, load_model
from src.data.preprocessing import prepare_modeling_data
from src.evaluation.metrics import comprehensive_evaluation

st.set_page_config(
    page_title="Loan Default Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Default Risk Prediction Dashboard")
st.markdown("""
**Nyanja Commercial Bank (Fictional) - Credit Risk Analytics**

This dashboard provides insights into loan default risk predictions using machine learning models.
**Note:** This uses entirely synthetic data for educational purposes only.
""")

# Sidebar
st.sidebar.header("Configuration")

# Load data
@st.cache_data
def load_data():
    return load_cleaned_data()

@st.cache_resource
def load_trained_model():
    model_path = Path("models") / "logistic_pipeline.joblib"
    if model_path.exists():
        return load_model(model_path)
    return None

df = load_data()
model = load_trained_model()

if model is None:
    st.error("No trained model found. Please train a model first using `python scripts/train_model.py`")
    st.stop()

# Prepare data
X, y = prepare_modeling_data(df, target_col="Default")

# Sidebar options
st.sidebar.subheader("Analysis Options")
show_predictions = st.sidebar.checkbox("Show Model Predictions", value=True)
show_risk_analysis = st.sidebar.checkbox("Show Risk Analysis", value=True)
show_portfolio = st.sidebar.checkbox("Show Portfolio Overview", value=True)

# Main content
if show_portfolio:
    st.header("Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Applications", f"{len(df):,}")
    
    with col2:
        st.metric("Default Rate", f"{df['Default'].mean():.2%}")
    
    with col3:
        st.metric("Total Portfolio", f"MWK {df['Loan_Amount'].sum():,.0f}")
    
    with col4:
        st.metric("Avg Loan Amount", f"MWK {df['Loan_Amount'].mean():,.0f}")
    
    # Default rate by segment
    st.subheader("Default Rate by Customer Segment")
    segment_defaults = df.groupby("Customer_Segment")["Default"].agg(["mean", "count"]).reset_index()
    segment_defaults.columns = ["Segment", "Default Rate", "Count"]
    segment_defaults["Default Rate"] = segment_defaults["Default Rate"] * 100
    
    fig_segment = px.bar(
        segment_defaults,
        x="Segment",
        y="Default Rate",
        color="Count",
        title="Default Rate by Segment",
        labels={"Default Rate": "Default Rate (%)", "Count": "Application Count"}
    )
    st.plotly_chart(fig_segment, use_container_width=True)
    
    # Default rate by credit category
    st.subheader("Default Rate by Credit Score Category")
    credit_defaults = df.groupby("Credit_Risk_Category")["Default"].agg(["mean", "count"]).reset_index()
    credit_defaults.columns = ["Credit Category", "Default Rate", "Count"]
    credit_defaults["Default Rate"] = credit_defaults["Default Rate"] * 100
    credit_defaults = credit_defaults.sort_values("Default Rate", ascending=False)
    
    fig_credit = px.bar(
        credit_defaults,
        x="Credit Category",
        y="Default Rate",
        title="Default Rate by Credit Category",
        labels={"Default Rate": "Default Rate (%)"},
        category_orders={"Credit Category": ["Poor", "Fair", "Good", "Very Good"]}
    )
    st.plotly_chart(fig_credit, use_container_width=True)

if show_risk_analysis:
    st.header("Risk Analysis")
    
    # Risk distribution
    st.subheader("Risk Score Distribution")
    fig_hist = px.histogram(
        df,
        x="Credit_Score",
        color="Default",
        nbins=50,
        title="Credit Score Distribution by Default Status",
        labels={"Credit_Score": "Credit Score", "Default": "Default Status"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # DTI vs Default
    st.subheader("Debt-to-Income Ratio vs Default")
    fig_dti = px.box(
        df,
        x="Default",
        y="Debt_to_Income_Ratio",
        title="Debt-to-Income Ratio by Default Status",
        labels={"Default": "Default Status", "Debt_to_Income_Ratio": "DTI Ratio"}
    )
    st.plotly_chart(fig_dti, use_container_width=True)
    
    # Loan burden analysis
    st.subheader("Loan Burden Category Analysis")
    burden_defaults = df.groupby("Loan_Burden_Category")["Default"].agg(["mean", "count"]).reset_index()
    burden_defaults.columns = ["Burden Category", "Default Rate", "Count"]
    burden_defaults["Default Rate"] = burden_defaults["Default Rate"] * 100
    
    order = ["Low", "Moderate", "High", "Severe", "Unknown"]
    burden_defaults["Burden Category"] = pd.Categorical(
        burden_defaults["Burden Category"], 
        categories=order, 
        ordered=True
    )
    burden_defaults = burden_defaults.sort_values("Burden Category")
    
    fig_burden = px.bar(
        burden_defaults,
        x="Burden Category",
        y="Default Rate",
        title="Default Rate by Loan Burden Category",
        labels={"Default Rate": "Default Rate (%)"}
    )
    st.plotly_chart(fig_burden, use_container_width=True)

if show_predictions:
    st.header("Model Predictions")
    
    # Generate predictions for a sample
    st.subheader("Sample Predictions")
    
    sample_size = st.slider("Sample Size", min_value=10, max_value=100, value=20)
    sample_df = df.sample(n=sample_size, random_state=42)
    
    X_sample, _ = prepare_modeling_data(sample_df, target_col="Default")
    y_prob = model.predict_proba(X_sample)[:, 1]
    
    sample_df["Predicted_Probability"] = y_prob
    sample_df["Risk_Level"] = pd.cut(
        sample_df["Predicted_Probability"],
        bins=[0, 0.2, 0.5, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    
    display_cols = [
        "Customer_ID",
        "Loan_ID",
        "Credit_Score",
        "Debt_to_Income_Ratio",
        "Predicted_Probability",
        "Risk_Level",
        "Default"
    ]
    
    st.dataframe(sample_df[display_cols].sort_values("Predicted_Probability", ascending=False))
    
    # Prediction distribution
    st.subheader("Prediction Distribution")
    fig_pred = px.histogram(
        sample_df,
        x="Predicted_Probability",
        color="Default",
        nbins=20,
        title="Predicted Probability Distribution",
        labels={"Predicted_Probability": "Predicted Default Probability"}
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer:** This dashboard uses synthetic data created for educational purposes. 
The bank, customers, loan records, branches, and model results do not represent any real commercial bank or actual banking data.
""")

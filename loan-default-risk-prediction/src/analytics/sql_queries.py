"""
SQL queries for loan default risk analytics.

This module provides reusable SQL queries for portfolio analysis,
risk segmentation, and reporting.
"""

SQL_QUERIES = {
    "overall_portfolio_summary": """
        SELECT 
            COUNT(*) as total_applications,
            SUM(CASE WHEN Default = 1 THEN 1 ELSE 0 END) as total_defaults,
            ROUND(AVG(Default), 4) as default_rate,
            ROUND(AVG(Loan_Amount), 2) as avg_loan_amount,
            ROUND(SUM(Loan_Amount), 2) as total_loan_portfolio,
            ROUND(AVG(Monthly_Income), 2) as avg_monthly_income,
            ROUND(AVG(Debt_to_Income_Ratio), 4) as avg_dti,
            ROUND(AVG(Credit_Score), 2) as avg_credit_score
        FROM loan_applications_clean
    """,
    
    "default_rate_by_segment": """
        SELECT 
            Customer_Segment,
            COUNT(*) as applications,
            SUM(CASE WHEN Default = 1 THEN 1 ELSE 0 END) as defaults,
            ROUND(AVG(Default), 4) as default_rate,
            ROUND(AVG(Loan_Amount), 2) as avg_loan_amount,
            ROUND(AVG(Monthly_Income), 2) as avg_income
        FROM loan_applications_clean
        GROUP BY Customer_Segment
        ORDER BY default_rate DESC
    """,
    
    "default_rate_by_branch": """
        SELECT 
            Branch_Location,
            COUNT(*) as applications,
            SUM(CASE WHEN Default = 1 THEN 1 ELSE 0 END) as defaults,
            ROUND(AVG(Default), 4) as default_rate,
            ROUND(SUM(Loan_Amount), 2) as total_loan_amount
        FROM loan_applications_clean
        GROUP BY Branch_Location
        ORDER BY default_rate DESC
    """,
    
    "default_rate_by_employment_status": """
        SELECT 
            Employment_Status,
            COUNT(*) as applications,
            SUM(CASE WHEN Default = 1 THEN 1 ELSE 0 END) as defaults,
            ROUND(AVG(Default), 4) as default_rate,
            ROUND(AVG(Monthly_Income), 2) as avg_income
        FROM loan_applications_clean
        GROUP BY Employment_Status
        ORDER BY default_rate DESC
    """,
    
    "default_rate_by_credit_category": """
        SELECT 
            Credit_Risk_Category,
            COUNT(*) as applications,
            SUM(CASE WHEN Default = 1 THEN 1 ELSE 0 END) as defaults,
            ROUND(AVG(Default), 4) as default_rate,
            ROUND(AVG(Credit_Score), 2) as avg_credit_score
        FROM loan_applications_clean
        GROUP BY Credit_Risk_Category
        ORDER BY default_rate DESC
    """,
    
    "default_rate_by_loan_burden": """
        SELECT 
            Loan_Burden_Category,
            COUNT(*) as applications,
            SUM(CASE WHEN Default = 1 THEN 1 ELSE 0 END) as defaults,
            ROUND(AVG(Default), 4) as default_rate,
            ROUND(AVG(Monthly_Repayment_Ratio), 4) as avg_repayment_ratio
        FROM loan_applications_clean
        GROUP BY Loan_Burden_Category
        ORDER BY 
            CASE Loan_Burden_Category
                WHEN 'Low' THEN 1
                WHEN 'Moderate' THEN 2
                WHEN 'High' THEN 3
                WHEN 'Severe' THEN 4
                ELSE 5
            END
    """,
    
    "risk_segmentation": """
        SELECT 
            CASE 
                WHEN Credit_Score >= 740 THEN 'Very Good'
                WHEN Credit_Score >= 670 THEN 'Good'
                WHEN Credit_Score >= 580 THEN 'Fair'
                ELSE 'Poor'
            END as credit_band,
            CASE 
                WHEN Debt_to_Income_Ratio < 0.20 THEN 'Low DTI'
                WHEN Debt_to_Income_Ratio < 0.35 THEN 'Medium DTI'
                ELSE 'High DTI'
            END as dti_band,
            COUNT(*) as applications,
            SUM(CASE WHEN Default = 1 THEN 1 ELSE 0 END) as defaults,
            ROUND(AVG(Default), 4) as default_rate
        FROM loan_applications_clean
        GROUP BY credit_band, dti_band
        ORDER BY default_rate DESC
    """,
    
    "loan_purpose_analysis": """
        SELECT 
            Loan_Purpose,
            COUNT(*) as applications,
            SUM(CASE WHEN Default = 1 THEN 1 ELSE 0 END) as defaults,
            ROUND(AVG(Default), 4) as default_rate,
            ROUND(AVG(Loan_Amount), 2) as avg_loan_amount,
            ROUND(AVG(Loan_Term_Months), 2) as avg_term_months
        FROM loan_applications_clean
        GROUP BY Loan_Purpose
        ORDER BY default_rate DESC
    """,
    
    "high_risk_applicants": """
        SELECT 
            Customer_ID,
            Loan_ID,
            Monthly_Income,
            Loan_Amount,
            Debt_to_Income_Ratio,
            Credit_Score,
            Previous_Default_Count,
            Number_of_Late_Payments,
            Credit_Risk_Category,
            Loan_Burden_Category
        FROM loan_applications_clean
        WHERE 
            Credit_Risk_Category IN ('Poor', 'Fair')
            OR Loan_Burden_Category IN ('High', 'Severe')
            OR Previous_Default_Count > 0
        ORDER BY Debt_to_Income_Ratio DESC
        LIMIT 100
    """,
    
    "portfolio_exposure_by_risk": """
        SELECT 
            CASE 
                WHEN Credit_Score >= 740 THEN 'Low Risk'
                WHEN Credit_Score >= 670 THEN 'Medium-Low Risk'
                WHEN Credit_Score >= 580 THEN 'Medium-High Risk'
                ELSE 'High Risk'
            END as risk_category,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage,
            ROUND(SUM(Loan_Amount), 2) as total_exposure_pct,
            ROUND(AVG(Default), 4) as default_rate
        FROM loan_applications_clean
        GROUP BY risk_category
        ORDER BY 
            CASE risk_category
                WHEN 'Low Risk' THEN 1
                WHEN 'Medium-Low Risk' THEN 2
                WHEN 'Medium-High Risk' THEN 3
                WHEN 'High Risk' THEN 4
            END
    """
}


def get_query(query_name: str) -> str:
    """
    Get a SQL query by name.
    
    Args:
        query_name: Name of the query
        
    Returns:
        SQL query string
    """
    if query_name not in SQL_QUERIES:
        raise ValueError(f"Query '{query_name}' not found. Available: {list(SQL_QUERIES.keys())}")
    
    return SQL_QUERIES[query_name]


def list_queries() -> list:
    """Return list of available query names."""
    return list(SQL_QUERIES.keys())

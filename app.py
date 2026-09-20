"""
Streamlit app for the loan approval prediction project.

The app loads the training data, engineers income and debt features,
trains a Random Forest pipeline, and exposes model diagnostics and an
interactive scoring form.
"""

from typing import Tuple, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Assessment Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58A6FF;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA INGESTION & FEATURE ENGINEERING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_engineer_features(filepath: str = "data/train_loan_data.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        raw_data = pd.read_csv(filepath)
    except FileNotFoundError:
        # Fallback if run from the wrong directory
        raw_data = pd.read_csv("train_loan_data.csv")

    df = raw_data.copy()

    if 'Loan_ID' in df.columns:
        df.drop(columns=['Loan_ID'], inplace=True)

    # Add income and debt features before model training.
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['EMI_Estimate'] = (df['LoanAmount'] * 1000) / df['Loan_Amount_Term'].replace(0, np.nan)
    df['Debt_To_Income'] = df['EMI_Estimate'] / (df['Total_Income'] + 1e-5)

    # Categorical Encoding
    encoded_df = df.copy()
    encoding_map = {
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Loan_Status': {'Y': 1, 'N': 0}
    }
    for col, mapping in encoding_map.items():
        if col in encoded_df.columns:
            encoded_df[col] = encoded_df[col].map(mapping)

    return df, encoded_df

cleaned_df, encoded_df = load_and_engineer_features()

# -----------------------------------------------------------------------------
# 3. LEAKAGE-FREE MACHINE LEARNING PIPELINE
# -----------------------------------------------------------------------------
FEATURE_COLUMNS = [
    'Credit_History',
    'Total_Income',
    'LoanAmount',
    'Loan_Amount_Term',
    'Debt_To_Income',
    'EMI_Estimate',
    'Property_Area',
    'Education',
    'Married'
]

@st.cache_resource(show_spinner=False)
def train_leakage_free_model(data: pd.DataFrame) -> Dict[str, Any]:
    # Drop rows where target is NaN
    valid_data = data.dropna(subset=['Loan_Status']).copy()
    X = valid_data[FEATURE_COLUMNS]
    y = valid_data['Loan_Status'].astype(int)

    # Split before fitting the preprocessing pipeline.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Keep imputation inside the pipeline so it is fitted on training data only.
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_split=6,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    # Estimate validation accuracy using five stratified folds.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

    # Fit pipeline on training cohort
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    rf_model = pipeline.named_steps['classifier']
    feature_importances = pd.Series(
        rf_model.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    return {
        'pipeline': pipeline,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'conf_matrix': confusion_matrix(y_test, y_pred),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'importances': feature_importances
    }

model_artifacts = train_leakage_free_model(encoded_df)

# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/IBM-SkillsBuild-052FAD?style=for-the-badge&logo=ibm&logoColor=white")
    st.markdown("### System Controls")
    nav = st.radio(
        "Module Navigation:",
        [
            "1. Executive Overview & Data Health",
            "2. Exploratory Data Analysis (EDA)",
            "3. Model Architecture & Diagnostics",
            "4. Automated Underwriting Simulator",
            "5. Algorithmic Fairness & UN SDG 10"
        ]
    )
    st.markdown("---")
    st.caption("**Candidate:** Srijan Das")
    st.caption("**Internship ID:** `IBMUEDA0483`")
    st.caption("**Institution:** CGEC (CSE)")

# -----------------------------------------------------------------------------
# 5. MODULE 1: EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
if nav == "1. Executive Overview & Data Health":
    st.title("🏦 Credit Risk Assessment & Automated Loan Underwriting Engine")
    st.markdown("#### Dataset summary and model inputs")
    st.write("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Audited Applicants</div>
            <div class="metric-value">{len(cleaned_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        approval_rate = (cleaned_df['Loan_Status'] == 'Y').mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Historical Approval Rate</div>
            <div class="metric-value">{approval_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        avg_income = int(cleaned_df['Total_Income'].dropna().mean())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Household Income</div>
            <div class="metric-value">₹{avg_income:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Pipeline Architecture</div>
            <div class="metric-value">Leakage-Free</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Sample of loaded records")
    st.dataframe(cleaned_df.head(6), use_container_width=True)

    with st.expander("Descriptive statistics", expanded=True):
        st.dataframe(cleaned_df.describe().T, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. MODULE 2: EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
elif nav == "2. Exploratory Data Analysis (EDA)":
    st.title("📈 Exploratory analysis")
    st.markdown("Evaluating demographic and financial relationships governing loan approval outcomes.")
    st.write("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### 1. Credit History as Primary Default Barrier")
        fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
        sns.countplot(
            data=cleaned_df.dropna(subset=['Credit_History', 'Loan_Status']),
            x='Credit_History',
            hue='Loan_Status',
            palette=["#D32F2F", "#1976D2"],
            ax=ax
        )
        ax.set_xticklabels(["0.0 (Prior Default / High Risk)", "1.0 (Clean Credit History)"])
        ax.set_ylabel("Applicant Count")
        ax.set_xlabel("")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        st.pyplot(fig)
        st.caption("Applicants lacking a clean credit record face an approval rate of < 10%, isolating credit history as the primary underwriting hurdle.")

    with col_right:
        st.markdown("##### 2. Capital Requested vs. Household Income")
        fig2, ax2 = plt.subplots(figsize=(6, 3.8), dpi=150)
        sns.scatterplot(
            data=cleaned_df,
            x='Total_Income',
            y='LoanAmount',
            hue='Loan_Status',
            palette=["#D32F2F", "#1976D2"],
            alpha=0.75,
            ax=ax2
        )
        ax2.set_xlim(0, 30000)
        ax2.set_xlabel("Total Household Income (₹)")
        ax2.set_ylabel("Loan Capital (₹ Thousands)")
        ax2.grid(linestyle='--', alpha=0.3)
        st.pyplot(fig2)
        st.caption("Risk concentrates when high loan amounts are requested with sub-₹5,000 monthly income.")

    st.markdown("##### 3. Property Location Demographics")
    fig3, ax3 = plt.subplots(figsize=(10, 2.8), dpi=150)
    sns.countplot(
        data=cleaned_df.dropna(subset=['Property_Area', 'Loan_Status']),
        x='Property_Area',
        hue='Loan_Status',
        palette=["#E57373", "#64B5F6"],
        ax=ax3
    )
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.set_ylabel("Volume")
    st.pyplot(fig3)

# -----------------------------------------------------------------------------
# 7. MODULE 3: MODEL DIAGNOSTICS & EVALUATION
# -----------------------------------------------------------------------------
elif nav == "3. Model Architecture & Diagnostics":
    st.title("🤖 Model evaluation")
    st.markdown("Ensemble Random Forest Classifier with automated `SimpleImputer` preprocessing.")
    st.write("---")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", f"{model_artifacts['accuracy']*100:.1f}%")
    
    # Delta used to prevent Streamlit ellipsis truncation on narrow columns
    m2.metric(
        "5-Fold CV Accuracy", 
        f"{model_artifacts['cv_mean']*100:.1f}%", 
        delta=f"±{model_artifacts['cv_std']*100:.1f}% (CV Std)", 
        delta_color="off"
    )
    
    m3.metric("Recall (Sensitivity)", f"{model_artifacts['recall']*100:.1f}%")
    m4.metric("ROC-AUC Score", f"{model_artifacts['roc_auc']:.2f}")

    col_diag_left, col_diag_right = st.columns(2)

    with col_diag_left:
        st.markdown("##### Confusion Matrix (Validation Cohort)")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8), dpi=150)
        sns.heatmap(
            model_artifacts['conf_matrix'],
            annot=True,
            fmt='d',
            cmap="Blues",
            xticklabels=["Rejected (0)", "Approved (1)"],
            yticklabels=["Rejected (0)", "Approved (1)"],
            ax=ax_cm
        )
        ax_cm.set_xlabel("Predicted Class")
        ax_cm.set_ylabel("Ground Truth")
        st.pyplot(fig_cm)

    with col_diag_right:
        st.markdown("##### Gini Feature Importance Ranking")
        fig_fi, ax_fi = plt.subplots(figsize=(5, 3.8), dpi=150)
        model_artifacts['importances'].plot(kind='barh', color="#1976D2", ax=ax_fi)
        ax_fi.invert_yaxis()
        ax_fi.set_xlabel("Relative Decision Weight")
        ax_fi.grid(axis='x', linestyle='--', alpha=0.3)
        st.pyplot(fig_fi)

# -----------------------------------------------------------------------------
# 8. MODULE 4: AUTOMATED UNDERWRITING SIMULATOR
# -----------------------------------------------------------------------------
elif nav == "4. Automated Underwriting Simulator":
    st.title("🔍 Automated Credit Risk Underwriting Interface")
    st.markdown("Enter applicant details to generate a model prediction.")
    st.write("---")

    with st.form("underwriting_simulation_form"):
        col_input_1, col_input_2, col_input_3 = st.columns(3)

        with col_input_1:
            st.markdown("**1. Financial Capacity**")
            applicant_income = st.number_input("Primary Monthly Income (₹)", 1000, 100000, 5500, step=500)
            coapplicant_income = st.number_input("Coapplicant Monthly Income (₹)", 0, 50000, 1500, step=500)
            married_status = st.selectbox("Marital Status", ["Yes", "No"])

        with col_input_2:
            st.markdown("**2. Facility Details**")
            loan_amount_k = st.number_input("Loan Amount (₹ in Thousands)", 10, 800, 130, step=10)
            term_months = st.selectbox("Amortization Period (Months)", [360, 180, 240, 120], index=0)
            education_status = st.selectbox("Educational Attainment", ["Graduate", "Not Graduate"])

        with col_input_3:
            st.markdown("**3. Risk Indicators**")
            credit_history_flag = st.selectbox(
                "Credit Bureau Record",
                [1.0, 0.0],
                format_func=lambda x: "Clean Record (No Defaults - 1.0)" if x == 1.0 else "Adverse Record / Past Default (0.0)"
            )
            property_location = st.selectbox("Property Zoning", ["Urban", "Semiurban", "Rural"])

        evaluate_button = st.form_submit_button("Compute Underwriting Decision", use_container_width=True)

    if evaluate_button:
        combined_income = applicant_income + coapplicant_income
        calculated_emi = (loan_amount_k * 1000) / term_months
        calculated_dti = calculated_emi / (combined_income + 1e-5)

        inference_payload = pd.DataFrame([{
            'Credit_History': credit_history_flag,
            'Total_Income': combined_income,
            'LoanAmount': loan_amount_k,
            'Loan_Amount_Term': term_months,
            'Debt_To_Income': calculated_dti,
            'EMI_Estimate': calculated_emi,
            'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}[property_location],
            'Education': 1 if education_status == "Graduate" else 0,
            'Married': 1 if married_status == "Yes" else 0
        }])[FEATURE_COLUMNS]

        prediction_class = model_artifacts['pipeline'].predict(inference_payload)[0]
        prediction_probabilities = model_artifacts['pipeline'].predict_proba(inference_payload)[0]

        st.markdown("### Underwriting Verdict")
        if prediction_class == 1:
            st.success(f"""
            #### ✅ APPLICATION APPROVED: LOW RISK PROFILE
            - **Approval Confidence:** `{prediction_probabilities[1]*100:.1f}%`
            - **Estimated DTI Ratio:** `{calculated_dti*100:.2f}%`
            - **Total Servicing Income:** `₹{combined_income:,}/month`
            - **Straight-Through Processing:** Qualifies for automated clearance under standard rate schedule.
            """)
        else:
            st.error(f"""
            #### ❌ APPLICATION FLAGGED: ELEVATED RISK OF DEFAULT
            - **Default Risk Probability:** `{prediction_probabilities[0]*100:.1f}%`
            - **Estimated DTI Ratio:** `{calculated_dti*100:.2f}%`
            - **Adverse Factors:** Insufficient credit bureau validation or high leverage relative to income.
            - **Action:** Route to Secondary Underwriter or request collateral enhancement.
            """)

# -------------------------------------------------------------
# 9. MODULE 5: FAIRNESS & UN SDG 10 AUDIT
# -------------------------------------------------------------
elif nav == "5. Algorithmic Fairness & UN SDG 10":
    st.title("⚖️ Algorithmic Fairness & Ethical AI Audit")
    st.markdown("Descriptive comparison of historical approval rates by gender.")
    st.write("---")

    st.markdown("""
    ### Why `Gender` Was Deliberately Excluded
    To prevent disparate impact and comply with fair lending standards, **`Gender` is intentionally omitted from the model's feature set**.
    """)

    # Demographic Parity Evaluation
    gender_approval = cleaned_df.groupby('Gender')['Loan_Status'].apply(lambda s: (s == 'Y').mean() * 100)

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("##### Historical Approval Rate by Gender")
        st.dataframe(gender_approval.rename("Approval Rate (%)"))
        disparate_impact = gender_approval.get('Female', 0) / (gender_approval.get('Male', 1) + 1e-5)
        st.metric("Historical Approval-Rate Ratio (Female / Male)", f"{disparate_impact:.3f}")
        st.caption("Ratio $\ge 0.80$ meets the Four-Fifths Rule for non-discrimination.")

    with f_col2:
        st.markdown("##### Governance Policy")
        st.info("""
        - **Objective:** Mitigate demographic bias in credit scoring.
        - **Implementation:** `Gender` is excluded from the model feature set. 
        - **Result:** The displayed ratio compares historical approval rates and should not be interpreted as absolute proof of algorithmic fairness in production.
        """)

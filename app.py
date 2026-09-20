"""
Streamlit application for the loan approval prediction project.

The app:
- Loads and validates the loan dataset.
- Engineers income and debt-related features.
- Trains a Random Forest model using a leakage-safe preprocessing pipeline.
- Displays data health, EDA, model diagnostics, predictions, and fairness metrics.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Assessment Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. CONSTANTS
# -----------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "Credit_History",
    "Total_Income",
    "LoanAmount",
    "Loan_Amount_Term",
    "Debt_To_Income",
    "EMI_Estimate",
    "Property_Area",
    "Education",
    "Married",
]

REQUIRED_COLUMNS = {
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
    "Education",
    "Married",
    "Loan_Status",
}

ENCODING_MAP = {
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
    "Loan_Status": {"Y": 1, "N": 0},
}


# -----------------------------------------------------------------------------
# 3. DATA INGESTION & FEATURE ENGINEERING
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_engineer_features(
    filepath: str = "data/train_loan_data.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset, validate required columns, and engineer model features.

    Returns:
        cleaned_df: Human-readable data used by the dashboard.
        encoded_df: Numerically encoded data used by the model.
    """
    project_dir = Path(__file__).resolve().parent

    candidate_paths = [
        project_dir / filepath,
        project_dir / Path(filepath).name,
        Path(filepath),
        Path("train_loan_data.csv"),
    ]

    data_path = next((path for path in candidate_paths if path.exists()), None)

    if data_path is None:
        searched_paths = "\n".join(f"- {path}" for path in candidate_paths)
        raise FileNotFoundError(
            "The loan dataset could not be found. Searched these locations:\n"
            f"{searched_paths}"
        )

    raw_data = pd.read_csv(data_path)

    missing_columns = REQUIRED_COLUMNS.difference(raw_data.columns)
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df = raw_data.copy()

    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # Normalize text columns to avoid accidental values such as " Yes ".
    text_columns = [
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Property_Area",
        "Loan_Status",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    # Convert financial columns to numeric values.
    numeric_columns = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Feature engineering.
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    valid_term = df["Loan_Amount_Term"].replace(0, np.nan)

    df["EMI_Estimate"] = (df["LoanAmount"] * 1000) / valid_term

    df["Debt_To_Income"] = df["EMI_Estimate"] / (
        df["Total_Income"].replace(0, np.nan)
    )

    # Make a separate encoded copy for model training.
    encoded_df = df.copy()

    for column, mapping in ENCODING_MAP.items():
        if column in encoded_df.columns:
            encoded_df[column] = encoded_df[column].map(mapping)

    return df, encoded_df


try:
    cleaned_df, encoded_df = load_and_engineer_features()
except Exception as exc:
    st.error("The application could not load the loan dataset.")
    st.exception(exc)
    st.stop()


# -----------------------------------------------------------------------------
# 4. MODEL TRAINING
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def train_leakage_safe_model(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Train and evaluate a Random Forest model.

    Missing-value imputation is inside the pipeline, so the imputer is fitted
    separately within each training fold and never learns from validation data.
    """
    missing_features = set(FEATURE_COLUMNS).difference(data.columns)
    if missing_features:
        raise ValueError(
            "The following model features are missing: "
            + ", ".join(sorted(missing_features))
        )

    if "Loan_Status" not in data.columns:
        raise ValueError("The target column 'Loan_Status' is missing.")

    valid_data = data.dropna(subset=["Loan_Status"]).copy()

    if valid_data.empty:
        raise ValueError("No rows with a valid Loan_Status value were found.")

    X = valid_data[FEATURE_COLUMNS]
    y = valid_data["Loan_Status"].astype(int)

    class_counts = y.value_counts()

    if len(class_counts) != 2:
        raise ValueError(
            "Loan_Status must contain both classes: approved and rejected."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Each Loan_Status class must contain at least two records."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=5,
                    min_samples_split=6,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # The number of folds cannot exceed the smallest class in the training set.
    train_class_counts = y_train.value_counts()
    n_splits = min(5, int(train_class_counts.min()))

    if n_splits < 2:
        raise ValueError(
            "There are not enough training records for stratified cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    classifier = pipeline.named_steps["classifier"]

    feature_importances = pd.Series(
        classifier.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=False)

    return {
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_folds": n_splits,
        "importances": feature_importances,
    }


try:
    model_artifacts = train_leakage_safe_model(encoded_df)
except Exception as exc:
    st.error("The machine-learning model could not be trained.")
    st.exception(exc)
    st.stop()


# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://img.shields.io/badge/IBM-SkillsBuild-052FAD?"
        "style=for-the-badge&logo=ibm&logoColor=white"
    )

    st.markdown("### System Controls")

    nav = st.radio(
        "Module Navigation:",
        [
            "1. Executive Overview & Data Health",
            "2. Exploratory Data Analysis (EDA)",
            "3. Model Architecture & Diagnostics",
            "4. Automated Underwriting Simulator",
            "5. Algorithmic Fairness & UN SDG 10",
        ],
    )

    st.markdown("---")
    st.caption("**Candidate:** Srijan Das")
    st.caption("**Internship ID:** `IBMUEDA0483`")
    st.caption("**Institution:** CGEC (CSE)")


# -----------------------------------------------------------------------------
# 6. MODULE 1: EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------

if nav == "1. Executive Overview & Data Health":
    st.title("🏦 Credit Risk Assessment & Automated Loan Underwriting Engine")
    st.markdown("#### Dataset summary and model inputs")
    st.write("---")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Audited Applicants</div>
                <div class="metric-value">{len(cleaned_df)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        approval_rate = (
            cleaned_df["Loan_Status"].eq("Y").mean() * 100
            if "Loan_Status" in cleaned_df.columns
            else np.nan
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Historical Approval Rate</div>
                <div class="metric-value">{approval_rate:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        avg_income = cleaned_df["Total_Income"].dropna().mean()

        average_income_text = (
            f"₹{int(avg_income):,}"
            if pd.notna(avg_income)
            else "Unavailable"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mean Household Income</div>
                <div class="metric-value">{average_income_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Pipeline Architecture</div>
                <div class="metric-value">Leakage-Safe</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Sample of loaded records")
    st.dataframe(cleaned_df.head(6), use_container_width=True)

    with st.expander("Descriptive statistics", expanded=True):
        st.dataframe(cleaned_df.describe(include="all").T, use_container_width=True)


# -----------------------------------------------------------------------------
# 7. MODULE 2: EXPLORATORY DATA ANALYSIS
# -----------------------------------------------------------------------------

elif nav == "2. Exploratory Data Analysis (EDA)":
    st.title("📈 Exploratory Analysis")
    st.markdown(
        "Evaluating demographic and financial relationships associated with "
        "historical loan approval outcomes."
    )
    st.write("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### 1. Credit History and Approval Outcome")

        credit_data = cleaned_df.dropna(
            subset=["Credit_History", "Loan_Status"]
        ).copy()

        fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)

        if not credit_data.empty:
            sns.countplot(
                data=credit_data,
                x="Credit_History",
                hue="Loan_Status",
                hue_order=["N", "Y"],
                palette={"N": "#D32F2F", "Y": "#1976D2"},
                ax=ax,
            )

            ax.set_xlabel("Credit History")
            ax.set_ylabel("Applicant Count")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.legend(title="Loan Status")
        else:
            ax.text(
                0.5,
                0.5,
                "No credit-history records available",
                ha="center",
                va="center",
            )
            ax.set_axis_off()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "This chart shows the historical association between credit history "
            "and loan approval. It should not be interpreted as causal evidence."
        )

    with col_right:
        st.markdown("##### 2. Requested Capital vs. Household Income")

        scatter_data = cleaned_df.dropna(
            subset=["Total_Income", "LoanAmount", "Loan_Status"]
        ).copy()

        fig2, ax2 = plt.subplots(figsize=(6, 3.8), dpi=150)

        if not scatter_data.empty:
            sns.scatterplot(
                data=scatter_data,
                x="Total_Income",
                y="LoanAmount",
                hue="Loan_Status",
                hue_order=["N", "Y"],
                palette={"N": "#D32F2F", "Y": "#1976D2"},
                alpha=0.75,
                ax=ax2,
            )

            ax2.set_xlim(left=0)
            ax2.set_xlabel("Total Household Income (₹)")
            ax2.set_ylabel("Loan Amount (₹ Thousands)")
            ax2.grid(linestyle="--", alpha=0.3)
            ax2.legend(title="Loan Status")
        else:
            ax2.text(
                0.5,
                0.5,
                "No financial records available",
                ha="center",
                va="center",
            )
            ax2.set_axis_off()

        st.pyplot(fig2)
        plt.close(fig2)

        st.caption(
            "This plot displays the historical relationship between household "
            "income and requested loan amount."
        )

    st.markdown("##### 3. Property Location Demographics")

    property_data = cleaned_df.dropna(
        subset=["Property_Area", "Loan_Status"]
    ).copy()

    fig3, ax3 = plt.subplots(figsize=(10, 2.8), dpi=150)

    if not property_data.empty:
        sns.countplot(
            data=property_data,
            x="Property_Area",
            hue="Loan_Status",
            hue_order=["N", "Y"],
            palette={"N": "#E57373", "Y": "#64B5F6"},
            order=["Rural", "Semiurban", "Urban"],
            ax=ax3,
        )

        ax3.set_xlabel("Property Area")
        ax3.set_ylabel("Volume")
        ax3.grid(axis="y", linestyle="--", alpha=0.3)
        ax3.legend(title="Loan Status")
    else:
        ax3.text(
            0.5,
            0.5,
            "No property-area records available",
            ha="center",
            va="center",
        )
        ax3.set_axis_off()

    st.pyplot(fig3)
    plt.close(fig3)


# -----------------------------------------------------------------------------
# 8. MODULE 3: MODEL DIAGNOSTICS
# -----------------------------------------------------------------------------

elif nav == "3. Model Architecture & Diagnostics":
    st.title("🤖 Model Evaluation")
    st.markdown(
        "Random Forest classifier with automated median imputation inside a "
        "scikit-learn pipeline."
    )
    st.write("---")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Test Accuracy",
        f"{model_artifacts['accuracy'] * 100:.1f}%",
    )

    m2.metric(
        f"{model_artifacts['cv_folds']}-Fold CV Accuracy",
        f"{model_artifacts['cv_mean'] * 100:.1f}%",
        delta=f"±{model_artifacts['cv_std'] * 100:.1f}% CV std",
        delta_color="off",
    )

    m3.metric(
        "Recall (Sensitivity)",
        f"{model_artifacts['recall'] * 100:.1f}%",
    )

    m4.metric(
        "ROC-AUC Score",
        f"{model_artifacts['roc_auc']:.2f}",
    )

    st.info(
        "These metrics are estimates from one train/test split and should not "
        "be treated as proof of production-level credit performance."
    )

    col_diag_left, col_diag_right = st.columns(2)

    with col_diag_left:
        st.markdown("##### Confusion Matrix")

        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8), dpi=150)

        sns.heatmap(
            model_artifacts["conf_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Rejected (0)", "Approved (1)"],
            yticklabels=["Rejected (0)", "Approved (1)"],
            ax=ax_cm,
        )

        ax_cm.set_xlabel("Predicted Class")
        ax_cm.set_ylabel("Ground Truth")

        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_diag_right:
        st.markdown("##### Random Forest Feature Importance")

        fig_fi, ax_fi = plt.subplots(figsize=(5, 3.8), dpi=150)

        model_artifacts["importances"].sort_values().plot(
            kind="barh",
            color="#1976D2",
            ax=ax_fi,
        )

        ax_fi.set_xlabel("Relative Feature Importance")
        ax_fi.grid(axis="x", linestyle="--", alpha=0.3)

        st.pyplot(fig_fi)
        plt.close(fig_fi)


# -----------------------------------------------------------------------------
# 9. MODULE 4: AUTOMATED UNDERWRITING SIMULATOR
# -----------------------------------------------------------------------------

elif nav == "4. Automated Underwriting Simulator":
    st.title("🔍 Automated Credit Risk Underwriting Interface")
    st.markdown(
        "Enter applicant details to generate a model prediction. This is a "
        "demonstration and must not be used as the sole basis for a lending "
        "decision."
    )
    st.write("---")

    with st.form("underwriting_simulation_form"):
        col_input_1, col_input_2, col_input_3 = st.columns(3)

        with col_input_1:
            st.markdown("**1. Financial Capacity**")

            applicant_income = st.number_input(
                "Primary Monthly Income (₹)",
                min_value=1000,
                max_value=100000,
                value=5500,
                step=500,
            )

            coapplicant_income = st.number_input(
                "Coapplicant Monthly Income (₹)",
                min_value=0,
                max_value=50000,
                value=1500,
                step=500,
            )

            married_status = st.selectbox(
                "Marital Status",
                ["Yes", "No"],
            )

        with col_input_2:
            st.markdown("**2. Facility Details**")

            loan_amount_k = st.number_input(
                "Loan Amount (₹ in Thousands)",
                min_value=10,
                max_value=800,
                value=130,
                step=10,
            )

            term_months = st.selectbox(
                "Amortization Period (Months)",
                [360, 180, 240, 120],
                index=0,
            )

            education_status = st.selectbox(
                "Educational Attainment",
                ["Graduate", "Not Graduate"],
            )

        with col_input_3:
            st.markdown("**3. Risk Indicators**")

            credit_history_flag = st.selectbox(
                "Credit Bureau Record",
                [1.0, 0.0],
                format_func=lambda value: (
                    "Clean Record / No Defaults (1.0)"
                    if value == 1.0
                    else "Adverse Record / Past Default (0.0)"
                ),
            )

            property_location = st.selectbox(
                "Property Zoning",
                ["Urban", "Semiurban", "Rural"],
            )

        evaluate_button = st.form_submit_button(
            "Compute Underwriting Decision",
            use_container_width=True,
        )

    if evaluate_button:
        combined_income = applicant_income + coapplicant_income

        if combined_income <= 0 or term_months <= 0:
            st.error(
                "Income and amortization period must both be greater than zero."
            )
            st.stop()

        calculated_emi = (loan_amount_k * 1000) / term_months
        calculated_dti = calculated_emi / combined_income

        inference_payload = pd.DataFrame(
            [
                {
                    "Credit_History": credit_history_flag,
                    "Total_Income": combined_income,
                    "LoanAmount": loan_amount_k,
                    "Loan_Amount_Term": term_months,
                    "Debt_To_Income": calculated_dti,
                    "EMI_Estimate": calculated_emi,
                    "Property_Area": {
                        "Rural": 0,
                        "Semiurban": 1,
                        "Urban": 2,
                    }[property_location],
                    "Education": (
                        1 if education_status == "Graduate" else 0
                    ),
                    "Married": 1 if married_status == "Yes" else 0,
                }
            ]
        )[FEATURE_COLUMNS]

        prediction_class = model_artifacts["pipeline"].predict(
            inference_payload
        )[0]

        prediction_probabilities = model_artifacts["pipeline"].predict_proba(
            inference_payload
        )[0]

        approval_probability = prediction_probabilities[1]
        rejection_probability = prediction_probabilities[0]

        st.markdown("### Model Output")

        if prediction_class == 1:
            st.success(
                f"""
                #### ✅ MODEL CLASSIFICATION: APPROVAL-LIKE PROFILE

                - **Estimated approval probability:** `{approval_probability * 100:.1f}%`
                - **Estimated DTI ratio:** `{calculated_dti * 100:.2f}%`
                - **Total servicing income:** `₹{combined_income:,}/month`

                This is a statistical model prediction, not a final lending
                decision. Additional verification and human review may be required.
                """
            )
        else:
            st.error(
                f"""
                #### ❌ MODEL CLASSIFICATION: HIGHER-RISK PROFILE

                - **Estimated rejection probability:** `{rejection_probability * 100:.1f}%`
                - **Estimated DTI ratio:** `{calculated_dti * 100:.2f}%`
                - **Total servicing income:** `₹{combined_income:,}/month`

                The prediction should be reviewed together with documentation,
                policy rules, and a qualified underwriter.
                """
            )


# -----------------------------------------------------------------------------
# 10. MODULE 5: FAIRNESS & UN SDG 10 AUDIT
# -----------------------------------------------------------------------------

elif nav == "5. Algorithmic Fairness & UN SDG 10":
    st.title("⚖️ Algorithmic Fairness & Ethical AI Audit")
    st.markdown(
        "Descriptive comparison of historical approval rates by gender."
    )
    st.write("---")

    st.markdown(
        """
        ### Why `Gender` Was Excluded from the Model

        `Gender` is intentionally excluded from the model feature set.
        However, excluding a protected attribute alone does not guarantee
        fairness because other variables may act as proxy variables.
        """
    )

    if "Gender" not in cleaned_df.columns:
        st.warning(
            "The dataset does not contain a Gender column, so the historical "
            "gender comparison cannot be calculated."
        )
    else:
        fairness_data = cleaned_df.dropna(
            subset=["Gender", "Loan_Status"]
        ).copy()

        gender_approval = (
            fairness_data.groupby("Gender")["Loan_Status"]
            .apply(lambda values: values.eq("Y").mean() * 100)
            .sort_index()
        )

        f_col1, f_col2 = st.columns(2)

        with f_col1:
            st.markdown("##### Historical Approval Rate by Gender")

            if gender_approval.empty:
                st.warning("No complete gender records are available.")
            else:
                st.dataframe(
                    gender_approval.rename("Approval Rate (%)"),
                    use_container_width=True,
                )

                female_rate = gender_approval.get("Female")
                male_rate = gender_approval.get("Male")

                if (
                    female_rate is not None
                    and male_rate is not None
                    and male_rate > 0
                ):
                    disparate_impact = female_rate / male_rate

                    st.metric(
                        "Historical Approval-Rate Ratio (Female / Male)",
                        f"{disparate_impact:.3f}",
                    )

                    st.caption(
                        "A ratio below 0.80 may indicate a disparity under the "
                        "Four-Fifths Rule, but this descriptive statistic is not "
                        "a complete legal or fairness assessment."
                    )
                else:
                    st.warning(
                        "Both Female and Male groups with valid approval data "
                        "are required to calculate the ratio."
                    )

        with f_col2:
            st.markdown("##### Governance Policy")

            st.info(
                """
                - **Objective:** Reduce avoidable demographic bias.
                - **Implementation:** Gender is excluded from the model features.
                - **Limitation:** Other variables may still act as proxies.
                - **Recommendation:** Evaluate group-level error rates,
                  calibration, sample sizes, and protected groups before any
                  real-world deployment.
                """
            )

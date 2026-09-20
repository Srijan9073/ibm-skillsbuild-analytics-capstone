# Credit Risk Assessment & Automated Loan Underwriting Engine

An end-to-end data analytics and predictive underwriting decision engine developed for the **AICTE–IBM SkillsBuild Data Analytics with AI Virtual Internship (2026)**.

[![Internship ID](https://img.shields.io/badge/Internship_ID-IBMUEDA0483-blue?style=flat-square)](#)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#)
[![UN SDG](https://img.shields.io/badge/UN_SDG-Goal_8_&_10-4C9F38?style=flat-square)](#)

---

## 📌 Summary
Traditional retail credit underwriting relies on manual reviews that introduce latency, human inconsistency, and high rejection rates for unbanked borrowers. This project delivers an automated, audit-cleared credit scoring pipeline that balances credit portfolio expansion against Non-Performing Asset (NPA) risk.

### 🌐 UN Sustainable Development Goals (UN SDGs) Alignment
- **UN SDG 8 (Decent Work & Economic Growth):** Eliminates underwriting backlogs to accelerate capital access for micro-enterprises and families.
- **UN SDG 10 (Reduced Inequalities via Algorithmic Fairness):** Employs objective financial capacity metrics while omitting protected demographic attributes (`Gender`). An empirical fairness audit confirmed a **Disparate Impact Ratio of 0.966**, fully complying with the regulatory Four-Fifths Rule.

---

## 📊 Dataset Reference
- **Source:** [Kaggle - Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Observations:** 614 credit applicant records with 12 demographic and financial parameters.
- **Target Variable:** `Loan_Status` (Binary: Approved `Y` / Rejected `N`).

---

## 🛠️ Architecture & Empirical Benchmarks
- **Data Engineering:** Integrated domain indicators (`Total_Income`, `EMI_Estimate`, `Debt_To_Income`).
- **Leakage-Free Pipeline:** Preprocessing and `SimpleImputer(strategy='median')` are wrapped in a Scikit-Learn `Pipeline` fitted strictly on training folds.
- **Model:** Regularized ensemble Random Forest Classifier (120 estimators, max depth = 5, balanced class weights).

### 📈 Verified Performance Metrics (Tested on Python 3.14)
- **5-Fold Stratified Cross-Validation:** **74.7% ± 5.1%** (+6.0% lift over 68.7% naive baseline)
- **Holdout Validation Accuracy:** **81.3%** (100 / 123)
- **Precision (Approval Class):** **87.8%** (72 / 82)
- **Recall (Sensitivity):** **84.7%** (72 / 85, minimizing false rejections)
- **ROC-AUC Score:** **0.85**
- **Disparate Impact Ratio:** **0.966** (No demographic bias)

---

## 📂 Repository Contents
- `app.py`: Single-file pipeline and interactive Streamlit underwriting simulator.
- `project_report.pdf`: 5-page publication-grade LaTeX report with all 7 UI figures.
- `train_loan_data.csv`: Source benchmark dataset.
- `requirements.txt` & `lockfile.txt`: Pinned, project-scoped environment dependencies.
- `output.txt`: Raw execution log from `verify_metrics.py`.
- `runtime.txt`: Specified Python 3.14 runtime.

---

## 🚀 How to Run Locally
```bash
# 1. Clone repository
git clone https://github.com/Srijan9073/ibm-skillsbuild-analytics-capstone.git
cd ibm-skillsbuild-analytics-capstone

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py

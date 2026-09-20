# Credit Risk Assessment & Loan Approval Prediction

![Streamlit dashboard screenshot](reports/dashboard-screenshot.png)

A Streamlit prototype that predicts loan approval using applicant income, loan details, credit history, education, marital status, and property area.

Built as the final project for the **AICTE–IBM SkillsBuild Data Analytics with AI Virtual Internship (2026)**.

## 📊 Results at a Glance

| Measure | Result |
| :--- | :--- |
| 5-fold cross-validation accuracy | **74.7% ± 5.1%** |
| Improvement over majority-class baseline | **+6.0 percentage points** |
| Holdout test accuracy | **81.3%** (100/123) |
| Approval precision | **87.8%** |
| Approval recall | **84.7%** (correctly identified 72 of 85 approved cases) |
| ROC-AUC | **0.85** |
| Historical female/male approval-rate ratio | **0.966** |

> **Scope & Limitations:** This is an educational prototype trained on 614 records from a public Kaggle dataset. The reported results demonstrate pipeline construction and baseline predictive lift. They should not be interpreted as evidence that this model is ready for production lending decisions or fully audited for systemic fairness.

---

## 🛠️ Architecture & Pipeline

- **Model validation:** Missing-value imputation is inside the Scikit-learn `Pipeline`, so the imputer is fitted separately within each training fold to prevent data leakage. The model uses a stratified 80/20 train-test split with a fixed random seed.
- **Model:** Random Forest Classifier (`n_estimators=120`, `max_depth=5`, `class_weight='balanced'`).
- **Feature Engineering:** `Total_Income`, `EMI_Estimate`, and `Debt_To_Income` are computed from raw applicant parameters prior to the train/test split.

---

## ⚖️ Fairness Check (UN SDG 10)

`Gender` is excluded from the model feature set. The application separately compares historical approval rates by gender and reports a female-to-male ratio of 0.966 for this dataset. 

*Note: This is a descriptive dataset check, not a complete fairness evaluation. A production assessment would also compare model predictions, false-positive/negative rates, sample sizes, and performance across additional protected groups.*

---

## 📂 Repository Layout

- [`app.py`](./app.py): Streamlit application and ML pipeline.
- [`data/train_loan_data.csv`](./data/train_loan_data.csv): Kaggle benchmark dataset.
- [`scripts/verify_metrics.py`](./scripts/verify_metrics.py): Script to locally reproduce the reported validation metrics.
- [`reports/project_report.pdf`](./reports/project_report.pdf): PDF report describing the dataset, model, and validation results.
- [`requirements.txt`](./requirements.txt): Direct project dependencies.
- [`requirements.lock`](./requirements.lock): Full environment freeze used for the reported run.
- [`runtime.txt`](./runtime.txt): Python runtime used for development (Python 3.14).

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

```
---

## 👨‍💻 Candidate & Submission Details
- **Candidate:** Srijan Das
- **Internship ID:** `IBMUEDA0483`
- **Institution:** Cooch Behar Government Engineering College (CGEC)
- **Program:** AICTE–BharatCares–IBM SkillsBuild Data Analytics with AI Virtual Internship
- **LinkedIn:** [linkedin.com/in/srijandas2099](https://www.linkedin.com/in/srijandas2099/)

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# Dynamically locate the dataset whether running from root or scripts/
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "train_loan_data.csv" if (BASE_DIR.parent / "data" / "train_loan_data.csv").exists() else BASE_DIR / "train_loan_data.csv"

# Ingest and engineer financial ratios
df = pd.read_csv(DATA_PATH)
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['EMI_Estimate'] = (df['LoanAmount'] * 1000) / df['Loan_Amount_Term'].replace(0, np.nan)
df['Debt_To_Income'] = df['EMI_Estimate'] / (df['Total_Income'] + 1e-5)

encoding_map = {
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Loan_Status': {'Y': 1, 'N': 0}
}
for col, mapping in encoding_map.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

FEATURE_COLUMNS = [
    'Credit_History', 'Total_Income', 'LoanAmount', 'Loan_Amount_Term',
    'Debt_To_Income', 'EMI_Estimate', 'Property_Area', 'Education', 'Married'
]

valid = df.dropna(subset=['Loan_Status']).copy()
X = valid[FEATURE_COLUMNS]
y = valid['Loan_Status'].astype(int)

# Stratified 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Pipeline encapsulates median imputer to prevent data leakage
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_split=6, random_state=42, class_weight='balanced'))
])

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Format confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_formatted = f"[[{cm[0, 0]}, {cm[0, 1]}],\n [{cm[1, 0]}, {cm[1, 1]}]]"

# Print exact layout to mirror output.txt
print("Model Evaluation Results")
print("========================")
print()
print("These metrics were produced by running the verification script locally.")
print()
print(f"5-fold cross-validation accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
print(f"Holdout test accuracy:           {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"Approval precision:              {precision_score(y_test, y_pred)*100:.1f}%")
print(f"Approval recall:                 {recall_score(y_test, y_pred)*100:.1f}%")
print(f"ROC-AUC:                         {roc_auc_score(y_test, y_proba):.2f}")
print()
print("Confusion matrix:")
print(cm_formatted)
print()
print("Class order: [Rejected, Approved]")

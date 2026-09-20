import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# Load data
df = pd.read_csv("train_loan_data.csv")
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_split=6, random_state=42, class_weight='balanced'))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("=" * 45)
print("  OFFICIAL AUDITED METRICS (LOCAL EXECUTION)")
print("=" * 45)
print(f"5-Fold CV Accuracy:   {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
print(f"Holdout Test Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"Test Precision:        {precision_score(y_test, y_pred)*100:.1f}%")
print(f"Test Recall:           {recall_score(y_test, y_pred)*100:.1f}%")
print(f"Test ROC-AUC:          {roc_auc_score(y_test, y_proba):.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("=" * 45)
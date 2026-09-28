"""
HR Employee Attrition: data cleaning, analytics, training, and evaluation.
Run: python train_model.py
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, log_loss)

INPUT_CSV = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUT_DIR = "outputs"
TARGET = "Attrition"
DROP_CANDIDATES = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    duplicate_count = int(df.duplicated().sum())
    missing_before = df.isna().sum()
    clean = df.drop_duplicates().copy()
    for col in clean.columns:
        if clean[col].isna().any():
            if pd.api.types.is_numeric_dtype(clean[col]):
                clean[col] = clean[col].fillna(clean[col].median())
            else:
                mode = clean[col].mode(dropna=True)
                clean[col] = clean[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
    clean.to_csv(os.path.join(OUT_DIR, "cleaned_hr_employee_attrition.csv"), index=False)

    if TARGET not in clean:
        raise ValueError(f"Target column {TARGET!r} not found.")
    y = clean[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Target must contain Yes/No values.")
    X = clean.drop(columns=[TARGET] + [c for c in DROP_CANDIDATES if c in clean.columns])
    constant_cols = [c for c in X if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X if c not in num_cols]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])
    pipe = Pipeline([("preprocessor", prep),
                     ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    metrics = {
        "Model": "Balanced Logistic Regression", "Train rows": len(X_train), "Test rows": len(X_test),
        "Missing cells before": int(missing_before.sum()), "Missing cells after": int(clean.isna().sum().sum()),
        "Duplicate rows removed": duplicate_count, "Accuracy": accuracy_score(y_test, pred),
        "Balanced accuracy": balanced_accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0), "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob), "MCC": matthews_corrcoef(y_test, pred),
        "Log loss": log_loss(y_test, prob), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "R2": "Not applicable for binary classification"
    }
    pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).to_csv(
        os.path.join(OUT_DIR, "model_metrics.csv"), index=False)
    pd.DataFrame(classification_report(y_test, pred, target_names=["No Attrition", "Attrition"],
                                       output_dict=True, zero_division=0)).transpose().to_csv(
        os.path.join(OUT_DIR, "classification_report.csv"))
    pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual No", "Actual Yes"],
                 columns=["Predicted No", "Predicted Yes"]).to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))
    joblib.dump(pipe, os.path.join(OUT_DIR, "attrition_model.sav"))
    print(pd.Series(metrics).to_string())
    print("Saved cleaned data, reports, and model to", OUT_DIR)

if __name__ == "__main__":
    main()

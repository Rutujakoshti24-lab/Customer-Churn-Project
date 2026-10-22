"""
Trains Logistic Regression and Random Forest on the telecom churn data,
compares them, and saves the best model (by ROC-AUC) to models/churn_model.joblib
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_model.joblib"
METRICS_PATH = "models/metrics.json"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    return df


def get_feature_lists(X: pd.DataFrame):
    num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    cat = [c for c in X.columns if c not in num]
    return num, cat


def main():
    df = clean(pd.read_csv(DATA_PATH))
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])
    num, cat = get_feature_lists(X)

    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
    }

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    results, fitted = {}, {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_te, proba)
        results[name] = {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, proba),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
            "roc_curve": {"fpr": fpr[::5].tolist(), "tpr": tpr[::5].tolist()},
        }
        fitted[name] = pipe
        print(f"{name:20s} AUC={results[name]['roc_auc']:.3f}  "
              f"Acc={results[name]['accuracy']:.3f}  Recall={results[name]['recall']:.3f}")

    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest model: {best}")

    # feature importance for the best model
    pipe = fitted[best]
    names = pipe.named_steps["pre"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    imp = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
    top = pd.Series(imp, index=names).sort_values(ascending=False).head(15)
    top.index = [i.split("__", 1)[1] for i in top.index]

    joblib.dump({"model": pipe, "name": best}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best": best, "results": results, "importance": top.to_dict()}, f, indent=2)
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()

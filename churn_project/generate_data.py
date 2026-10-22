"""
Generates a synthetic telecom churn dataset with the same columns as the
public IBM / Kaggle "Telco Customer Churn" dataset.

If you have the real dataset, just save it as  data/telco_churn.csv  and skip this script.
(Kaggle: "Telco Customer Churn" by blastchar)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 7043


def build(n=N):
    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure = np.clip(rng.gamma(2.0, 18, n).astype(int), 0, 72)

    phone = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multi = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n))
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def addon(p=0.4):
        return np.where(internet == "No", "No internet service",
                        rng.choice(["Yes", "No"], n, p=[p, 1 - p]))

    security, backup, protection = addon(0.30), addon(0.35), addon(0.35)
    support, tv, movies = addon(0.30), addon(0.45), addon(0.45)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21])

    base = np.where(internet == "Fiber optic", 75, np.where(internet == "DSL", 52, 20))
    monthly = base + np.where(tv == "Yes", 9, 0) + np.where(movies == "Yes", 9, 0) \
        + np.where(multi == "Yes", 5, 0) + rng.normal(0, 4, n)
    monthly = np.clip(monthly, 18.25, 118.75).round(2)
    total = (monthly * np.maximum(tenure, 1) * rng.uniform(0.95, 1.05, n)).round(2)

    # churn probability driven by realistic factors
    z = (-1.0
         + 1.5 * (contract == "Month-to-month")
         - 1.2 * (contract == "Two year")
         + 0.8 * (internet == "Fiber optic")
         + 0.7 * (payment == "Electronic check")
         + 0.4 * (paperless == "Yes")
         + 0.3 * senior
         - 0.9 * (security == "Yes") - 0.7 * (support == "Yes")
         - 0.04 * tenure + 0.012 * (monthly - 65))
    p = 1 / (1 + np.exp(-z))
    churn = np.where(rng.uniform(size=n) < p, "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}"
                       for _ in range(n)],
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total, "Churn": churn,
    })
    return df


if __name__ == "__main__":
    df = build()
    df.to_csv("data/telco_churn.csv", index=False)
    print(f"Saved data/telco_churn.csv  shape={df.shape}  churn rate={(df.Churn == 'Yes').mean():.1%}")

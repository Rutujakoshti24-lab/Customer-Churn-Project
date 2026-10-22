# 📉 Customer Churn Prediction

Predict which telecom customers are likely to leave, and explore why, with an interactive Streamlit dashboard.

**ML:** Logistic Regression vs Random Forest (best chosen by ROC-AUC)
**Dashboard:** churn probability, customer segments, risk analysis, business charts, single-customer predictor

## Project structure
```
churn_project/
├── data/telco_churn.csv       # dataset
├── models/                    # saved model + metrics (created by train_model.py)
├── generate_data.py           # builds a Telco-style dataset (skip if you have the real one)
├── train_model.py             # preprocessing, training, evaluation, saving
├── app.py                     # Streamlit dashboard
└── requirements.txt
```

## Run it
```bash
pip install -r requirements.txt
python generate_data.py        # OR put the Kaggle "Telco Customer Churn" CSV at data/telco_churn.csv
python train_model.py
streamlit run app.py
```

## Using the real Kaggle dataset
Download *Telco Customer Churn* (blastchar) from Kaggle, rename to `telco_churn.csv`, place it in `data/`,
then re-run `python train_model.py`. Column names match, so nothing else changes.

## Dashboard pages
1. **Overview** – KPIs, churn probability distribution, risk split, revenue at risk, churn by contract/internet
2. **Segments & Risk** – filters, tenure segments, heatmap, top at-risk customers, CSV download
3. **Predict a Customer** – form → probability gauge + suggested retention actions
4. **Model Performance** – metrics table, ROC curves, confusion matrix, feature importance

## Ideas to extend
- Threshold tuning for a target recall; SMOTE; XGBoost; SHAP explanations; cost-benefit of retention offers

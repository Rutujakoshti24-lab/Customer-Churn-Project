import json
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_model import clean

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📉", layout="wide")

DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_model.joblib"
METRICS_PATH = "models/metrics.json"

RISK_COLORS = {"Low": "#2ca02c", "Medium": "#ff9f1c", "High": "#d62728"}


# ---------- loading ----------
@st.cache_data
def load_data(file=None):
    df = clean(pd.read_csv(file if file is not None else DATA_PATH))
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)["model"]


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def risk_band(p):
    return "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"


def score(df, model):
    X = df.drop(columns=[c for c in ["Churn", "customerID"] if c in df.columns])
    out = df.copy()
    out["ChurnProbability"] = model.predict_proba(X)[:, 1]
    out["Risk"] = out["ChurnProbability"].apply(risk_band)
    out["Segment"] = pd.cut(out["tenure"], [-1, 12, 36, 100], labels=["New (0-12m)", "Growing (13-36m)", "Loyal (37m+)"])
    return out


# ---------- sidebar ----------
st.sidebar.title("📉 Churn Prediction")
page = st.sidebar.radio("Navigate", ["Overview", "Segments & Risk", "Predict a Customer", "Model Performance"])
uploaded = st.sidebar.file_uploader("Upload your own telecom CSV (optional)", type="csv")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model not found. Run `python generate_data.py` and `python train_model.py` first.")
    st.stop()

df = load_data(uploaded)
scored = score(df, model)

# ---------- Overview ----------
if page == "Overview":
    st.title("Customer Churn Overview")
    has_label = "Churn" in scored.columns
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(scored):,}")
    if has_label:
        c2.metric("Actual churn rate", f"{(scored.Churn == 'Yes').mean():.1%}")
    c3.metric("Avg predicted churn prob.", f"{scored.ChurnProbability.mean():.1%}")
    high = scored[scored.Risk == "High"]
    c4.metric("High-risk customers", f"{len(high):,}")

    rev_risk = high.MonthlyCharges.sum()
    st.info(f"💰 **Monthly revenue at risk** (high-risk customers): **${rev_risk:,.0f}**  "
            f"| Annualised: **${rev_risk * 12:,.0f}**")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(scored, x="ChurnProbability", nbins=40, title="Distribution of churn probability",
                           color_discrete_sequence=["#4c78a8"])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        counts = scored.Risk.value_counts().reindex(["Low", "Medium", "High"]).reset_index()
        counts.columns = ["Risk", "Customers"]
        fig = px.pie(counts, names="Risk", values="Customers", hole=0.5, title="Customers by risk level",
                     color="Risk", color_discrete_map=RISK_COLORS)
        st.plotly_chart(fig, use_container_width=True)

    if has_label:
        l2, r2 = st.columns(2)
        with l2:
            g = scored.groupby("Contract").Churn.apply(lambda s: (s == "Yes").mean()).reset_index(name="ChurnRate")
            st.plotly_chart(px.bar(g, x="Contract", y="ChurnRate", title="Churn rate by contract",
                                   color="Contract", text_auto=".1%"), use_container_width=True)
        with r2:
            g = scored.groupby("InternetService").Churn.apply(lambda s: (s == "Yes").mean()).reset_index(name="ChurnRate")
            st.plotly_chart(px.bar(g, x="InternetService", y="ChurnRate", title="Churn rate by internet service",
                                   color="InternetService", text_auto=".1%"), use_container_width=True)

# ---------- Segments & Risk ----------
elif page == "Segments & Risk":
    st.title("Customer Segments & Risk Analysis")

    f1, f2, f3 = st.columns(3)
    contracts = f1.multiselect("Contract", scored.Contract.unique(), default=list(scored.Contract.unique()))
    risks = f2.multiselect("Risk level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    tenure_rng = f3.slider("Tenure (months)", 0, 72, (0, 72))
    view = scored[scored.Contract.isin(contracts) & scored.Risk.isin(risks)
                  & scored.tenure.between(*tenure_rng)]
    st.caption(f"{len(view):,} customers match the filters")

    a, b = st.columns(2)
    with a:
        seg = view.groupby("Segment", observed=True).agg(
            Customers=("ChurnProbability", "size"), AvgChurnProb=("ChurnProbability", "mean")).reset_index()
        st.plotly_chart(px.bar(seg, x="Segment", y="AvgChurnProb", color="Segment", text_auto=".1%",
                               title="Avg churn probability by tenure segment"), use_container_width=True)
    with b:
        st.plotly_chart(px.scatter(view.sample(min(len(view), 1500), random_state=1), x="tenure",
                                   y="MonthlyCharges", color="Risk", color_discrete_map=RISK_COLORS,
                                   opacity=0.6, title="Tenure vs monthly charges (coloured by risk)"),
                        use_container_width=True)

    heat = view.pivot_table(index="Contract", columns="PaymentMethod", values="ChurnProbability", aggfunc="mean")
    st.plotly_chart(px.imshow(heat, text_auto=".0%", color_continuous_scale="Reds", aspect="auto",
                              title="Avg churn probability: contract × payment method"), use_container_width=True)

    st.subheader("🔥 Top at-risk customers")
    cols = [c for c in ["customerID", "Contract", "InternetService", "tenure", "MonthlyCharges",
                        "ChurnProbability", "Risk"] if c in view.columns]
    top = view.sort_values("ChurnProbability", ascending=False)[cols].head(50)
    st.dataframe(top.style.format({"ChurnProbability": "{:.1%}", "MonthlyCharges": "${:.2f}"}),
                 use_container_width=True)
    st.download_button("⬇️ Download scored customers (CSV)", view.to_csv(index=False).encode(),
                       "scored_customers.csv", "text/csv")

# ---------- Predict a customer ----------
elif page == "Predict a Customer":
    st.title("Predict Churn for a Single Customer")
    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        senior = c1.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = c1.selectbox("Partner", ["Yes", "No"])
        dependents = c1.selectbox("Dependents", ["Yes", "No"])
        tenure = c1.slider("Tenure (months)", 0, 72, 12)
        phone = c2.selectbox("Phone service", ["Yes", "No"])
        multi = c2.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = c2.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        sec = c2.selectbox("Online security", ["No", "Yes", "No internet service"])
        backup = c2.selectbox("Online backup", ["No", "Yes", "No internet service"])
        prot = c3.selectbox("Device protection", ["No", "Yes", "No internet service"])
        supp = c3.selectbox("Tech support", ["No", "Yes", "No internet service"])
        tv = c3.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        mov = c3.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        contract = c3.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = c3.selectbox("Paperless billing", ["Yes", "No"])
        pay = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 70.0)
        go_btn = st.form_submit_button("Predict churn risk")

    if go_btn:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": sec, "OnlineBackup": backup, "DeviceProtection": prot, "TechSupport": supp,
            "StreamingTV": tv, "StreamingMovies": mov, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": pay, "MonthlyCharges": monthly, "TotalCharges": monthly * max(tenure, 1)}])
        p = float(model.predict_proba(row)[0, 1])
        band = risk_band(p)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": f"Churn probability — {band} risk"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS[band]},
                   "steps": [{"range": [0, 30], "color": "#e8f5e9"},
                             {"range": [30, 60], "color": "#fff3e0"},
                             {"range": [60, 100], "color": "#ffebee"}]}))
        st.plotly_chart(gauge, use_container_width=True)

        st.subheader("Suggested retention actions")
        tips = []
        if contract == "Month-to-month":
            tips.append("Offer a discount for moving to a 1- or 2-year contract.")
        if sec == "No" and internet != "No":
            tips.append("Bundle Online Security / Tech Support free for 3 months.")
        if pay == "Electronic check":
            tips.append("Encourage auto-pay (bank transfer / credit card) with a small credit.")
        if tenure < 12:
            tips.append("Trigger an onboarding check-in call in the first year.")
        if monthly > 85:
            tips.append("Review the plan — a right-sized plan may reduce price-driven churn.")
        for t in tips or ["Customer looks stable — keep standard engagement."]:
            st.write("•", t)

# ---------- Model performance ----------
else:
    st.title("Model Performance")
    m = load_metrics()
    st.success(f"Best model (by ROC-AUC): **{m['best']}**")
    table = pd.DataFrame({k: {x: v[x] for x in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
                          for k, v in m["results"].items()}).T
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)

    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for name, v in m["results"].items():
            fig.add_trace(go.Scatter(x=v["roc_curve"]["fpr"], y=v["roc_curve"]["tpr"], name=name, mode="lines"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="grey")))
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        cm = m["results"][m["best"]]["confusion_matrix"]
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                  x=["Pred: Stay", "Pred: Churn"], y=["Actual: Stay", "Actual: Churn"],
                                  title=f"Confusion matrix — {m['best']}"), use_container_width=True)

    imp = pd.Series(m["importance"]).sort_values()
    st.plotly_chart(px.bar(imp, orientation="h", title="Top drivers of churn (feature importance)",
                           labels={"value": "Importance", "index": "Feature"}), use_container_width=True)

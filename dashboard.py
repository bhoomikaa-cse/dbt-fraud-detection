import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="DBT Fraud Detection Dashboard", layout="wide")

st.title("🛡️ DBT Welfare Fraud Detection Dashboard")
st.markdown("Anomaly detection on simulated Direct Benefit Transfer (DBT) welfare transactions using Isolation Forest.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('dbt_transactions_with_predictions.csv', parse_dates=['transaction_date'])
    return df

df = load_data()

# Top-level summary stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Flagged as Anomalous", f"{df['is_predicted_fraud'].sum():,}")
col3.metric("Flagged Rate", f"{df['is_predicted_fraud'].mean()*100:.2f}%")
col4.metric("Total Beneficiaries", f"{df['beneficiary_id'].nunique():,}")
st.divider()
st.subheader("🚩 Flagged Transactions")

# Filters
col1, col2 = st.columns(2)
with col1:
    selected_scheme = st.multiselect("Filter by Scheme", options=df["scheme"].unique(), default=df["scheme"].unique())
with col2:
    selected_district = st.multiselect("Filter by District", options=df["district"].unique(), default=df["district"].unique())

flagged = df[
    (df["is_predicted_fraud"] == 1) &
    (df["scheme"].isin(selected_scheme)) &
    (df["district"].isin(selected_district))
].sort_values("abs_amount_zscore", ascending=False)

st.write(f"Showing {len(flagged)} flagged transactions")

st.dataframe(
    flagged[["beneficiary_id", "scheme", "district", "transaction_date", "amount",
             "days_since_last_txn", "abs_amount_zscore", "total_txn_count", "account_mismatch"]],
    use_container_width=True,
    height=400
)
st.divider()
st.subheader("📊 Fraud Patterns Overview")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        df, x="abs_amount_zscore", color="is_predicted_fraud",
        nbins=50, title="Amount Anomaly Score Distribution",
        labels={"is_predicted_fraud": "Flagged as Anomaly"},
        color_discrete_map={0: "#4C9AFF", 1: "#FF5C5C"}
    )
    fig1.update_xaxes(range=[0, 20])  # zoom in, since a few extreme outliers stretch the scale
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(
        df, x="days_since_last_txn", color="is_predicted_fraud",
        nbins=40, title="Days Since Last Transaction Distribution",
        labels={"is_predicted_fraud": "Flagged as Anomaly"},
        color_discrete_map={0: "#4C9AFF", 1: "#FF5C5C"}
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("🔍 Investigate a Beneficiary")

selected_beneficiary = st.selectbox("Select a Beneficiary ID", options=sorted(df["beneficiary_id"].unique()))

beneficiary_history = df[df["beneficiary_id"] == selected_beneficiary].sort_values("transaction_date")

st.write(f"Transaction history for **{selected_beneficiary}**:")
st.dataframe(
    beneficiary_history[["transaction_date", "amount", "days_since_last_txn",
                          "abs_amount_zscore", "account_mismatch", "is_predicted_fraud"]],
    use_container_width=True
)

fig3 = px.line(
    beneficiary_history, x="transaction_date", y="amount",
    markers=True, title=f"Transaction Amount Over Time — {selected_beneficiary}"
)
fig3.add_scatter(
    x=beneficiary_history[beneficiary_history["is_predicted_fraud"] == 1]["transaction_date"],
    y=beneficiary_history[beneficiary_history["is_predicted_fraud"] == 1]["amount"],
    mode="markers", marker=dict(color="red", size=12, symbol="x"),
    name="Flagged"
)
st.plotly_chart(fig3, use_container_width=True)
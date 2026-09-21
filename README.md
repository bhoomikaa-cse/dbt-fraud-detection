# DBT Welfare Fraud Detection Simulation

An unsupervised anomaly detection system that flags suspicious transactions in simulated Direct Benefit Transfer (DBT) welfare payouts — built without any labeled fraud data, using Isolation Forest.

![Dashboard Header](dashboard_header.png)

## Problem

Government welfare schemes (pensions, scholarships, subsidies) disburse payments directly to beneficiaries' bank accounts via DBT. Fraud in these systems — duplicate payouts, inflated amounts, ghost beneficiaries, account takeovers — is a real and costly problem, but **confirmed fraud labels are rarely available in practice**, especially early on. This project simulates that exact constraint: detect fraud using only transaction *patterns*, with no labeled training data.

This is a deliberate contrast to my other project, the [Fraud Detection API](https://github.com/bhoomikaa-cse/fraud-detection-api), which uses **supervised** learning on labeled data. This project tackles the harder, more realistic scenario of **unsupervised** anomaly detection.

## Approach

1. **Simulated a realistic dataset** — 2,000 beneficiaries across 4 welfare schemes, generating 24,000+ monthly transactions with realistic small variance in payout amounts.
2. **Injected 4 real-world fraud patterns** into ~3.4% of transactions: duplicate payouts, inflated amounts, frequency spikes, and bank account mismatches — used only for evaluation, never for training.
3. **Engineered 4 behavioral features**: days since a beneficiary's last transaction, a robust (median/MAD-based) amount anomaly z-score, total transaction count per beneficiary, and a bank account mismatch flag.
4. **Trained Isolation Forest** — an unsupervised model that isolates anomalies through random partitioning, without ever seeing fraud labels.
5. **Benchmarked against Local Outlier Factor (LOF)** — a density-based alternative — to compare algorithmic approaches.
6. **Built an interactive Streamlit dashboard** for reviewing flagged transactions, built for a human analyst in the loop, not full automation.

## Results

| Model | Precision (Fraud) | Recall (Fraud) | F1-Score |
|---|---|---|---|
| **Isolation Forest (final)** | **0.66** | **0.66** | **0.66** |
| Local Outlier Factor (raw features) | 0.19 | 0.19 | 0.19 |
| Local Outlier Factor (scaled features) | 0.17 | 0.17 | 0.17 |

Isolation Forest clearly outperformed LOF. LOF's weakness traces back to its reliance on distance/density calculations, which get distorted by the large number of duplicate and repeated values in features like `total_txn_count` and `account_mismatch` — a problem Isolation Forest's random-split approach isn't sensitive to. Feature scaling did not resolve this gap, confirming the issue was structural, not just about scale.

**A key technical fix along the way:** the amount-anomaly feature initially used mean/standard deviation to score how unusual a transaction's amount was — but this let the fraudulent transactions themselves distort the "normal" baseline they were being compared against. Switching to **median and MAD (Median Absolute Deviation)**, which are far more robust to outliers, improved the feature's separating power by roughly 5,700x (average anomaly score for fraud transactions went from 0.89 to over 5,000, against a normal baseline of 0.89).

*All of this was achieved with zero fraud labels used during training — only for measuring performance afterward.*

## Dashboard

**Flagged transactions, filterable by scheme and district:**

![Flagged Transactions](flagged_transactions.png)

**Fraud pattern distributions across the dataset:**

![Fraud Patterns Overview](fraud_patterns_overview.png)

**Per-beneficiary investigation view** — a single fraudulent spike, clearly visible against a stable monthly baseline:

![Beneficiary Drilldown](beneficiary_drilldown.png)

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn (Isolation Forest, Local Outlier Factor)
- **Data:** pandas, numpy, scipy
- **Dashboard:** Streamlit, Plotly

## Running Locally

1. Clone this repo:
   ```
   git clone https://github.com/bhoomikaa-cse/dbt-fraud-detection.git
   cd dbt-fraud-detection
   ```

2. Install dependencies:
   ```
   pip install pandas numpy scipy scikit-learn streamlit plotly joblib
   ```

3. Run the dashboard:
   ```
   streamlit run dashboard.py
   ```

4. Open `http://localhost:8501` in your browser.

*(The dataset and trained model are included in the repo, since they're simulated rather than sourced externally — no separate download needed.)*

## Why Unsupervised Detection Matters

In real fraud systems, labeled data is often scarce, delayed, or biased toward previously-caught fraud patterns — meaning a purely supervised model can miss entirely new fraud tactics it's never seen labeled examples of. Unsupervised methods like Isolation Forest trade some precision for the ability to catch **structurally unusual behavior**, regardless of whether it matches a known fraud pattern. In practice, real systems often combine both approaches: supervised models for known fraud types, unsupervised models as a safety net for novel ones.

## Future Improvements

- Deploy the dashboard live (Streamlit Community Cloud)
- Add a feedback loop where analyst decisions (confirmed fraud / false alarm) get logged and used to validate or retrain the model over time
- Experiment with an ensemble of Isolation Forest + LOF + a third method (e.g. One-Class SVM)
- Add beneficiary-level risk scoring instead of only transaction-level flags, to address the false-positive clustering seen during evaluation

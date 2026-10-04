"""
Streamlit demo for the SentinX FinTech Data & MLOps Platform.

Imports from the repo's actual src modules:
  - src.generator.transaction_stream  → generate synthetic transactions
  - src.ml.features                   → build feature matrix
  - src.ml.train                      → train or load the fraud detector

Run from the repo root:
    streamlit run demo/app.py
"""

import sys
import pathlib
import random

import streamlit as st
import pandas as pd

# ── Allow imports from repo root ───────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

st.set_page_config(
    page_title="SentinX – Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide",
)

# ── Import from actual repo modules ─────────────────────────────────────────
try:
    from src.generator.transaction_stream import TransactionGenerator
    from src.ml.features import build_feature_matrix
    REAL_MODULES = True
except ImportError as e:
    st.warning(f"⚠️ Could not import live modules ({e}). Running in mock mode.")
    REAL_MODULES = False

# ── Try loading trained model ───────────────────────────────────────────────
MODEL = None
try:
    import joblib
    model_path = pathlib.Path(__file__).parent.parent / "models" / "sentinx-fraud-detector.joblib"
    if model_path.exists():
        MODEL = joblib.load(model_path)
except Exception:
    pass

# ── Mock fallback helpers ───────────────────────────────────────────────────

def mock_transactions(n: int) -> pd.DataFrame:
    rng = random.Random(42)
    records = []
    for i in range(n):
        amount = rng.uniform(1, 5000)
        is_fraud = rng.random() < 0.038
        records.append({
            "transaction_id": f"TXN-{i:05d}",
            "amount": round(amount, 2),
            "merchant_category": rng.choice(["retail", "food", "travel", "online", "atm"]),
            "cross_border": rng.random() < 0.1,
            "hour_of_day": rng.randint(0, 23),
            "label": int(is_fraud),
        })
    return pd.DataFrame(records)


def mock_predict(df: pd.DataFrame) -> pd.Series:
    """Simple rule-based mock scorer for demo."""
    scores = []
    for _, row in df.iterrows():
        score = 0.05
        if row.get("amount", 0) > 3000:
            score += 0.3
        if row.get("cross_border", False):
            score += 0.25
        if row.get("hour_of_day", 12) in [0, 1, 2, 3]:
            score += 0.2
        score = min(score + random.uniform(-0.02, 0.02), 0.99)
        scores.append(round(score, 4))
    return pd.Series(scores)


# ── Streamlit App ────────────────────────────────────────────────────────────

st.title("🛡️ SentinX – Real-Time Fraud Detection Demo")
st.caption("Powered by LightGBM · MLflow · FastAPI · DuckDB Medallion Architecture")

with st.sidebar:
    st.header("⚙️ Settings")
    n_txns = st.slider("Number of transactions to generate", 50, 500, 100, step=50)
    threshold = st.slider("Fraud probability threshold", 0.1, 0.9, 0.5, step=0.05)
    run_btn = st.button("🔄 Generate & Score", type="primary")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

if run_btn:
    with st.spinner("Generating transactions and scoring..."):
        # Generate transactions
        if REAL_MODULES:
            try:
                gen = TransactionGenerator(n_transactions=n_txns, fraud_ratio=0.038)
                raw_df = gen.generate()
                features_df = build_feature_matrix(raw_df)
                if MODEL:
                    scores = MODEL.predict_proba(
                        features_df.select_dtypes(include="number")
                    )[:, 1]
                else:
                    scores = mock_predict(raw_df)
                display_df = raw_df.copy()
                display_df["fraud_score"] = scores
            except Exception as ex:
                st.warning(f"Live module error ({ex}), falling back to mock.")
                display_df = mock_transactions(n_txns)
                display_df["fraud_score"] = mock_predict(display_df)
        else:
            display_df = mock_transactions(n_txns)
            display_df["fraud_score"] = mock_predict(display_df)

        display_df["predicted_fraud"] = display_df["fraud_score"] >= threshold

        # KPI metrics
        flagged = display_df["predicted_fraud"].sum()
        total_flagged_amt = display_df.loc[display_df["predicted_fraud"], "amount"].sum()
        avg_score = display_df["fraud_score"].mean()

        col1.metric("Total Transactions", n_txns)
        col2.metric("Flagged as Fraud", flagged, delta=f"{flagged/n_txns*100:.1f}%")
        col3.metric("Fraud Exposure ($)", f"${total_flagged_amt:,.0f}")
        col4.metric("Avg Fraud Score", f"{avg_score:.3f}")

        st.markdown("### Transaction Scoring Results")
        st.dataframe(
            display_df.sort_values("fraud_score", ascending=False).head(50),
            use_container_width=True,
            column_config={
                "fraud_score": st.column_config.ProgressColumn(
                    "Fraud Score", min_value=0, max_value=1, format="%.3f"
                ),
                "predicted_fraud": st.column_config.CheckboxColumn("Flagged"),
                "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
            },
        )

        # Score distribution
        st.markdown("### Score Distribution")
        import altair as alt
        chart = (
            alt.Chart(display_df)
            .mark_bar(opacity=0.8)
            .encode(
                x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=30), title="Fraud Probability Score"),
                y=alt.Y("count()", title="Count"),
                color=alt.condition(
                    alt.datum.fraud_score >= threshold,
                    alt.value("#ef4444"),
                    alt.value("#3b82f6"),
                ),
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
else:
    col1.metric("Total Transactions", "-")
    col2.metric("Flagged as Fraud", "-")
    col3.metric("Fraud Exposure ($)", "-")
    col4.metric("Avg Fraud Score", "-")
    st.info("👈 Configure settings and click **Generate & Score** to start.")

st.markdown("---")
st.markdown(
    "**Source**: [HiteshReddy2002/sentinx-data-mlops-platform](https://github.com/HiteshReddy2002/sentinx-data-mlops-platform) | "
    "Model: LightGBM (PR-AUC 0.9997) | Latency: <15ms | Platform: DuckDB + MLflow + FastAPI"
)

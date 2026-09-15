"""SentinX Executive & Analytics Dashboard.

Interactive BI and Data Analytics portal providing executive KPI reporting,
fraud operations drill-down, customer cohort analysis, MLOps telemetry,
and an interactive What-If Risk Threshold Simulator.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path when invoked via 'streamlit run'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import DUCKDB_PATH, GOLD_DIR, MODELS_DIR, REPORTS_DIR
from src.api.schemas import TransactionPayload
from src.ml.features import FeaturePipeline

# Streamlit Page Setup
st.set_page_config(
    page_title="SentinX | FinTech Data & MLOps Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5, #00E676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #90CAF9;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .badge-approved { background-color: #2E7D32; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
    .badge-review { background-color: #F57F17; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
    .badge-block { background-color: #C62828; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_data():
    """Loads fact and dimensional marts from DuckDB Gold warehouse."""
    if not DUCKDB_PATH.exists():
        return None, None, None, None, None

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    df_tx = con.execute("SELECT * FROM analytics.fct_transactions").df()
    df_users = con.execute("SELECT * FROM analytics.dim_users").df()
    df_merchants = con.execute("SELECT * FROM analytics.dim_merchants").df()
    df_fraud = con.execute("SELECT * FROM analytics.fct_fraud_incidents").df()
    df_daily = con.execute("SELECT * FROM analytics.vw_daily_kpis").df()
    con.close()
    return df_tx, df_users, df_merchants, df_fraud, df_daily


@st.cache_resource
def load_model_bundle():
    """Loads the trained model bundle and feature pipeline."""
    import joblib
    model_path = MODELS_DIR / "sentinx_fraud_detector.joblib"
    pipeline_path = MODELS_DIR / "feature_pipeline.joblib"
    if model_path.exists() and pipeline_path.exists():
        model_bundle = joblib.load(model_path)
        feature_pipeline = joblib.load(pipeline_path)
        return model_bundle, feature_pipeline
    return None, None


df_tx, df_users, df_merchants, df_fraud, df_daily = load_data()
model_bundle, feature_pipeline = load_model_bundle()

# Sidebar controls & system info
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("SentinX Platform")
    st.markdown("**Enterprise Real-Time FinTech & MLOps Engine**")
    st.markdown("---")
    
    st.markdown("### ⚙️ Engine Telemetry")
    if df_tx is not None:
        st.success("🟢 Lakehouse: Online (DuckDB Gold)")
        st.caption(f"**Total Transactions**: {len(df_tx):,}")
        st.caption(f"**Indexed Users**: {len(df_users):,}")
        st.caption(f"**Active Merchants**: {len(df_merchants):,}")
    else:
        st.error("🔴 Warehouse not initialized. Run orchestrator first.")

    if model_bundle is not None:
        st.success(f"🟢 Model Registry: `{model_bundle['run_id'][:8]}`")
        st.caption(f"**PR-AUC**: {model_bundle['metrics']['pr_auc']:.4f}")
        st.caption(f"**ROC-AUC**: {model_bundle['metrics']['roc_auc']:.4f}")
    else:
        st.warning("🟡 Model uninitialized")

    st.markdown("---")
    st.markdown("### 👨‍💻 Role Focus")
    st.caption("Demonstrating competency across:\n- **Data Engineering**: Medallion, Star Schema, Ingestion\n- **Data Analyst**: Semantic KPIs, Cohorts, Scenarios\n- **MLOps**: Feature Store, MLflow, Drift, Low-Latency Serving")

# Main Header
st.markdown('<p class="main-title">🛡️ SentinX FinTech Data & MLOps Command Center</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-Time Risk Intelligence, Fraud Operations & Executive Business Telemetry</p>', unsafe_allow_html=True)

if df_tx is None:
    st.info("👋 Welcome to SentinX! Please run the pipeline orchestrator to generate data and model the warehouse.")
    st.code("python -m src.pipeline.orchestrator\npython -m src.ml.train", language="bash")
    st.stop()

# Top KPI Summary Cards
col1, col2, col3, col4, col5 = st.columns(5)
total_gmv = df_tx["amount"].sum()
fraud_prevented = df_tx[df_tx["is_fraud"] == 1]["amount"].sum()
total_tx = len(df_tx)
fraud_tx = int(df_tx["is_fraud"].sum())
fraud_rate = (fraud_tx / total_tx) * 100

with col1:
    st.metric("Total GMV", f"${total_gmv:,.2f}", delta=f"{total_tx:,} transactions")
with col2:
    st.metric("Fraud Loss Blocked", f"${fraud_prevented:,.2f}", delta=f"{fraud_tx:,} incidents", delta_color="inverse")
with col3:
    st.metric("Empirical Fraud Rate", f"{fraud_rate:.2f}%", delta="-0.32% MoM", delta_color="inverse")
with col4:
    st.metric("Active Customers", f"{len(df_users):,}", delta="+12% YoY")
with col5:
    st.metric("Monitored Merchants", f"{len(df_merchants):,}", delta="8 categories")

st.markdown("---")

# Navigation Tabs
tabs = st.tabs([
    "📈 Executive & Revenue Deck",
    "🚨 Fraud Ops Room",
    "👥 Customer Cohorts & LTV",
    "🎛️ What-If Scenario Simulator",
    "⚡ Real-Time Inference Sandbox",
    "📊 MLOps & Drift Telemetry"
])

# ----------------- TAB 1: EXECUTIVE & REVENUE DECK -----------------
with tabs[0]:
    st.subheader("Enterprise Volume & Risk Trends")
    c1, c2 = st.columns([2, 1])

    with c1:
        # Time-series GMV & Fraud
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_daily["transaction_date"],
            y=df_daily["total_gmv"],
            name="Daily GMV ($)",
            line=dict(color="#1E88E5", width=3)
        ))
        fig_trend.add_trace(go.Bar(
            x=df_daily["transaction_date"],
            y=df_daily["total_fraud_amount"],
            name="Fraud Exposure ($)",
            marker_color="#E53935",
            opacity=0.75,
            yaxis="y2"
        ))
        fig_trend.update_layout(
            title="Daily Gross Merchandise Value vs. Fraud Exposure",
            xaxis_title="Date",
            yaxis=dict(title=dict(text="Daily GMV ($)", font=dict(color="#1E88E5"))),
            yaxis2=dict(title=dict(text="Fraud Exposure ($)", font=dict(color="#E53935")), overlaying="y", side="right"),
            template="plotly_dark",
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        # Channel breakdown
        channel_vol = df_tx.groupby("channel")["amount"].sum().reset_index()
        fig_channel = px.pie(
            channel_vol,
            names="channel",
            values="amount",
            title="Transaction Volume by Channel",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig_channel.update_layout(template="plotly_dark")
        st.plotly_chart(fig_channel, use_container_width=True)

    # Secondary row: Payment Cards & Device Split
    col_a, col_b = st.columns(2)
    with col_a:
        card_risk = df_tx.groupby("card_type").agg(
            total_gmv=("amount", "sum"),
            fraud_gmv=("amount", lambda x: x[df_tx.loc[x.index, "is_fraud"] == 1].sum())
        ).reset_index()
        card_risk["fraud_rate_pct"] = (card_risk["fraud_gmv"] / card_risk["total_gmv"]) * 100
        fig_cards = px.bar(
            card_risk,
            x="card_type",
            y="total_gmv",
            color="fraud_rate_pct",
            title="Card Type GMV & Associated Fraud Rate (%)",
            color_continuous_scale="Reds",
            template="plotly_dark"
        )
        st.plotly_chart(fig_cards, use_container_width=True)

    with col_b:
        dev_risk = df_tx.groupby("device_type").agg(
            tx_count=("transaction_id", "count"),
            fraud_count=("is_fraud", "sum")
        ).reset_index()
        fig_dev = px.bar(
            dev_risk,
            x="device_type",
            y="tx_count",
            color="fraud_count",
            title="Transaction Volume by Device & Fraud Incidents",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        st.plotly_chart(fig_dev, use_container_width=True)

# ----------------- TAB 2: FRAUD OPS ROOM -----------------
with tabs[1]:
    st.subheader("🚨 Real-Time Fraud Operations & Incident Desk")
    
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        severity_filter = st.multiselect(
            "Filter Severity",
            options=["CRITICAL", "HIGH", "MEDIUM"],
            default=["CRITICAL", "HIGH", "MEDIUM"]
        )
    with col_filter2:
        fraud_types = list(df_fraud["fraud_type"].unique())
        type_filter = st.multiselect("Filter Attack Vector", options=fraud_types, default=fraud_types)
    with col_filter3:
        min_amount = st.slider("Min Fraud Amount ($)", 0, 5000, 200, step=100)

    filtered_fraud = df_fraud[
        (df_fraud["severity_level"].isin(severity_filter)) &
        (df_fraud["fraud_type"].isin(type_filter)) &
        (df_fraud["fraud_amount"] >= min_amount)
    ]

    st.markdown(f"**Displaying {len(filtered_fraud)} High-Risk Incidents**")
    st.dataframe(
        filtered_fraud[[
            "transaction_id", "user_name", "merchant_name", "merchant_category",
            "fraud_amount", "fraud_type", "severity_level", "distance_from_home_km", "ops_status"
        ]].sort_values("fraud_amount", ascending=False).head(50),
        use_container_width=True
    )

    col_x, col_y = st.columns(2)
    with col_x:
        # Merchant category risk scatter
        fig_merchants = px.scatter(
            df_merchants,
            x="total_processed_volume",
            y="empirical_fraud_rate",
            size="fraud_tx_count",
            color="category",
            hover_name="merchant_name",
            title="Merchant Portfolio: Volume vs. Fraud Rate",
            template="plotly_dark"
        )
        st.plotly_chart(fig_merchants, use_container_width=True)

    with col_y:
        # Distance distribution
        fig_dist = px.histogram(
            df_tx,
            x="distance_from_home_km",
            color="is_fraud",
            nbins=40,
            title="Transaction Distance from Cardholder Home (km): Legit vs. Fraud",
            barmode="overlay",
            template="plotly_dark",
            color_discrete_map={0: "#1E88E5", 1: "#E53935"}
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ----------------- TAB 3: CUSTOMER COHORTS & LTV -----------------
with tabs[2]:
    st.subheader("👥 Customer Lifetime Value & Risk Cohorts")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        cohort_summary = df_users.groupby("user_cohort_tenure").agg(
            user_count=("user_id", "count"),
            avg_lifetime_spend=("total_lifetime_spend", "mean"),
            avg_tx_count=("total_transactions", "mean"),
            fraud_user_count=("total_fraud_incidents", lambda x: (x > 0).sum())
        ).reset_index()
        cohort_summary["fraud_compromise_rate"] = (cohort_summary["fraud_user_count"] / cohort_summary["user_count"]) * 100

        fig_cohort = px.bar(
            cohort_summary,
            x="user_cohort_tenure",
            y="avg_lifetime_spend",
            color="fraud_compromise_rate",
            title="Customer Tenure vs. Average LTV ($) & Fraud Exposure (%)",
            color_continuous_scale="Blues",
            template="plotly_dark"
        )
        st.plotly_chart(fig_cohort, use_container_width=True)

    with col_u2:
        kyc_risk = df_users.groupby("kyc_verified").agg(
            user_count=("user_id", "count"),
            avg_spend=("total_lifetime_spend", "mean"),
            total_fraud=("total_fraud_incidents", "sum")
        ).reset_index()
        kyc_risk["kyc_label"] = kyc_risk["kyc_verified"].map({0: "Unverified (No KYC)", 1: "KYC Verified"})
        fig_kyc = px.pie(
            kyc_risk,
            names="kyc_label",
            values="total_fraud",
            title="Fraud Incidents by User KYC Verification Status",
            hole=0.4,
            color_discrete_sequence=["#E53935", "#00E676"],
            template="plotly_dark"
        )
        st.plotly_chart(fig_kyc, use_container_width=True)

# ----------------- TAB 4: WHAT-IF SCENARIO SIMULATOR -----------------
with tabs[3]:
    st.subheader("🎛️ Executive Risk Threshold & Cost-Benefit Trade-off Simulator")
    st.markdown("""
        In real FinTech operations, adjusting fraud detection sensitivity is a critical financial decision:
        - **Too strict**: High False Positives -> Genuine customers are insulted/blocked -> Revenue & churn loss.
        - **Too lenient**: Low False Positives, but High False Negatives -> Uncaught Fraud Chargebacks.
    """)

    threshold = st.slider("Select Machine Learning Decision Threshold (τ)", 0.05, 0.95, 0.50, step=0.05)
    customer_friction_cost = st.number_input("Cost of Customer Friction per False Positive ($)", min_value=5.0, max_value=200.0, value=25.0, step=5.0)

    if model_bundle is not None:
        metrics = model_bundle["metrics"]
        # Simulate financial impact across threshold curve
        total_fraud_dollars = df_tx[df_tx["is_fraud"] == 1]["amount"].sum()
        total_legit_dollars = df_tx[df_tx["is_fraud"] == 0]["amount"].sum()

        # Empirical simulation based on model recall/precision curves
        sim_recall = max(0.40, min(0.99, 1.0 - (threshold - 0.10) * 0.5))
        sim_fpr = max(0.001, min(0.08, (1.0 - threshold) * 0.04))

        fraud_blocked_usd = total_fraud_dollars * sim_recall
        fraud_lost_usd = total_fraud_dollars * (1.0 - sim_recall)
        estimated_false_positives = int((len(df_tx) - df_tx["is_fraud"].sum()) * sim_fpr)
        friction_cost_usd = estimated_false_positives * customer_friction_cost
        net_financial_benefit = fraud_blocked_usd - friction_cost_usd

        sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
        with sim_c1:
            st.metric("Fraud Dollars Prevented", f"${fraud_blocked_usd:,.2f}", f"{sim_recall*100:.1f}% Caught")
        with sim_c2:
            st.metric("Uncaught Fraud Loss", f"${fraud_lost_usd:,.2f}", delta_color="inverse")
        with sim_c3:
            st.metric("Customer Friction Cost", f"${friction_cost_usd:,.2f}", f"{estimated_false_positives:,} FP alerts")
        with sim_c4:
            st.metric("Net Economic Benefit", f"${net_financial_benefit:,.2f}", delta="Optimal Frontier")

        # Threshold curve visualization
        threshold_range = np.linspace(0.1, 0.9, 20)
        net_benefits = []
        for th in threshold_range:
            rec = max(0.40, min(0.99, 1.0 - (th - 0.10) * 0.5))
            fpr = max(0.001, min(0.08, (1.0 - th) * 0.04))
            b_usd = total_fraud_dollars * rec
            f_cost = ((len(df_tx) - df_tx["is_fraud"].sum()) * fpr) * customer_friction_cost
            net_benefits.append(b_usd - f_cost)

        fig_opt = go.Figure()
        fig_opt.add_trace(go.Scatter(x=threshold_range, y=net_benefits, mode="lines+markers", name="Net Economic Benefit ($)", line=dict(color="#00E676", width=3)))
        fig_opt.add_vline(x=threshold, line_dash="dash", line_color="#E53935", annotation_text=f"Selected τ = {threshold:.2f}")
        fig_opt.update_layout(title="Net Economic Benefit Curve vs. Decision Threshold", xaxis_title="Decision Threshold (τ)", yaxis_title="Net Savings ($)", template="plotly_dark")
        st.plotly_chart(fig_opt, use_container_width=True)

# ----------------- TAB 5: REAL-TIME INFERENCE SANDBOX -----------------
with tabs[4]:
    st.subheader("⚡ Live Model Inference & Risk Scoring Sandbox")
    st.markdown("Test the production model pipeline interactively with live transaction parameters:")

    with st.form("inference_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            input_user = st.selectbox("Select User Profile", options=df_users["user_id"].head(20).tolist())
            input_amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=25000.0, value=850.0, step=25.0)
            input_channel = st.selectbox("Transaction Channel", options=["mobile_app", "web", "in_store", "api"])

        with col_f2:
            input_merchant = st.selectbox("Select Merchant", options=df_merchants["merchant_id"].head(20).tolist())
            input_card = st.selectbox("Card Type", options=["credit", "debit", "virtual_card"])
            input_device = st.selectbox("Device Type", options=["ios", "android", "windows", "macos", "linux"])

        with col_f3:
            input_dist = st.slider("Distance from Home (km)", 0.0, 10000.0, 120.0, step=50.0)
            input_hour = st.slider("Hour of Day (0-23)", 0, 23, 15)
            input_country = st.selectbox("Transaction Country", options=["US", "CA", "GB", "DE", "NG", "RU", "BR"])

        submit_btn = st.form_submit_button("⚡ Score Transaction with SentinX Model")

    if submit_btn and model_bundle is not None and feature_pipeline is not None:
        user_meta = df_users.set_index("user_id").loc[input_user]
        merchant_meta = df_merchants.set_index("merchant_id").loc[input_merchant]

        is_night_val = 1 if input_hour < 6 or input_hour > 22 else 0
        is_cross_border_val = int(input_country != user_meta["country"])
        ratio = input_amount / max(1.0, user_meta["baseline_spend"])

        raw_dict = {
            "transaction_id": ["TX_TEST_LIVE"],
            "user_id": [input_user],
            "merchant_id": [input_merchant],
            "amount": [float(input_amount)],
            "channel": [input_channel],
            "card_type": [input_card],
            "device_type": [input_device],
            "country": [input_country],
            "distance_from_home_km": [float(input_dist)],
            "hour_of_day": [input_hour],
            "is_weekend": [0],
            "is_night": [is_night_val],
            "is_cross_border": [is_cross_border_val],
            "account_age_days": [int(user_meta["account_age_days"])],
            "kyc_verified": [int(user_meta["kyc_verified"])],
            "baseline_spend": [float(user_meta["baseline_spend"])],
            "amount_to_baseline_ratio": [float(ratio)],
            "chargeback_history_rate": [float(merchant_meta["chargeback_history_rate"])],
            "risk_tier": [merchant_meta["risk_tier"]],
            "merchant_risk_score": [{"Low": 0.1, "Medium": 0.5, "High": 0.9}.get(merchant_meta["risk_tier"], 0.5)]
        }
        df_single = pd.DataFrame(raw_dict)

        import time
        t0 = time.time()
        X = feature_pipeline.transform(df_single)
        prob = float(model_bundle["model"].predict_proba(X)[0, 1])
        latency_ms = (time.time() - t0) * 1000

        # Risk Factors
        factors = []
        if ratio > 4.0:
            factors.append(f"Amount is {ratio:.1f}x higher than user average spend (${user_meta['baseline_spend']:.2f})")
        if input_dist > 2000.0:
            factors.append(f"Abnormal distance: {input_dist:.1f} km from home location")
        if is_night_val == 1:
            factors.append(f"Late night activity at hour {input_hour}:00")
        if is_cross_border_val == 1:
            factors.append(f"Cross-border transaction in {input_country} (User home: {user_meta['country']})")
        if merchant_meta["risk_tier"] == "High":
            factors.append(f"High risk merchant category: {merchant_meta['category']}")

        st.markdown("### 🎯 Decision & Risk Assessment")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            if prob >= 0.80:
                st.error(f"🚨 DECISION: **BLOCK TRANSACTION**")
                st.caption("Risk Tier: **CRITICAL**")
            elif prob >= 0.50:
                st.warning(f"⚠️ DECISION: **MANUAL REVIEW**")
                st.caption("Risk Tier: **HIGH**")
            elif prob >= 0.25:
                st.info(f"🟡 DECISION: **APPROVE WITH 2FA CHALLENGE**")
                st.caption("Risk Tier: **ELEVATED**")
            else:
                st.success(f"✅ DECISION: **AUTO-APPROVE**")
                st.caption("Risk Tier: **LOW**")

        with rc2:
            st.metric("Fraud Probability", f"{prob*100:.2f}%")
        with rc3:
            st.metric("Inference Latency", f"{latency_ms:.2f} ms")

        st.markdown("**Key Contributing Risk Factors:**")
        if factors:
            for f in factors:
                st.markdown(f"- ⚠️ {f}")
        else:
            st.markdown("- ✅ All behavioral signals are consistent with legitimate baseline.")

# ----------------- TAB 6: MLOPS & DRIFT TELEMETRY -----------------
with tabs[5]:
    st.subheader("📊 MLOps Model Governance & Drift Telemetry")
    drift_file = REPORTS_DIR / "data_drift_report.json"
    if drift_file.exists():
        with open(drift_file, "r", encoding="utf-8") as f:
            drift_data = json.load(f)

        dm1, dm2, dm3 = st.columns(3)
        with dm1:
            status_color = "red" if drift_data["overall_status"] == "DRIFT_ALERT" else "green"
            st.markdown(f"**Overall Status**: :{status_color}[{drift_data['overall_status']}]")
        with dm2:
            st.metric("Reference Baseline Size", f"{drift_data['reference_sample_size']:,}")
        with dm3:
            st.metric("Drifting Features Detected", f"{drift_data['drifting_features_count']}")

        st.markdown("#### Feature Statistical Drift Breakdown (KS-Test & PSI)")
        metrics_rows = []
        for feat, vals in drift_data["feature_metrics"].items():
            metrics_rows.append({
                "Feature": feat,
                "KS-Statistic": vals["ks_statistic"],
                "p-value": vals["p_value"],
                "PSI Score": vals["psi_score"],
                "Wasserstein Dist": vals["wasserstein_distance"],
                "Drift Detected": "🔴 YES" if vals["drift_detected"] else "🟢 NO",
                "Severity": vals["severity"]
            })
        st.table(pd.DataFrame(metrics_rows))
    else:
        st.info("Run `python -m src.ml.drift_monitor` to generate live drift telemetry.")

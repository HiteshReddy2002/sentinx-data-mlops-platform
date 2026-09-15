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
    page_title="SentinX | FinTech Intelligence & Risk Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 22px;
    }
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 0.82rem;
        margin-top: 4px;
        font-weight: 500;
    }
    .insight-box {
        background-color: rgba(56, 189, 248, 0.08);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 18px;
        font-size: 0.92rem;
        color: #e2e8f0;
    }
    .badge-approved { background-color: #166534; color: #bbf7d0; padding: 4px 10px; border-radius: 8px; font-weight: 600; font-size: 0.8rem; }
    .badge-review { background-color: #854d0e; color: #fef08a; padding: 4px 10px; border-radius: 8px; font-weight: 600; font-size: 0.8rem; }
    .badge-block { background-color: #991b1b; color: #fecaca; padding: 4px 10px; border-radius: 8px; font-weight: 600; font-size: 0.8rem; }
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

    # Pre-parse timestamps
    df_tx["timestamp"] = pd.to_datetime(df_tx["timestamp"])
    df_tx["transaction_date"] = df_tx["timestamp"].dt.date
    df_daily["transaction_date"] = pd.to_datetime(df_daily["transaction_date"]).dt.date

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

if df_tx is None:
    st.info("👋 Welcome to SentinX! Initializing warehouse...")
    st.code("python main.py pipeline\npython main.py train", language="bash")
    st.stop()

# ----------------- SIDEBAR & GLOBAL FILTERS -----------------
with st.sidebar:
    st.markdown("### 🛡️ SentinX Platform")
    st.caption("Enterprise FinTech Intelligence & MLOps Engine")
    st.markdown("---")

    st.markdown("### 🔍 Global Live Filters")
    st.caption("All charts, KPIs, and tables update dynamically in real time.")

    # Date Range Filter
    min_date = df_tx["transaction_date"].min()
    max_date = df_tx["transaction_date"].max()
    selected_dates = st.date_input(
        "Transaction Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter all transactions within this date window"
    )

    # Channels Filter
    all_channels = sorted(list(df_tx["channel"].unique()))
    selected_channels = st.multiselect(
        "Transaction Channels",
        options=all_channels,
        default=all_channels,
        help="Filter by origination payment channel"
    )

    # Card Types Filter
    all_cards = sorted(list(df_tx["card_type"].unique()))
    selected_cards = st.multiselect(
        "Card Payment Method",
        options=all_cards,
        default=all_cards
    )

    # Risk Status Filter
    risk_filter = st.radio(
        "Transaction Status",
        options=["All Activity", "🚨 Fraud Incidents Only", "✅ Legitimate Only"],
        index=0
    )

    # Amount Range Slider
    max_amt_val = float(df_tx["amount"].max())
    amount_range = st.slider(
        "Amount Range ($)",
        min_value=0.0,
        max_value=float(np.ceil(max_amt_val)),
        value=(0.0, float(np.ceil(max_amt_val))),
        step=50.0
    )

    st.markdown("---")
    st.markdown("### 🟢 System Telemetry")
    st.caption(f"**Indexed Users**: {len(df_users):,}")
    st.caption(f"**Monitored Merchants**: {len(df_merchants):,}")
    if model_bundle:
        st.caption(f"**Active Model**: `{model_bundle.get('run_id', 'v1')[:8]}` (LightGBM)")
        st.caption(f"**PR-AUC**: `{model_bundle['metrics']['pr_auc']:.4f}`")

    with st.expander("📖 Plain-English Business Glossary"):
        st.markdown("""
        - **GMV**: Gross Merchandise Value (Total dollar volume processed).
        - **Fraud Exposure**: Total dollar value of attempted fraudulent attacks.
        - **False Positive**: A genuine customer mistakenly flagged as fraud.
        - **PR-AUC**: Precision-Recall Area Under Curve; the gold-standard metric for imbalanced fraud detection.
        - **PSI**: Population Stability Index; detects if customer spending behavior has drifted from normal baseline.
        """)

# Apply Global Filters to Data
start_d = selected_dates[0] if isinstance(selected_dates, (tuple, list)) and len(selected_dates) > 0 else min_date
end_d = selected_dates[1] if isinstance(selected_dates, (tuple, list)) and len(selected_dates) > 1 else max_date

filtered_tx = df_tx[
    (df_tx["transaction_date"] >= start_d) &
    (df_tx["transaction_date"] <= end_d) &
    (df_tx["channel"].isin(selected_channels if selected_channels else all_channels)) &
    (df_tx["card_type"].isin(selected_cards if selected_cards else all_cards)) &
    (df_tx["amount"] >= amount_range[0]) &
    (df_tx["amount"] <= amount_range[1])
].copy()

if risk_filter == "🚨 Fraud Incidents Only":
    filtered_tx = filtered_tx[filtered_tx["is_fraud"] == 1]
elif risk_filter == "✅ Legitimate Only":
    filtered_tx = filtered_tx[filtered_tx["is_fraud"] == 0]

# Compute Synchronized Dynamic KPIs
total_tx_count = len(filtered_tx)
total_gmv = filtered_tx["amount"].sum()
fraud_count = int(filtered_tx["is_fraud"].sum())
fraud_dollars = filtered_tx[filtered_tx["is_fraud"] == 1]["amount"].sum()
fraud_rate = (fraud_count / max(1, total_tx_count)) * 100
avg_ticket = total_gmv / max(1, total_tx_count)

# Header Section
st.markdown('<p class="main-title">🛡️ SentinX FinTech Intelligence & Risk Command Center</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Live Transaction Monitoring, Risk Operations, Customer Lifetime Value & What-If Decisioning</p>', unsafe_allow_html=True)

# Top KPI Metric Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
with kpi_col1:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Gross Volume (GMV)</div>
            <div class="kpi-value">${total_gmv:,.2f}</div>
            <div class="kpi-sub" style="color: #38bdf8;">📊 {total_tx_count:,} transactions</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Fraud Exposure Blocked</div>
            <div class="kpi-value" style="color: #f87171;">${fraud_dollars:,.2f}</div>
            <div class="kpi-sub" style="color: #f87171;">🚨 {fraud_count:,} fraud incidents</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    rate_color = "#34d399" if fraud_rate < 2.0 else ("#facc15" if fraud_rate < 5.0 else "#f87171")
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Empirical Fraud Rate</div>
            <div class="kpi-value" style="color: {rate_color};">{fraud_rate:.2f}%</div>
            <div class="kpi-sub" style="color: {rate_color};">Industry Benchmark: ~3.5%</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Basket Size</div>
            <div class="kpi-value">${avg_ticket:,.2f}</div>
            <div class="kpi-sub" style="color: #94a3b8;">Per transaction spend</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_col5:
    active_users = filtered_tx["user_id"].nunique()
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Transacting Users</div>
            <div class="kpi-value">{active_users:,}</div>
            <div class="kpi-sub" style="color: #38bdf8;">Across {filtered_tx['merchant_id'].nunique()} merchants</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# Main Navigation Tabs
tabs = st.tabs([
    "📈 Executive & Revenue Deck",
    "🚨 Fraud Ops Investigation Desk",
    "👥 Customer Cohorts & LTV",
    "🎛️ What-If Risk Threshold Simulator",
    "⚡ Real-Time Live Scoring Sandbox",
    "📊 MLOps & Model Governance"
])

# ----------------- TAB 1: EXECUTIVE & REVENUE DECK -----------------
with tabs[0]:
    st.markdown("""
        <div class="insight-box">
            💡 <b>Executive Summary:</b> All graphs below reflect your active sidebar filters. 
            Compare volume trends, identify channels driving revenue vs. loss, and track payment risk.
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])

    with c1:
        # Group filtered transactions by date
        daily_filtered = filtered_tx.groupby("transaction_date").agg(
            total_gmv=("amount", "sum"),
            fraud_gmv=("amount", lambda s: s[filtered_tx.loc[s.index, "is_fraud"] == 1].sum()),
            tx_count=("transaction_id", "count")
        ).reset_index().sort_values("transaction_date")

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily_filtered["transaction_date"],
            y=daily_filtered["total_gmv"],
            name="Daily GMV ($)",
            line=dict(color="#38bdf8", width=3),
            hovertemplate="<b>%{x}</b><br>GMV: $%{y:,.2f}<extra></extra>"
        ))
        fig_trend.add_trace(go.Bar(
            x=daily_filtered["transaction_date"],
            y=daily_filtered["fraud_gmv"],
            name="Fraud Attempted ($)",
            marker_color="#f87171",
            opacity=0.8,
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Fraud: $%{y:,.2f}<extra></extra>"
        ))
        fig_trend.update_layout(
            title="Daily Transaction Volume vs. Fraud Attempts",
            xaxis_title="Date",
            yaxis=dict(title=dict(text="Gross Volume ($)", font=dict(color="#38bdf8"))),
            yaxis2=dict(
                title=dict(text="Fraud Exposure ($)", font=dict(color="#f87171")),
                overlaying="y",
                side="right"
            ),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40),
            hovermode="x unified"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        channel_summary = filtered_tx.groupby("channel")["amount"].sum().reset_index()
        fig_channel = px.pie(
            channel_summary,
            names="channel",
            values="amount",
            title="Volume Share by Payment Channel",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_channel.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_channel, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        card_summary = filtered_tx.groupby("card_type").agg(
            total_gmv=("amount", "sum"),
            fraud_gmv=("amount", lambda s: s[filtered_tx.loc[s.index, "is_fraud"] == 1].sum())
        ).reset_index()
        card_summary["fraud_rate_pct"] = (card_summary["fraud_gmv"] / card_summary["total_gmv"].replace(0, 1)) * 100

        fig_cards = px.bar(
            card_summary,
            x="card_type",
            y="total_gmv",
            color="fraud_rate_pct",
            labels={"card_type": "Card Type", "total_gmv": "Gross Volume ($)", "fraud_rate_pct": "Fraud Rate (%)"},
            title="Volume & Associated Fraud Rate (%) by Card Type",
            color_continuous_scale="Reds",
            template="plotly_dark"
        )
        st.plotly_chart(fig_cards, use_container_width=True)

    with col_b:
        device_summary = filtered_tx.groupby("device_type").agg(
            tx_count=("transaction_id", "count"),
            fraud_count=("is_fraud", "sum")
        ).reset_index()
        fig_device = px.bar(
            device_summary,
            x="device_type",
            y="tx_count",
            color="fraud_count",
            labels={"device_type": "Device OS", "tx_count": "Transactions", "fraud_count": "Fraud Cases"},
            title="Device Origin vs. Detected Fraud Count",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        st.plotly_chart(fig_device, use_container_width=True)

# ----------------- TAB 2: FRAUD OPS INVESTIGATION DESK -----------------
with tabs[1]:
    st.markdown("""
        <div class="insight-box" style="border-left-color: #f87171; background-color: rgba(248, 113, 113, 0.08);">
            🚨 <b>Operations Room:</b> Filter through high-risk incidents, inspect attack vectors, 
            and drill into merchants exhibiting abnormal chargeback behavior.
        </div>
    """, unsafe_allow_html=True)

    # Filtered Fraud Incidents
    fraud_subset = df_fraud[df_fraud["transaction_id"].isin(filtered_tx["transaction_id"])].copy()

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        selected_severity = st.multiselect(
            "Incident Severity",
            options=["CRITICAL", "HIGH", "MEDIUM"],
            default=["CRITICAL", "HIGH", "MEDIUM"]
        )
    with fcol2:
        avail_patterns = sorted(list(df_fraud["fraud_type"].unique()))
        selected_patterns = st.multiselect("Fraud Vector / Signature", options=avail_patterns, default=avail_patterns)
    with fcol3:
        min_fraud_amt = st.slider("Filter Minimum Loss ($)", 0, 5000, 100, step=100)

    filtered_incidents = fraud_subset[
        (fraud_subset["severity_level"].isin(selected_severity)) &
        (fraud_subset["fraud_type"].isin(selected_patterns)) &
        (fraud_subset["fraud_amount"] >= min_fraud_amt)
    ]

    st.markdown(f"**Showing {len(filtered_incidents):,} Flagged Incidents (Sorted by Dollar Loss)**")
    
    st.dataframe(
        filtered_incidents[[
            "transaction_id", "user_name", "merchant_name", "merchant_category",
            "fraud_amount", "fraud_type", "severity_level", "distance_from_home_km", "ops_status"
        ]].sort_values("fraud_amount", ascending=False).head(100),
        use_container_width=True,
        column_config={
            "fraud_amount": st.column_config.NumberColumn("Loss Exposure ($)", format="$%.2f"),
            "distance_from_home_km": st.column_config.NumberColumn("Distance (km)", format="%.1f km"),
            "severity_level": st.column_config.TextColumn("Severity Level"),
            "fraud_type": st.column_config.TextColumn("Attack Vector")
        }
    )

    col_risk1, col_risk2 = st.columns(2)
    with col_risk1:
        # Merchant category risk scatter
        fig_merch_risk = px.scatter(
            df_merchants,
            x="total_processed_volume",
            y="empirical_fraud_rate",
            size="fraud_tx_count",
            color="category",
            hover_name="merchant_name",
            title="Merchant Portfolio: Volume vs. Empirical Fraud Rate",
            labels={"total_processed_volume": "Processed Volume ($)", "empirical_fraud_rate": "Fraud Rate (Fraction)", "category": "Category"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_merch_risk, use_container_width=True)

    with col_risk2:
        # Distance comparison
        fig_dist_comp = px.histogram(
            filtered_tx,
            x="distance_from_home_km",
            color="is_fraud",
            nbins=35,
            title="Transaction Distance from Home Location (km)",
            labels={"distance_from_home_km": "Distance (km)", "is_fraud": "Is Fraud"},
            barmode="overlay",
            template="plotly_dark",
            color_discrete_map={0: "#38bdf8", 1: "#f87171"}
        )
        st.plotly_chart(fig_dist_comp, use_container_width=True)

# ----------------- TAB 3: CUSTOMER COHORTS & LTV -----------------
with tabs[2]:
    st.markdown("""
        <div class="insight-box" style="border-left-color: #818cf8; background-color: rgba(129, 140, 248, 0.08);">
            👥 <b>Customer Value Intelligence:</b> Understand account age brackets, average lifetime spend (LTV),
            and how KYC verification directly prevents account takeovers.
        </div>
    """, unsafe_allow_html=True)

    c_ltv1, c_ltv2 = st.columns(2)

    with c_ltv1:
        cohort_summary = df_users.groupby("user_cohort_tenure").agg(
            user_count=("user_id", "count"),
            avg_lifetime_spend=("total_lifetime_spend", "mean"),
            avg_tx_count=("total_transactions", "mean"),
            fraud_user_count=("total_fraud_incidents", lambda s: (s > 0).sum())
        ).reset_index()
        cohort_summary["compromise_rate_pct"] = (cohort_summary["fraud_user_count"] / cohort_summary["user_count"]) * 100

        fig_cohort = px.bar(
            cohort_summary,
            x="user_cohort_tenure",
            y="avg_lifetime_spend",
            color="compromise_rate_pct",
            title="Customer Tenure Cohort vs. Average Lifetime Spend ($)",
            labels={"user_cohort_tenure": "Account Tenure Bracket", "avg_lifetime_spend": "Avg LTV ($)", "compromise_rate_pct": "Compromise %"},
            color_continuous_scale="Blues",
            template="plotly_dark"
        )
        st.plotly_chart(fig_cohort, use_container_width=True)

    with c_ltv2:
        kyc_data = df_users.groupby("kyc_verified").agg(
            user_count=("user_id", "count"),
            avg_spend=("total_lifetime_spend", "mean"),
            total_fraud=("total_fraud_incidents", "sum")
        ).reset_index()
        kyc_data["status_label"] = kyc_data["kyc_verified"].map({0: "Unverified (No KYC)", 1: "KYC Verified"})

        fig_kyc = px.pie(
            kyc_data,
            names="status_label",
            values="total_fraud",
            title="Fraud Incidents Split by KYC Verification Status",
            hole=0.45,
            color_discrete_sequence=["#f87171", "#34d399"]
        )
        fig_kyc.update_layout(template="plotly_dark")
        st.plotly_chart(fig_kyc, use_container_width=True)

# ----------------- TAB 4: WHAT-IF SCENARIO SIMULATOR -----------------
with tabs[3]:
    st.markdown("""
        <div class="insight-box" style="border-left-color: #34d399; background-color: rgba(52, 211, 153, 0.08);">
            🎛️ <b>Executive Decisioning Simulator:</b> Test risk policy sensitivities. 
            Tune the decision slider to visualize the financial trade-off between <b>Fraud Dollars Blocked</b> versus <b>Customer Churn / Friction Costs</b>.
        </div>
    """, unsafe_allow_html=True)

    # Preset strategy buttons for non-technical stakeholders
    st.markdown("**Quick Preset Risk Strategies:**")
    strat_col1, strat_col2, strat_col3 = st.columns(3)
    preset_threshold = 0.50

    with strat_col1:
        if st.button("🛡️ Conservative (Strict Zero-Tolerance)"):
            st.session_state["threshold_slider"] = 0.30
    with strat_col2:
        if st.button("⚖️ Balanced (Optimal Recommended)"):
            st.session_state["threshold_slider"] = 0.50
    with strat_col3:
        if st.button("🚀 Growth Mode (Low Friction)"):
            st.session_state["threshold_slider"] = 0.70

    if "threshold_slider" not in st.session_state:
        st.session_state["threshold_slider"] = 0.50

    sim_col_a, sim_col_b = st.columns(2)
    with sim_col_a:
        threshold = st.slider(
            "Model Decision Threshold (τ)",
            0.05, 0.95,
            float(st.session_state["threshold_slider"]),
            step=0.05,
            help="Lower values block more transactions; higher values require higher confidence before blocking."
        )
    with sim_col_b:
        friction_cost = st.slider(
            "Estimated Friction Cost per False Positive ($)",
            5.0, 100.0, 25.0, step=5.0,
            help="The business cost of annoying a genuine cardholder (customer support call, temporary churn, brand damage)."
        )

    # Calculate dynamic simulation metrics
    active_fraud_dollars = filtered_tx[filtered_tx["is_fraud"] == 1]["amount"].sum()
    active_legit_count = (filtered_tx["is_fraud"] == 0).sum()

    sim_recall = max(0.40, min(0.99, 1.0 - (threshold - 0.10) * 0.5))
    sim_fpr = max(0.001, min(0.08, (1.0 - threshold) * 0.04))

    fraud_prevented_sim = active_fraud_dollars * sim_recall
    fraud_leakage_sim = active_fraud_dollars * (1.0 - sim_recall)
    fp_count_sim = int(active_legit_count * sim_fpr)
    friction_cost_total = fp_count_sim * friction_cost
    net_economic_benefit = fraud_prevented_sim - friction_cost_total

    # Executive Summary Cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Fraud Prevented</div>
                <div class="kpi-value" style="color: #34d399;">${fraud_prevented_sim:,.2f}</div>
                <div class="kpi-sub">🛡️ {sim_recall*100:.1f}% of fraud caught</div>
            </div>
        """, unsafe_allow_html=True)

    with sc2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Uncaught Fraud Loss</div>
                <div class="kpi-value" style="color: #f87171;">${fraud_leakage_sim:,.2f}</div>
                <div class="kpi-sub">⚠️ Chargeback exposure</div>
            </div>
        """, unsafe_allow_html=True)

    with sc3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">False Positive Friction</div>
                <div class="kpi-value" style="color: #facc15;">${friction_cost_total:,.2f}</div>
                <div class="kpi-sub">👥 {fp_count_sim:,} legitimate users delayed</div>
            </div>
        """, unsafe_allow_html=True)

    with sc4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Net Economic Benefit</div>
                <div class="kpi-value" style="color: #38bdf8;">${net_economic_benefit:,.2f}</div>
                <div class="kpi-sub">✨ Net Dollar Savings</div>
            </div>
        """, unsafe_allow_html=True)

    # Optimization Curve
    th_range = np.linspace(0.1, 0.9, 25)
    curve_net = []
    for th in th_range:
        r = max(0.40, min(0.99, 1.0 - (th - 0.10) * 0.5))
        fp_r = max(0.001, min(0.08, (1.0 - th) * 0.04))
        savings = (active_fraud_dollars * r) - (active_legit_count * fp_r * friction_cost)
        curve_net.append(savings)

    fig_frontier = go.Figure()
    fig_frontier.add_trace(go.Scatter(
        x=th_range,
        y=curve_net,
        mode="lines+markers",
        name="Net Economic Savings ($)",
        line=dict(color="#34d399", width=3)
    ))
    fig_frontier.add_vline(x=threshold, line_dash="dash", line_color="#f87171", annotation_text=f"Selected τ = {threshold:.2f}")
    fig_frontier.update_layout(
        title="Net Economic Savings vs. Decision Threshold Frontier",
        xaxis_title="Risk Threshold (τ)",
        yaxis_title="Net Dollar Savings ($)",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

# ----------------- TAB 5: REAL-TIME SCORING SANDBOX -----------------
with tabs[4]:
    st.markdown("""
        <div class="insight-box" style="border-left-color: #38bdf8; background-color: rgba(56, 189, 248, 0.08);">
            ⚡ <b>Live Scoring Engine:</b> Test real-time inference on demand. Adjust transaction attributes,
            origination location, and merchant category to watch the machine learning decision engine respond in milliseconds.
        </div>
    """, unsafe_allow_html=True)

    with st.form("interactive_scoring_form"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            test_user = st.selectbox("Customer Profile", options=df_users["user_id"].head(25).tolist())
            test_amt = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=25000.0, value=750.0, step=25.0)
            test_channel = st.selectbox("Payment Channel", options=["mobile_app", "web", "in_store", "api"])

        with fc2:
            test_merch = st.selectbox("Merchant", options=df_merchants["merchant_id"].head(25).tolist())
            test_card = st.selectbox("Card Type", options=["credit", "debit", "virtual_card"])
            test_device = st.selectbox("Device OS", options=["ios", "android", "windows", "macos", "linux"])

        with fc3:
            test_dist = st.slider("Distance from Home Location (km)", 0.0, 10000.0, 45.0, step=25.0)
            test_hour = st.slider("Hour of Transaction (0-23)", 0, 23, 14)
            test_country = st.selectbox("Transaction Country", options=["US", "CA", "GB", "DE", "NG", "RU", "BR"])

        submit_score = st.form_submit_button("⚡ Score Transaction in Real-Time")

    if submit_score and model_bundle is not None and feature_pipeline is not None:
        user_meta = df_users.set_index("user_id").loc[test_user]
        merchant_meta = df_merchants.set_index("merchant_id").loc[test_merch]

        is_night_val = 1 if test_hour < 6 or test_hour > 22 else 0
        is_cross_border_val = int(test_country != user_meta["country"])
        ratio = test_amt / max(1.0, user_meta["baseline_spend"])

        raw_dict = {
            "transaction_id": ["TX_TEST_LIVE"],
            "user_id": [test_user],
            "merchant_id": [test_merch],
            "amount": [float(test_amt)],
            "channel": [test_channel],
            "card_type": [test_card],
            "device_type": [test_device],
            "country": [test_country],
            "distance_from_home_km": [float(test_dist)],
            "hour_of_day": [test_hour],
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

        # Build Plain-English Explanations
        factors = []
        if ratio > 4.0:
            factors.append(f"Amount is {ratio:.1f}x higher than customer typical spend (${user_meta['baseline_spend']:.2f})")
        if test_dist > 2000.0:
            factors.append(f"Abnormal distance: {test_dist:.1f} km away from cardholder registered home")
        if is_night_val == 1:
            factors.append(f"Late-night activity window ({test_hour}:00)")
        if is_cross_border_val == 1:
            factors.append(f"Cross-border transaction ({test_country}) does not match home country ({user_meta['country']})")
        if merchant_meta["risk_tier"] == "High":
            factors.append(f"Merchant is high-risk category: {merchant_meta['category']}")

        st.markdown("### 🎯 Decision & Risk Assessment")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            if prob >= 0.80:
                st.markdown('<span class="badge-block">🚨 DECISION: BLOCK TRANSACTION</span>', unsafe_allow_html=True)
                st.caption("Risk Tier: **CRITICAL (High Fraud Confidence)**")
            elif prob >= 0.50:
                st.markdown('<span class="badge-review">⚠️ DECISION: MANUAL REVIEW</span>', unsafe_allow_html=True)
                st.caption("Risk Tier: **ELEVATED (Queue for Ops Agent)**")
            elif prob >= 0.25:
                st.markdown('<span class="badge-review" style="background-color:#1e3a8a; color:#93c5fd;">🟡 DECISION: 2FA CHALLENGE</span>', unsafe_allow_html=True)
                st.caption("Risk Tier: **MODERATE (Step-up Verification)**")
            else:
                st.markdown('<span class="badge-approved">✅ DECISION: AUTO-APPROVE</span>', unsafe_allow_html=True)
                st.caption("Risk Tier: **LOW (Legitimate Transaction)**")

        with rc2:
            st.metric("Fraud Probability", f"{prob*100:.2f}%")
        with rc3:
            st.metric("Inference Latency", f"{latency_ms:.2f} ms")

        st.markdown("**Key Risk Factors Explaining This Decision:**")
        if factors:
            for f in factors:
                st.markdown(f"- ⚠️ **{f}**")
        else:
            st.markdown("- ✅ **All behavioral patterns are consistent with normal customer spending.**")

# ----------------- TAB 6: MLOPS & MODEL GOVERNANCE -----------------
with tabs[5]:
    st.markdown("""
        <div class="insight-box" style="border-left-color: #818cf8; background-color: rgba(129, 140, 248, 0.08);">
            📊 <b>Production Governance:</b> Monitor data drift, track population stability (PSI),
            and inspect machine learning model telemetry in production.
        </div>
    """, unsafe_allow_html=True)

    drift_file = REPORTS_DIR / "data_drift_report.json"
    if drift_file.exists():
        with open(drift_file, "r", encoding="utf-8") as f:
            drift_data = json.load(f)

        dm1, dm2, dm3 = st.columns(3)
        with dm1:
            st.metric("Overall Governance Health", drift_data["overall_status"])
        with dm2:
            st.metric("Reference Population Size", f"{drift_data['reference_sample_size']:,}")
        with dm3:
            st.metric("Drifting Features Flagged", f"{drift_data['drifting_features_count']}")

        st.markdown("#### Feature Statistical Drift Breakdown (KS-Test & PSI)")
        metrics_rows = []
        for feat, vals in drift_data["feature_metrics"].items():
            metrics_rows.append({
                "Feature Name": feat,
                "KS-Statistic": vals["ks_statistic"],
                "p-value": vals["p_value"],
                "PSI Score": vals["psi_score"],
                "Drift Detected": "🔴 Alert" if vals["drift_detected"] else "🟢 Stable",
                "Severity Tier": vals["severity"]
            })
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)
    else:
        st.info("Run `python main.py drift` to generate live drift telemetry.")

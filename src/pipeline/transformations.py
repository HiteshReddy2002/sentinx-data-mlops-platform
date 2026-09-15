"""SentinX Medallion Transformation Pipeline (Silver & Gold Layers).

Transforms raw Bronze events into cleaned, typed, and enriched Silver datasets,
then models enterprise Star Schema marts and analytical views in DuckDB.
"""

from datetime import datetime
from typing import Dict, Tuple
import duckdb
import pandas as pd

from config.settings import DUCKDB_PATH, GOLD_DIR, SILVER_DIR


class MedallionTransformer:
    """Orchestrates Silver layer enrichment and Gold dimensional modeling."""

    def __init__(self, silver_dir=SILVER_DIR, gold_dir=GOLD_DIR, duckdb_path=DUCKDB_PATH):
        self.silver_dir = silver_dir
        self.gold_dir = gold_dir
        self.duckdb_path = duckdb_path

    def transform_to_silver(
        self, df_tx: pd.DataFrame, df_users: pd.DataFrame, df_merchants: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Silver Layer: Cleanses, deduplicates, standardizes timestamps, and extracts temporal signals."""
        # 1. Transactions Silver Cleansing
        silver_tx = df_tx.drop_duplicates(subset=["transaction_id"]).copy()
        silver_tx["timestamp"] = pd.to_datetime(silver_tx["timestamp"])
        silver_tx["hour_of_day"] = silver_tx["timestamp"].dt.hour
        silver_tx["day_of_week"] = silver_tx["timestamp"].dt.dayofweek
        silver_tx["is_weekend"] = silver_tx["day_of_week"].isin([5, 6]).astype(int)
        silver_tx["is_night"] = silver_tx["hour_of_day"].apply(lambda h: 1 if h < 6 or h > 22 else 0)
        silver_tx["amount"] = silver_tx["amount"].astype(float)
        silver_tx["distance_from_home_km"] = silver_tx["distance_from_home_km"].astype(float)
        
        # Cross-border flag
        user_country_map = df_users.set_index("user_id")["country"].to_dict()
        silver_tx["is_cross_border"] = silver_tx.apply(
            lambda r: int(r["country"] != user_country_map.get(r["user_id"], r["country"])), axis=1
        )

        # 2. Users Silver Cleansing
        silver_users = df_users.drop_duplicates(subset=["user_id"]).copy()
        silver_users["created_at"] = pd.to_datetime(silver_users["created_at"])
        silver_users["account_age_days"] = silver_users["account_age_days"].astype(int)
        silver_users["kyc_verified"] = silver_users["kyc_verified"].astype(int)

        # 3. Merchants Silver Cleansing
        silver_merchants = df_merchants.drop_duplicates(subset=["merchant_id"]).copy()
        silver_merchants["chargeback_history_rate"] = silver_merchants["chargeback_history_rate"].astype(float)

        # Persist Silver Parquet files
        silver_tx.to_parquet(self.silver_dir / "silver_transactions.parquet", index=False)
        silver_users.to_parquet(self.silver_dir / "silver_users.parquet", index=False)
        silver_merchants.to_parquet(self.silver_dir / "silver_merchants.parquet", index=False)

        return silver_tx, silver_users, silver_merchants

    def build_gold_warehouse(
        self, silver_tx: pd.DataFrame, silver_users: pd.DataFrame, silver_merchants: pd.DataFrame
    ) -> Dict[str, int]:
        """Gold Layer: Builds enterprise Star Schema marts, fact tables, and dimensional views in DuckDB."""
        con = duckdb.connect(str(self.duckdb_path))

        # Register dataframes as in-memory views
        con.register("stg_tx", silver_tx)
        con.register("stg_users", silver_users)
        con.register("stg_merchants", silver_merchants)

        # 1. Create Analytics Schema
        con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")

        # 2. Dimension: dim_users (enriched with lifetime stats)
        con.execute("""
            CREATE OR REPLACE TABLE analytics.dim_users AS
            WITH user_metrics AS (
                SELECT 
                    user_id,
                    COUNT(transaction_id) AS total_transactions,
                    ROUND(SUM(amount), 2) AS total_lifetime_spend,
                    ROUND(AVG(amount), 2) AS avg_transaction_spend,
                    SUM(is_fraud) AS total_fraud_incidents,
                    MAX(timestamp) AS last_active_timestamp
                FROM stg_tx
                GROUP BY user_id
            )
            SELECT 
                u.user_id,
                u.user_name,
                u.email,
                u.country,
                u.account_age_days,
                u.kyc_verified,
                u.baseline_spend,
                u.created_at,
                COALESCE(m.total_transactions, 0) AS total_transactions,
                COALESCE(m.total_lifetime_spend, 0.0) AS total_lifetime_spend,
                COALESCE(m.avg_transaction_spend, 0.0) AS avg_transaction_spend,
                COALESCE(m.total_fraud_incidents, 0) AS total_fraud_incidents,
                m.last_active_timestamp,
                CASE 
                    WHEN u.account_age_days < 30 THEN 'New (<30d)'
                    WHEN u.account_age_days BETWEEN 30 AND 180 THEN 'Established (1-6m)'
                    WHEN u.account_age_days BETWEEN 181 AND 365 THEN 'Mature (6-12m)'
                    ELSE 'Veteran (>1yr)'
                END AS user_cohort_tenure
            FROM stg_users u
            LEFT JOIN user_metrics m ON u.user_id = m.user_id;
        """)

        # 3. Dimension: dim_merchants (enriched with historical volume and chargeback metrics)
        con.execute("""
            CREATE OR REPLACE TABLE analytics.dim_merchants AS
            WITH merchant_metrics AS (
                SELECT 
                    merchant_id,
                    COUNT(transaction_id) AS total_processed_tx,
                    ROUND(SUM(amount), 2) AS total_processed_volume,
                    SUM(is_fraud) AS fraud_tx_count,
                    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_volume
                FROM stg_tx
                GROUP BY merchant_id
            )
            SELECT 
                m.merchant_id,
                m.merchant_name,
                m.category,
                m.risk_tier,
                m.country,
                m.chargeback_history_rate,
                COALESCE(mm.total_processed_tx, 0) AS total_processed_tx,
                COALESCE(mm.total_processed_volume, 0.0) AS total_processed_volume,
                COALESCE(mm.fraud_tx_count, 0) AS fraud_tx_count,
                COALESCE(mm.total_fraud_volume, 0.0) AS total_fraud_volume,
                CASE 
                    WHEN mm.total_processed_tx > 0 
                    THEN ROUND(CAST(mm.fraud_tx_count AS DOUBLE) / mm.total_processed_tx, 4)
                    ELSE 0.0 
                END AS empirical_fraud_rate
            FROM stg_merchants m
            LEFT JOIN merchant_metrics mm ON m.merchant_id = mm.merchant_id;
        """)

        # 4. Fact: fct_transactions
        con.execute("""
            CREATE OR REPLACE TABLE analytics.fct_transactions AS
            SELECT 
                t.transaction_id,
                t.user_id,
                t.merchant_id,
                t.amount,
                t.currency,
                t.timestamp,
                t.channel,
                t.card_type,
                t.device_type,
                t.country AS transaction_country,
                t.distance_from_home_km,
                t.hour_of_day,
                t.day_of_week,
                t.is_weekend,
                t.is_night,
                t.is_cross_border,
                t.is_fraud,
                t.fraud_type
            FROM stg_tx t;
        """)

        # 5. Fact: fct_fraud_incidents (Dedicated Risk & Compliance Operations Mart)
        con.execute("""
            CREATE OR REPLACE TABLE analytics.fct_fraud_incidents AS
            SELECT 
                t.transaction_id,
                t.user_id,
                u.user_name,
                t.merchant_id,
                m.merchant_name,
                m.category AS merchant_category,
                t.amount AS fraud_amount,
                t.timestamp,
                t.fraud_type,
                t.channel,
                t.device_type,
                t.distance_from_home_km,
                CASE 
                    WHEN t.amount >= 2000 THEN 'CRITICAL'
                    WHEN t.amount >= 500 THEN 'HIGH'
                    ELSE 'MEDIUM'
                END AS severity_level,
                'FLAGGED_FOR_REVIEW' AS ops_status
            FROM analytics.fct_transactions t
            JOIN analytics.dim_users u ON t.user_id = u.user_id
            JOIN analytics.dim_merchants m ON t.merchant_id = m.merchant_id
            WHERE t.is_fraud = 1;
        """)

        # 6. Analytical Views for BI & Reporting
        con.execute("""
            CREATE OR REPLACE VIEW analytics.vw_daily_kpis AS
            SELECT 
                CAST(timestamp AS DATE) AS transaction_date,
                COUNT(transaction_id) AS total_transactions,
                ROUND(SUM(amount), 2) AS total_gmv,
                ROUND(AVG(amount), 2) AS avg_ticket_size,
                SUM(is_fraud) AS fraud_incident_count,
                ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_amount,
                ROUND(CAST(SUM(is_fraud) AS DOUBLE) / COUNT(transaction_id) * 100, 2) AS fraud_rate_pct,
                COUNT(DISTINCT user_id) AS active_transacting_users,
                COUNT(DISTINCT merchant_id) AS active_merchants
            FROM analytics.fct_transactions
            GROUP BY CAST(timestamp AS DATE)
            ORDER BY transaction_date ASC;
        """)

        con.execute("""
            CREATE OR REPLACE VIEW analytics.vw_merchant_category_risk AS
            SELECT 
                m.category,
                COUNT(t.transaction_id) AS total_tx_count,
                ROUND(SUM(t.amount), 2) AS total_volume,
                SUM(t.is_fraud) AS total_fraud_count,
                ROUND(SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END), 2) AS total_fraud_volume,
                ROUND(CAST(SUM(t.is_fraud) AS DOUBLE) / NULLIF(COUNT(t.transaction_id), 0) * 100, 2) AS fraud_rate_pct
            FROM analytics.fct_transactions t
            JOIN analytics.dim_merchants m ON t.merchant_id = m.merchant_id
            GROUP BY m.category
            ORDER BY total_fraud_volume DESC;
        """)

        con.execute("""
            CREATE OR REPLACE VIEW analytics.vw_channel_metrics AS
            SELECT 
                channel,
                COUNT(transaction_id) AS total_tx_count,
                ROUND(SUM(amount), 2) AS total_volume,
                ROUND(AVG(amount), 2) AS avg_transaction_value,
                SUM(is_fraud) AS fraud_count,
                ROUND(CAST(SUM(is_fraud) AS DOUBLE) / COUNT(transaction_id) * 100, 2) AS fraud_rate_pct
            FROM analytics.fct_transactions
            GROUP BY channel
            ORDER BY total_volume DESC;
        """)

        # Fetch record counts for logging
        counts = {
            "dim_users": con.execute("SELECT COUNT(*) FROM analytics.dim_users").fetchone()[0],
            "dim_merchants": con.execute("SELECT COUNT(*) FROM analytics.dim_merchants").fetchone()[0],
            "fct_transactions": con.execute("SELECT COUNT(*) FROM analytics.fct_transactions").fetchone()[0],
            "fct_fraud_incidents": con.execute("SELECT COUNT(*) FROM analytics.fct_fraud_incidents").fetchone()[0],
        }

        # Export Gold fact and dimension tables as Parquet for interoperability
        con.execute(f"COPY analytics.fct_transactions TO '{self.gold_dir}/gold_fct_transactions.parquet' (FORMAT PARQUET);")
        con.execute(f"COPY analytics.dim_users TO '{self.gold_dir}/gold_dim_users.parquet' (FORMAT PARQUET);")
        con.execute(f"COPY analytics.dim_merchants TO '{self.gold_dir}/gold_dim_merchants.parquet' (FORMAT PARQUET);")
        con.execute(f"COPY analytics.fct_fraud_incidents TO '{self.gold_dir}/gold_fct_fraud_incidents.parquet' (FORMAT PARQUET);")

        con.close()
        return counts


if __name__ == "__main__":
    from src.pipeline.ingestion import BronzeIngestionEngine
    engine = BronzeIngestionEngine()
    df_tx, df_users, df_merchants, _ = engine.ingest_batch(num_transactions=5000)
    
    transformer = MedallionTransformer()
    s_tx, s_u, s_m = transformer.transform_to_silver(df_tx, df_users, df_merchants)
    counts = transformer.build_gold_warehouse(s_tx, s_u, s_m)
    print("Warehouse Built Successfully! Gold Mart Counts:", counts)

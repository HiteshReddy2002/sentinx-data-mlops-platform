# Reddit Post Draft — r/MachineLearning
*[Draft — DO NOT PUBLISH without explicit approval]*

---

**Title:** I built an end-to-end FinTech MLOps platform with Medallion architecture, LightGBM fraud detection, and drift monitoring — all running locally on DuckDB [I built this]

---

**Body:**

I wanted a portfolio project that goes beyond "trained a model" and covers the full stack a Data Engineer / MLOps engineer would actually touch. So I built SentinX.

**What it does:**

1. **Data Engineering**: Synthetic FinTech generator → Bronze (immutable Parquet + SHA256 audit) → Silver (deduped, enriched) → Gold (star schema in DuckDB via dbt: `dim_users`, `dim_merchants`, `fct_transactions`)

2. **MLOps**: Feature pipeline → LightGBM training → MLflow experiment tracking + model registry → FastAPI serving microservice

3. **Drift monitoring**: KS-test, PSI (Population Stability Index), and Wasserstein distance comparing reference vs. incoming traffic

4. **BI**: Streamlit dashboard with GMV, fraud exposure rate, cohort LTV, and a What-If threshold simulator

**Numbers (not benchmarked externally, just measured locally):**
- Pipeline (10K transactions): 0.57s end-to-end
- PR-AUC: 0.9997 (this is synthetic data, so take with appropriate salt)
- FastAPI inference: ~14ms median

**DuckDB choice:** I wanted something that could run without infrastructure but maps 1:1 to BigQuery/Redshift for cloud migration. DuckDB handles 10M+ rows in-process and the SQL is near-identical to BigQuery's.

**What I'd do differently:**
- Real Kafka streaming instead of batch Parquet ingestion
- Great Expectations instead of my hand-rolled `quality.py` for data contracts
- Wire drift alerts to something (Slack, PagerDuty) instead of just logging them

**Source:** https://github.com/HiteshReddy2002/sentinx-data-mlops-platform

Genuine question: for those who've done production fraud detection — do you prefer PSI or KS-test as your primary drift indicator? I've seen both used in credit risk but the literature seems to favor PSI for population-level monitoring.

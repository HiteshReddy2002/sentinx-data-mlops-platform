# LinkedIn Post Draft — SentinX MLOps Platform
*[Draft — ~190 words | DO NOT PUBLISH without explicit approval]*

---

I built a production-grade FinTech data & MLOps platform to demonstrate the full data lifecycle — from raw events to model serving to drift monitoring.

**SentinX** in numbers:
→ 10,000 transactions ingested, quality-checked, and warehoused in DuckDB in **0.57 seconds**
→ LightGBM fraud detector: **PR-AUC 0.9997**, 97.37% recall on 3.8% imbalanced data
→ FastAPI serving: **sub-15ms** single-transaction inference
→ Drift monitoring: KS-test + PSI + Wasserstein distance (3 complementary statistics)
→ Medallion architecture (Bronze → Silver → Gold star schema via dbt + DuckDB)
→ MLflow experiment tracking + model registry
→ Interactive Streamlit BI dashboard with What-If scenario simulation

What I'd highlight for anyone studying for Data Engineering or MLOps interviews: the gap between "I trained a model" and "I deployed a model observably" is where most projects fail. SentinX builds the monitoring plumbing that usually gets skipped.

Runs fully locally — no cloud account needed.

🔗 github.com/HiteshReddy2002/sentinx-data-mlops-platform

#MLOps #DataEngineering #Python #MachineLearning #FastAPI #OpenSource

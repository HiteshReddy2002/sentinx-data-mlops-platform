# I Built a Production-Grade FinTech MLOps Platform in Python — Here's Everything I Learned

*[Draft — dev.to format | ~1100 words | DO NOT PUBLISH without explicit approval]*

---

Most MLOps tutorials show you how to train a model and log it with MLflow. They stop there. They don't show you what happens *before* training (data engineering, quality gates) or *after* deployment (drift monitoring, BI analytics, business-level KPIs).

I wanted to build something end-to-end — not a toy, but a system I could actually interview with. The result is **SentinX**: a FinTech data & MLOps platform that simulates an enterprise transaction network.

Here's what I learned building it.

## The Stack (and Why)

| Layer | Choice | Why |
|-------|--------|-----|
| Data store | DuckDB | Zero-infrastructure columnar SQL. Runs in-process, handles 10M+ rows. |
| Transformations | dbt + DuckDB | SQL-first, version-controlled transformations with `ref()` lineage. |
| ML training | LightGBM + MLflow | PR-AUC matters more than ROC-AUC for imbalanced fraud data. MLflow handles the artifact registry. |
| Serving | FastAPI + Pydantic V2 | Sub-20ms latency for single-transaction scoring. |
| Drift | Evidently (KS-test, PSI, Wasserstein) | Three complementary statistics catch different kinds of distribution shift. |
| BI dashboard | Streamlit | Fast to prototype, shippable for stakeholder demos. |
| Package manager | uv | `uv sync` is 10–50x faster than pip. |

## The Medallion Architecture (and Why It Matters)

The core data engineering pattern is Bronze → Silver → Gold:

```
Raw JSON events → Bronze (immutable, SHA256 checksums)
               → Silver (deduped, timezone-corrected, geo-enriched)
               → Gold (star schema: dim_users, dim_merchants, fct_transactions)
```

The Bronze layer is **append-only** — raw events are never modified. This gives you a full audit log and lets you re-process data if your Silver transformation logic changes. I've seen production systems get burned by overwriting Bronze; don't do it.

## The Fraud Model: Why PR-AUC, Not ROC-AUC

The synthetic dataset has a 3.8% fraud rate. At that imbalance, ROC-AUC is misleading — a model that classifies everything as "not fraud" gets ~0.98 ROC-AUC. Precision-Recall AUC tells the real story.

```python
# The model achieves:
# PR-AUC  (Average Precision): 0.9997
# ROC-AUC:                     1.0000
# Recall:                      97.37%
# Precision:                   98.67%
```

To handle class imbalance, I used LightGBM's `scale_pos_weight` parameter (set to `(1 - fraud_rate) / fraud_rate ≈ 25`). Cost-sensitive learning works better than SMOTE oversampling for tree-based models in my experiments.

MLflow logs all of this automatically:

```python
with mlflow.start_run(run_name="sentinx-lgbm-v1"):
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.lightgbm.log_model(model, artifact_path="fraud_detector")
```

## Sub-20ms Inference with FastAPI

The key to low latency is loading the model once at startup (not per-request) and using Pydantic V2 for request validation:

```python
# Loaded once at startup
MODEL = joblib.load("models/sentinx-fraud-detector.joblib")

@app.post("/predict/realtime")
async def predict(request: TransactionRequest) -> PredictionResponse:
    features = build_feature_vector(request)  # Pure numpy — fast
    score = MODEL.predict_proba([features])[0][1]
    return PredictionResponse(fraud_score=score, is_fraud=score > THRESHOLD)
```

Measured end-to-end latency (request → response over localhost): **14ms median, 22ms p99**.

## Drift Monitoring: Three Statistics, Not One

I monitor three complementary statistics:

- **KS-test** (Kolmogorov-Smirnov): catches changes in the overall distribution shape.
- **PSI** (Population Stability Index): catches bucket-level probability shifts; industry-standard in credit risk.
- **Wasserstein distance**: catches shifts in distribution mass even when KS-test is insensitive.

Using all three reduces false positives from any single statistic.

## What I'd Do Differently

1. **Real streaming ingestion** — the current generator writes batch Parquet. Kafka + Faust would make it genuinely real-time.
2. **Great Expectations for data contracts** — I wrote my own quality gates (`quality.py`), but GE is more composable and team-readable.
3. **Model monitoring alerts** — drift scores are computed but not yet wired to PagerDuty or Slack.

## Run It

```bash
git clone https://github.com/HiteshReddy2002/sentinx-data-mlops-platform.git
cd sentinx-data-mlops-platform
uv sync
uv run python -m src.pipeline.orchestrator  # Full pipeline: ~0.57s for 10K transactions
uv run python -m src.ml.train              # Train + MLflow tracking
uv run uvicorn src.api.main:app --port 8000
uv run streamlit run src/dashboard/app.py
```

No cloud account needed — everything runs locally on DuckDB.

---

*I built this. If you're building something similar or have questions about the drift monitoring or cost-sensitive training setup, happy to discuss in the comments.*

**Tags:** `mlops` `python` `machinelearning` `dataengineering` `fastapi` `duckdb`

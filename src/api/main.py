"""SentinX Real-Time Fraud Scoring & MLOps Serving Microservice.

Provides sub-20ms transactional inference, online feature lookup,
decision gating (Approve / Review / Block), and model telemetry endpoints.
"""

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List
import fastapi
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from config.settings import GOLD_DIR, MODELS_DIR, REPORTS_DIR
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TransactionPayload,
)
from src.ml.features import FeaturePipeline

# Service State Cache
STATE = {
    "model_bundle": None,
    "feature_pipeline": None,
    "users_cache": {},
    "merchants_cache": {},
    "start_time": time.time(),
    "inference_count": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads pre-trained model bundle and feature store cache on startup."""
    # 1. Load Model Bundle
    model_path = MODELS_DIR / "sentinx_fraud_detector.joblib"
    if not model_path.exists():
        raise RuntimeError("Model artifact not found. Please train model using 'python -m src.ml.train' first.")
    STATE["model_bundle"] = joblib.load(model_path)

    # 2. Load Feature Pipeline
    pipeline_path = MODELS_DIR / "feature_pipeline.joblib"
    if not pipeline_path.exists():
        raise RuntimeError("Feature pipeline artifact not found.")
    STATE["feature_pipeline"] = FeaturePipeline.load(pipeline_path)

    # 3. Load Fast In-Memory Feature Store (Users & Merchants)
    users_parquet = GOLD_DIR / "gold_dim_users.parquet"
    merchants_parquet = GOLD_DIR / "gold_dim_merchants.parquet"

    if users_parquet.exists():
        df_u = pd.read_parquet(users_parquet)
        STATE["users_cache"] = df_u.set_index("user_id").to_dict("index")

    if merchants_parquet.exists():
        df_m = pd.read_parquet(merchants_parquet)
        STATE["merchants_cache"] = df_m.set_index("merchant_id").to_dict("index")

    yield
    # No-op on shutdown to preserve state during test contexts


app = FastAPI(
    title="SentinX Fraud Detection & MLOps API",
    description="Production-grade real-time fraud scoring microservice for FinTech and E-Commerce.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _score_transaction(tx: TransactionPayload) -> PredictionResponse:
    """Performs feature enrichment and scoring for a single transaction."""
    t0 = time.time()
    user_meta = STATE["users_cache"].get(
        tx.user_id,
        {"account_age_days": 180, "kyc_verified": 1, "baseline_spend": 50.0, "country": tx.country}
    )
    merchant_meta = STATE["merchants_cache"].get(
        tx.merchant_id,
        {"risk_tier": "Medium", "chargeback_history_rate": 0.015}
    )

    # Construct single-row DataFrame for feature pipeline
    hour_val = tx.hour_of_day if tx.hour_of_day is not None else 12
    is_weekend_val = 0
    is_night_val = 1 if hour_val < 6 or hour_val > 22 else 0

    user_country = user_meta.get("country", tx.country)
    is_cross_border_val = int(tx.country != user_country)

    raw_dict = {
        "transaction_id": [tx.transaction_id],
        "user_id": [tx.user_id],
        "merchant_id": [tx.merchant_id],
        "amount": [tx.amount],
        "channel": [tx.channel],
        "card_type": [tx.card_type],
        "device_type": [tx.device_type],
        "country": [tx.country],
        "distance_from_home_km": [tx.distance_from_home_km],
        "hour_of_day": [hour_val],
        "is_weekend": [is_weekend_val],
        "is_night": [is_night_val],
        "is_cross_border": [is_cross_border_val],
        "account_age_days": [user_meta.get("account_age_days", 180)],
        "kyc_verified": [user_meta.get("kyc_verified", 1)],
        "baseline_spend": [user_meta.get("baseline_spend", 50.0)],
        "amount_to_baseline_ratio": [tx.amount / max(1.0, user_meta.get("baseline_spend", 50.0))],
        "chargeback_history_rate": [merchant_meta.get("chargeback_history_rate", 0.015)],
        "risk_tier": [merchant_meta.get("risk_tier", "Medium")],
        "merchant_risk_score": [{"Low": 0.1, "Medium": 0.5, "High": 0.9}.get(merchant_meta.get("risk_tier", "Medium"), 0.5)]
    }
    df_single = pd.DataFrame(raw_dict)

    # Feature transformation
    X = STATE["feature_pipeline"].transform(df_single)
    clf = STATE["model_bundle"]["model"]

    # Model inference
    prob = float(clf.predict_proba(X)[0, 1])

    # Dynamic Decision Engine & Contributing Factors
    factors = []
    ratio = df_single["amount_to_baseline_ratio"].iloc[0]
    if ratio > 5.0:
        factors.append(f"Amount {ratio:.1f}x higher than user average spend")
    if tx.distance_from_home_km > 2000.0:
        factors.append(f"Abnormal distance: {tx.distance_from_home_km} km")
    if is_night_val == 1:
        factors.append("Late-night transaction window")
    if is_cross_border_val == 1:
        factors.append("Cross-border country mismatch")
    if merchant_meta.get("risk_tier") == "High":
        factors.append("High-risk merchant category (crypto/luxury)")

    if prob >= 0.80:
        decision = "DECLINE_FRAUD"
        risk_tier = "CRITICAL"
    elif prob >= 0.50:
        decision = "MANUAL_REVIEW"
        risk_tier = "HIGH"
    elif prob >= 0.25:
        decision = "APPROVE_WITH_CHALLENGE"
        risk_tier = "ELEVATED"
    else:
        decision = "APPROVE"
        risk_tier = "LOW"

    latency_ms = round((time.time() - t0) * 1000, 2)
    STATE["inference_count"] = STATE.get("inference_count", 0) + 1

    return PredictionResponse(
        transaction_id=tx.transaction_id,
        fraud_probability=round(prob, 4),
        decision=decision,
        risk_tier=risk_tier,
        inference_latency_ms=latency_ms,
        factors=factors or ["Normal transactional pattern"],
        model_version=STATE["model_bundle"].get("run_id", "v1.0.0")[:8]
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health status and telemetry."""
    return HealthResponse(
        status="HEALTHY",
        service="SentinX Fraud Scoring Engine",
        model_version=STATE["model_bundle"].get("run_id", "v1")[:8] if STATE["model_bundle"] else "unloaded",
        total_inferences=STATE["inference_count"],
        uptime_seconds=round(time.time() - STATE["start_time"], 2)
    )


@app.post("/predict/realtime", response_model=PredictionResponse, tags=["Inference"])
async def predict_realtime(payload: TransactionPayload):
    """Real-time single transaction fraud scoring with sub-20ms latency."""
    return _score_transaction(payload)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(payload: BatchPredictionRequest):
    """Batch transaction scoring endpoint for async queues and historical backfills."""
    t0 = time.time()
    results = [_score_transaction(tx) for tx in payload.transactions]
    flagged = sum(1 for r in results if r.decision in ["DECLINE_FRAUD", "MANUAL_REVIEW"])
    elapsed_ms = (time.time() - t0) * 1000
    avg_latency = round(elapsed_ms / max(1, len(results)), 2)

    return BatchPredictionResponse(
        total_processed=len(results),
        flagged_count=flagged,
        avg_latency_ms=avg_latency,
        results=results
    )


@app.get("/drift/latest", tags=["Monitoring"])
async def get_latest_drift():
    """Returns the most recent data drift evaluation report."""
    drift_file = REPORTS_DIR / "data_drift_report.json"
    if not drift_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No drift report found. Run drift monitor first.")
    with open(drift_file, "r", encoding="utf-8") as f:
        return json.load(f)

"""SentinX API Pydantic Schemas for Real-Time Model Serving."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TransactionPayload(BaseModel):
    transaction_id: str = Field(..., json_schema_extra={"example": "TX_89F1A42B9C10"})
    user_id: str = Field(..., json_schema_extra={"example": "USR_00124"})
    merchant_id: str = Field(..., json_schema_extra={"example": "MER_0018"})
    amount: float = Field(..., gt=0.0, json_schema_extra={"example": 1250.00})
    channel: str = Field(default="web", json_schema_extra={"example": "web"})
    card_type: str = Field(default="credit", json_schema_extra={"example": "credit"})
    device_type: str = Field(default="windows", json_schema_extra={"example": "windows"})
    country: str = Field(default="US", json_schema_extra={"example": "US"})
    distance_from_home_km: float = Field(default=15.0, ge=0.0, json_schema_extra={"example": 450.0})
    hour_of_day: Optional[int] = Field(default=14, ge=0, le=23)
    timestamp: Optional[str] = Field(default=None, json_schema_extra={"example": "2026-09-15 14:30:00"})


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str  # "APPROVE", "APPROVE_WITH_CHALLENGE", "MANUAL_REVIEW", "DECLINE_FRAUD"
    risk_tier: str  # "LOW", "ELEVATED", "HIGH", "CRITICAL"
    inference_latency_ms: float
    factors: List[str]
    model_version: str


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionPayload]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    flagged_count: int
    avg_latency_ms: float
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    service: str
    model_version: str
    total_inferences: int
    uptime_seconds: float

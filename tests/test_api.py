"""Integration tests for SentinX FastAPI serving microservice."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "HEALTHY"
        assert "uptime_seconds" in data


def test_predict_realtime_legitimate_tx():
    with TestClient(app) as test_client:
        payload = {
            "transaction_id": "TX_TEST_LEGIT_001",
            "user_id": "USR_00001",
            "merchant_id": "MER_0001",
            "amount": 25.50,
            "channel": "mobile_app",
            "card_type": "debit",
            "device_type": "ios",
            "country": "US",
            "distance_from_home_km": 5.2,
            "hour_of_day": 14
        }
        response = test_client.post("/predict/realtime", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["transaction_id"] == "TX_TEST_LEGIT_001"
        assert "fraud_probability" in data
        assert data["decision"] in ["APPROVE", "APPROVE_WITH_CHALLENGE", "MANUAL_REVIEW", "DECLINE_FRAUD"]
        assert data["inference_latency_ms"] >= 0.0


def test_predict_realtime_anomalous_fraud():
    with TestClient(app) as test_client:
        payload = {
            "transaction_id": "TX_TEST_FRAUD_002",
            "user_id": "USR_00001",
            "merchant_id": "MER_0001",
            "amount": 4999.00,  # Extreme surge
            "channel": "api",
            "card_type": "virtual_card",
            "device_type": "linux",
            "country": "RU",
            "distance_from_home_km": 8500.0,  # Impossible travel
            "hour_of_day": 3  # Late night
        }
        response = test_client.post("/predict/realtime", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["fraud_probability"] > 0.40
        assert data["risk_tier"] in ["HIGH", "CRITICAL"]

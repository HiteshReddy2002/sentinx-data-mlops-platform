"""Unit tests for SentinX MLOps Feature Engineering & Drift Monitor."""

import numpy as np
import pandas as pd
from src.generator.transaction_stream import TransactionGenerator
from src.ml.drift_monitor import DriftMonitor
from src.ml.features import FeaturePipeline


def test_feature_pipeline_fit_and_transform():
    generator = TransactionGenerator(num_users=25, num_merchants=8)
    tx, users, merchants = generator.generate_batch(num_transactions=100)

    pipeline = FeaturePipeline()
    features_df = pipeline.compute_features(tx, users, merchants)

    assert "amount_to_baseline_ratio" in features_df.columns
    assert "merchant_risk_score" in features_df.columns

    X, y = pipeline.fit_transform(features_df)
    assert X.shape[0] == 100
    assert len(pipeline.feature_names) == X.shape[1]

    # Test transform on new batch
    X_new = pipeline.transform(features_df.head(10))
    assert X_new.shape[0] == 10
    assert X_new.shape[1] == X.shape[1]


def test_drift_monitor_detects_distribution_shift():
    generator = TransactionGenerator(num_users=20, num_merchants=5)
    tx_ref, _, _ = generator.generate_batch(num_transactions=200)

    # Induce massive shift in amounts
    tx_curr = tx_ref.copy()
    tx_curr["amount"] = tx_curr["amount"] * 5.0

    monitor = DriftMonitor(psi_threshold=0.15)
    report = monitor.evaluate_drift(tx_ref, tx_curr, numerical_features=["amount"])

    assert report["overall_status"] == "DRIFT_ALERT"
    assert "amount" in report["drifting_features"]
    assert report["feature_metrics"]["amount"]["drift_detected"] is True

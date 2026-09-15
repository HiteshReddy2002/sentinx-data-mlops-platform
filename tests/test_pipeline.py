"""Unit tests for SentinX Data Engineering Pipeline & Quality Gates."""

import pandas as pd
from src.generator.transaction_stream import TransactionGenerator
from src.pipeline.quality import DataQualityGate
from src.pipeline.transformations import MedallionTransformer


def test_quality_gate_passes_on_valid_data():
    generator = TransactionGenerator(num_users=30, num_merchants=10)
    tx, users, merchants = generator.generate_batch(num_transactions=150)

    gate = DataQualityGate()
    result = gate.validate_transactions(tx, users, merchants)

    assert result["passed"] is True
    assert result["checks_run"] >= 7
    assert len(result["failures"]) == 0


def test_quality_gate_catches_domain_violations():
    generator = TransactionGenerator(num_users=20, num_merchants=5)
    tx, users, merchants = generator.generate_batch(num_transactions=100)

    # Corrupt data with negative amount and orphan user
    corrupted_tx = tx.copy()
    corrupted_tx.loc[0, "amount"] = -50.0
    corrupted_tx.loc[1, "user_id"] = "NON_EXISTENT_USR"

    gate = DataQualityGate()
    result = gate.validate_transactions(corrupted_tx, users, merchants)

    assert result["passed"] is False
    assert len(result["failures"]) >= 2

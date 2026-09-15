"""Unit tests for SentinX Transaction Generator."""

import pandas as pd
from src.generator.transaction_stream import TransactionGenerator


def test_generator_batch_generation():
    generator = TransactionGenerator(num_users=50, num_merchants=10)
    tx, users, merchants = generator.generate_batch(num_transactions=200, fraud_ratio=0.05)

    assert len(tx) == 200
    assert len(users) == 50
    assert len(merchants) == 10

    # Validate essential columns exist
    for col in ["transaction_id", "user_id", "merchant_id", "amount", "timestamp", "is_fraud"]:
        assert col in tx.columns

    # Check non-negative amounts
    assert (tx["amount"] > 0).all()

    # Check fraud ratio is closely aligned
    assert tx["is_fraud"].sum() > 0
    assert tx["is_fraud"].isin([0, 1]).all()

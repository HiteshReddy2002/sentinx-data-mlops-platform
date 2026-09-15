"""SentinX Data Quality & Contract Testing Engine.

Implements automated schema validation, null checks, range invariants,
uniqueness constraints, and referential integrity assertions.
"""

from typing import Dict, List
import pandas as pd


class DataQualityGate:
    """Evaluates data quality contracts on transactional datasets."""

    def __init__(self):
        self.validation_results = []

    def validate_transactions(self, df_tx: pd.DataFrame, df_users: pd.DataFrame, df_merchants: pd.DataFrame) -> Dict:
        """Runs a battery of rigorous data quality checks."""
        failures = []
        checks_run = 0

        # Check 1: Primary Key Uniqueness
        checks_run += 1
        pk_duplicates = df_tx["transaction_id"].duplicated().sum()
        if pk_duplicates > 0:
            failures.append(f"Primary key violation: {pk_duplicates} duplicate transaction_ids found.")

        # Check 2: Non-null Constraints
        checks_run += 1
        null_cols = df_tx[["transaction_id", "user_id", "merchant_id", "amount", "timestamp"]].isnull().sum()
        if null_cols.any():
            failing = null_cols[null_cols > 0].to_dict()
            failures.append(f"Null constraint violation in critical columns: {failing}")

        # Check 3: Domain Invariants (Amount must be strictly positive)
        checks_run += 1
        invalid_amounts = (df_tx["amount"] <= 0).sum()
        if invalid_amounts > 0:
            failures.append(f"Amount domain violation: {invalid_amounts} transactions with amount <= 0.")

        # Check 4: Distance invariant (Distance cannot be negative)
        checks_run += 1
        invalid_distance = (df_tx["distance_from_home_km"] < 0).sum()
        if invalid_distance > 0:
            failures.append(f"Distance invariant violation: {invalid_distance} rows with negative distance.")

        # Check 5: Referential Integrity (Every transaction user_id exists in users)
        checks_run += 1
        user_ids = set(df_users["user_id"])
        orphan_users = (~df_tx["user_id"].isin(user_ids)).sum()
        if orphan_users > 0:
            failures.append(f"Referential integrity failure: {orphan_users} transactions reference non-existent user_ids.")

        # Check 6: Referential Integrity (Every merchant_id exists in merchants)
        checks_run += 1
        merchant_ids = set(df_merchants["merchant_id"])
        orphan_merchants = (~df_tx["merchant_id"].isin(merchant_ids)).sum()
        if orphan_merchants > 0:
            failures.append(f"Referential integrity failure: {orphan_merchants} transactions reference non-existent merchant_ids.")

        # Check 7: Label Consistency (is_fraud binary check)
        checks_run += 1
        invalid_fraud_labels = (~df_tx["is_fraud"].isin([0, 1])).sum()
        if invalid_fraud_labels > 0:
            failures.append(f"Fraud label invalid: {invalid_fraud_labels} rows with non-binary values.")

        passed = len(failures) == 0
        summary = {
            "checks_run": checks_run,
            "passed": passed,
            "total_records_tested": len(df_tx),
            "failures": failures,
            "status": "PASSED" if passed else "FAILED"
        }
        return summary

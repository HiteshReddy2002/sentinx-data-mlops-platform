"""SentinX Bronze Layer Ingestion Engine.

Ingests raw streaming and batch event payloads into the immutable Bronze Data Lake (Parquet format)
with cryptographic checksums and ingestion audit logs.
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, Tuple
import pandas as pd

from config.settings import BRONZE_DIR
from src.generator.transaction_stream import TransactionGenerator


class BronzeIngestionEngine:
    """Handles raw event ingestion and immutable Bronze layer storage."""

    def __init__(self, bronze_dir=BRONZE_DIR):
        self.bronze_dir = bronze_dir
        self.audit_log_path = self.bronze_dir / "_ingestion_audit_log.jsonl"

    def _calculate_checksum(self, df: pd.DataFrame) -> str:
        """Computes a SHA256 checksum over the serialized dataframe bytes."""
        return hashlib.sha256(df.to_parquet()).hexdigest()

    def _log_audit_entry(self, entry: Dict):
        """Appends an ingestion event to the audit ledger."""
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def ingest_batch(
        self,
        num_transactions: int = 15000,
        fraud_ratio: float = 0.038,
        batch_label: str = "initial_load"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """Generates and writes raw events into Bronze storage."""
        generator = TransactionGenerator(num_users=600, num_merchants=80)
        df_tx, df_users, df_merchants = generator.generate_batch(
            num_transactions=num_transactions,
            fraud_ratio=fraud_ratio
        )

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = f"batch_{timestamp_str}_{batch_label}"

        # Write immutable raw parquet files
        tx_filename = f"raw_transactions_{batch_id}.parquet"
        users_filename = "raw_users.parquet"
        merchants_filename = "raw_merchants.parquet"

        tx_path = self.bronze_dir / tx_filename
        users_path = self.bronze_dir / users_filename
        merchants_path = self.bronze_dir / merchants_filename

        df_tx.to_parquet(tx_path, index=False)
        df_users.to_parquet(users_path, index=False)
        df_merchants.to_parquet(merchants_path, index=False)

        audit_record = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "transaction_count": len(df_tx),
            "users_count": len(df_users),
            "merchants_count": len(df_merchants),
            "fraud_count": int(df_tx["is_fraud"].sum()),
            "sha256_checksum": self._calculate_checksum(df_tx),
            "tx_path": str(tx_path),
            "status": "INGESTION_SUCCESS"
        }
        self._log_audit_entry(audit_record)

        return df_tx, df_users, df_merchants, audit_record


if __name__ == "__main__":
    engine = BronzeIngestionEngine()
    tx, u, m, audit = engine.ingest_batch(num_transactions=1000)
    print(f"Ingested {audit['transaction_count']} raw records. Checksum: {audit['sha256_checksum'][:16]}...")

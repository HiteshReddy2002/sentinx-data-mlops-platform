"""SentinX Pipeline Orchestrator.

Coordinates ingestion, contract validation, and medallion data transformations
with comprehensive execution state tracking, failure handling, and runtime telemetry.
"""

import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict

from config.settings import REPORTS_DIR
from src.pipeline.ingestion import BronzeIngestionEngine
from src.pipeline.quality import DataQualityGate
from src.pipeline.transformations import MedallionTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SentinX-Orchestrator")


class PipelineOrchestrator:
    """Manages DAG execution across Bronze, Silver, and Gold pipeline stages."""

    def __init__(self):
        self.ingestion_engine = BronzeIngestionEngine()
        self.quality_gate = DataQualityGate()
        self.transformer = MedallionTransformer()

    def run_pipeline(self, num_transactions: int = 15000, fraud_ratio: float = 0.038) -> Dict:
        """Executes the full end-to-end data pipeline."""
        start_time = time.time()
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f" Starting SentinX Data Pipeline Execution: {run_id}")

        pipeline_telemetry = {
            "run_id": run_id,
            "started_at": datetime.now().isoformat(),
            "stages": {}
        }

        try:
            # Stage 1: Bronze Ingestion
            logger.info("Stage 1/4: Ingesting raw streaming & batch transactions into Bronze Lake...")
            t0 = time.time()
            df_tx, df_users, df_merchants, audit_record = self.ingestion_engine.ingest_batch(
                num_transactions=num_transactions,
                fraud_ratio=fraud_ratio,
                batch_label=run_id
            )
            pipeline_telemetry["stages"]["bronze_ingestion"] = {
                "duration_seconds": round(time.time() - t0, 3),
                "records_ingested": len(df_tx),
                "fraud_records": audit_record["fraud_count"],
                "status": "COMPLETED"
            }
            logger.info(f" Bronze ingestion complete: {len(df_tx)} transactions.")

            # Stage 2: Quality Gates
            logger.info("Stage 2/4: Running data contract assertions and schema tests...")
            t0 = time.time()
            quality_report = self.quality_gate.validate_transactions(df_tx, df_users, df_merchants)
            pipeline_telemetry["stages"]["quality_gates"] = {
                "duration_seconds": round(time.time() - t0, 3),
                "checks_run": quality_report["checks_run"],
                "passed": quality_report["passed"],
                "status": quality_report["status"]
            }
            if not quality_report["passed"]:
                logger.error(f" Quality Gate Failed: {quality_report['failures']}")
                raise ValueError(f"Pipeline halted due to data contract violations: {quality_report['failures']}")
            logger.info(f" All {quality_report['checks_run']} data quality checks passed successfully.")

            # Stage 3: Silver Cleansing & Temporal Enrichment
            logger.info("Stage 3/4: Processing Silver layer cleansing and feature enrichment...")
            t0 = time.time()
            silver_tx, silver_users, silver_merchants = self.transformer.transform_to_silver(
                df_tx, df_users, df_merchants
            )
            pipeline_telemetry["stages"]["silver_transformation"] = {
                "duration_seconds": round(time.time() - t0, 3),
                "silver_tx_count": len(silver_tx),
                "status": "COMPLETED"
            }
            logger.info(f" Silver layer ready: {len(silver_tx)} enriched transactions.")

            # Stage 4: Gold Star Schema & DuckDB Warehouse
            logger.info("Stage 4/4: Modeling Gold dimensional warehouse (Star Schema & Views)...")
            t0 = time.time()
            gold_counts = self.transformer.build_gold_warehouse(silver_tx, silver_users, silver_merchants)
            pipeline_telemetry["stages"]["gold_warehouse"] = {
                "duration_seconds": round(time.time() - t0, 3),
                "mart_counts": gold_counts,
                "status": "COMPLETED"
            }
            logger.info(f" Gold warehouse modeled successfully: {gold_counts}")

            total_elapsed = round(time.time() - start_time, 3)
            pipeline_telemetry["completed_at"] = datetime.now().isoformat()
            pipeline_telemetry["total_duration_seconds"] = total_elapsed
            pipeline_telemetry["status"] = "SUCCESS"

            # Save execution telemetry report
            report_file = REPORTS_DIR / f"pipeline_run_{run_id}.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(pipeline_telemetry, f, indent=2)

            logger.info(f" SentinX Data Pipeline Run {run_id} Succeeded in {total_elapsed}s. Report saved to {report_file.name}")
            return pipeline_telemetry

        except Exception as e:
            pipeline_telemetry["status"] = "FAILED"
            pipeline_telemetry["error"] = str(e)
            logger.exception(f" Pipeline execution failed: {e}")
            raise


if __name__ == "__main__":
    orchestrator = PipelineOrchestrator()
    orchestrator.run_pipeline(num_transactions=10000)

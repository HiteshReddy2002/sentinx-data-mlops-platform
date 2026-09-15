"""SentinX CLI Entrypoint."""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="SentinX: Enterprise FinTech Data & MLOps Command Line Interface"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available platform commands")

    subparsers.add_parser("pipeline", help="Run Bronze -> Silver -> Gold data pipeline orchestrator")
    subparsers.add_parser("train", help="Train LightGBM fraud model with MLflow experiment tracking")
    subparsers.add_parser("drift", help="Run statistical covariate & model drift monitoring")
    subparsers.add_parser("serve", help="Launch FastAPI real-time model serving microservice")
    subparsers.add_parser("dashboard", help="Launch Streamlit Executive & BI Analytics portal")
    subparsers.add_parser("test", help="Run full automated test suite with pytest")

    args = parser.parse_args()

    if args.command == "pipeline":
        from src.pipeline.orchestrator import PipelineOrchestrator
        orchestrator = PipelineOrchestrator()
        orchestrator.run_pipeline()
    elif args.command == "train":
        from src.ml.train import ModelTrainer
        trainer = ModelTrainer()
        trainer.train_and_evaluate()
    elif args.command == "drift":
        from src.ml.drift_monitor import DriftMonitor
        import pandas as pd
        from config.settings import GOLD_DIR
        df_ref = pd.read_parquet(GOLD_DIR / "gold_fct_transactions.parquet")
        monitor = DriftMonitor()
        report = monitor.evaluate_drift(df_ref, df_ref)
        print("Drift Monitoring Check Complete. Overall Status:", report["overall_status"])
    elif args.command == "serve":
        subprocess.run([sys.executable, "-m", "uvicorn", "src.api.main:app", "--reload", "--port", "8000"])
    elif args.command == "dashboard":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"])
    elif args.command == "test":
        subprocess.run([sys.executable, "-m", "pytest", "-v"])
    else:
        print("\n🛡️ SentinX FinTech Data & MLOps Platform")
        print("Run with a subcommand, for example:")
        print("  python main.py pipeline    # Run end-to-end data pipeline")
        print("  python main.py train       # Train model & log to MLflow")
        print("  python main.py dashboard   # Launch Streamlit dashboard")
        print("  python main.py serve       # Launch FastAPI microservice")
        print("  python main.py test        # Run pytest test suite\n")


if __name__ == "__main__":
    main()

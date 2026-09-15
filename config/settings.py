import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
for directory in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database configuration
DUCKDB_PATH = GOLD_DIR / "sentinx_warehouse.duckdb"

# MLflow Tracking
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
MLFLOW_DB_PATH = (BASE_DIR / "mlflow.db").resolve().as_posix()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")
MLFLOW_EXPERIMENT_NAME = "sentinx-fraud-detection"
PRODUCTION_MODEL_NAME = "sentinx-fraud-classifier"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Fraud detection thresholds
FRAUD_RISK_THRESHOLD = float(os.getenv("FRAUD_RISK_THRESHOLD", 0.65))
ALERT_HIGH_RISK_THRESHOLD = 0.85

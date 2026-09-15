"""SentinX ML Training, Experiment Tracking & Model Registry Pipeline.

Trains an industrial LightGBM fraud detection model with cost-sensitive weighting,
logs hyperparameter experiments and evaluation metrics to MLflow, and registers the production model.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from config.settings import (
    GOLD_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.ml.features import FeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SentinX-Trainer")


class ModelTrainer:
    """Orchestrates ML model training, evaluation, MLflow tracking, and model serialization."""

    def __init__(self):
        self.feature_pipeline = FeaturePipeline()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def train_and_evaluate(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Trains LightGBM classifier with full experiment tracking and artifact logging."""
        logger.info("Loading Gold layer dimensional datasets...")
        df_tx = pd.read_parquet(GOLD_DIR / "gold_fct_transactions.parquet")
        df_users = pd.read_parquet(GOLD_DIR / "gold_dim_users.parquet")
        df_merchants = pd.read_parquet(GOLD_DIR / "gold_dim_merchants.parquet")

        logger.info(f"Loaded {len(df_tx)} transactions with {df_tx['is_fraud'].sum()} confirmed fraud incidents.")

        # Compute engineered features
        df_features = self.feature_pipeline.compute_features(df_tx, df_users, df_merchants)
        X, y = self.feature_pipeline.fit_transform(df_features)
        feature_names = self.feature_pipeline.feature_names

        # Save feature pipeline artifact
        self.feature_pipeline.save()

        # Train/Test split (stratified by fraud class)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        fraud_count = int(y_train.sum())
        non_fraud_count = len(y_train) - fraud_count
        scale_pos_weight = non_fraud_count / max(1, fraud_count)

        params = {
            "n_estimators": 120,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "scale_pos_weight": round(scale_pos_weight * 0.75, 2),  # Tuned balance between recall and precision
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "n_jobs": -1
        }

        with mlflow.start_run(run_name=f"lightgbm_fraud_{int(time.time())}") as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow Run: {run_id}")

            # Log Hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("fraud_ratio_train", round(fraud_count / len(y_train), 4))

            # Train Model
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train, y_train)

            # Predictions & Probabilities
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            # Metrics Computation
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            metrics = {
                "roc_auc": round(float(roc_auc), 4),
                "pr_auc": round(float(pr_auc), 4),
                "f1_score": round(float(f1), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "true_positives": int(cm[1, 1]),
                "false_positives": int(cm[0, 1]),
                "true_negatives": int(cm[0, 0]),
                "false_negatives": int(cm[1, 0])
            }
            mlflow.log_metrics(metrics)
            logger.info(f" Model Evaluation Results: PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")

            # Artifact 1: Feature Importance Chart
            importances = clf.feature_importances_
            sorted_idx = np.argsort(importances)[::-1][:12]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh([feature_names[i] for i in sorted_idx[::-1]], importances[sorted_idx[::-1]], color="#1E88E5")
            ax.set_title("Top 12 Fraud Prediction Features (LightGBM)")
            ax.set_xlabel("Importance Score")
            plt.tight_layout()
            feat_imp_path = REPORTS_DIR / "feature_importance.png"
            fig.savefig(feat_imp_path)
            plt.close(fig)
            mlflow.log_artifact(str(feat_imp_path))

            # Artifact 2: Precision-Recall Curve Chart
            prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(rec, prec, color="#D81B60", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve for Imbalanced Fraud Detection")
            ax.legend(loc="lower left")
            plt.tight_layout()
            pr_curve_path = REPORTS_DIR / "precision_recall_curve.png"
            fig.savefig(pr_curve_path)
            plt.close(fig)
            mlflow.log_artifact(str(pr_curve_path))

            # Serialize model package for low-latency serving
            model_artifact_path = MODELS_DIR / "sentinx_fraud_detector.joblib"
            model_bundle = {
                "model": clf,
                "feature_names": feature_names,
                "metrics": metrics,
                "run_id": run_id,
                "threshold": 0.50
            }
            joblib.dump(model_bundle, model_artifact_path)
            mlflow.log_artifact(str(model_artifact_path))

            # Save summary report
            summary = {
                "run_id": run_id,
                "metrics": metrics,
                "params": params,
                "top_features": [feature_names[i] for i in sorted_idx[:5]],
                "model_path": str(model_artifact_path)
            }
            with open(REPORTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            return summary


if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate()
    print("Training Complete! Summary:", json.dumps(results, indent=2))

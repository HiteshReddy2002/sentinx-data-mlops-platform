"""SentinX Feature Engineering Pipeline.

Extracts customer behavioral signals, transactional velocity ratios,
and risk vectors from Gold/Silver tables for ML training and real-time online inference.
"""

from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import MODELS_DIR

NUMERICAL_FEATURES = [
    "amount",
    "distance_from_home_km",
    "hour_of_day",
    "is_weekend",
    "is_night",
    "is_cross_border",
    "account_age_days",
    "kyc_verified",
    "baseline_spend",
    "amount_to_baseline_ratio",
    "chargeback_history_rate",
    "merchant_risk_score"
]

CATEGORICAL_FEATURES = ["channel", "card_type", "device_type"]

RISK_TIER_MAP = {"Low": 0.1, "Medium": 0.5, "High": 0.9}


class FeaturePipeline:
    """Handles feature engineering and stateful preprocessing for training and online serving."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.fitted = False
        self.feature_names: List[str] = []

    def compute_features(self, df_tx: pd.DataFrame, df_users: pd.DataFrame, df_merchants: pd.DataFrame) -> pd.DataFrame:
        """Enriches raw transactional data with user profile and merchant risk signals."""
        users_lookup = df_users.set_index("user_id")[["account_age_days", "kyc_verified", "baseline_spend", "country"]]
        merchants_lookup = df_merchants.set_index("merchant_id")[["risk_tier", "chargeback_history_rate"]]

        df = df_tx.copy()
        
        # Merge profile metadata
        df = df.join(users_lookup, on="user_id", rsuffix="_user")
        df = df.join(merchants_lookup, on="merchant_id", rsuffix="_merchant")

        # Derive velocity and anomaly ratios
        df["amount_to_baseline_ratio"] = df["amount"] / df["baseline_spend"].clip(lower=1.0)
        df["merchant_risk_score"] = df["risk_tier"].map(RISK_TIER_MAP).fillna(0.2)

        # Temporal signals
        if "hour_of_day" not in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            df["hour_of_day"] = ts.dt.hour
            df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
            df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if h < 6 or h > 22 else 0)

        # Cross border flag
        if "is_cross_border" not in df.columns:
            df["is_cross_border"] = (df["country"] != df["country_user"]).astype(int)

        # Fill any missing values
        df["distance_from_home_km"] = df["distance_from_home_km"].fillna(10.0)
        df["chargeback_history_rate"] = df["chargeback_history_rate"].fillna(0.01)

        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fits encoders and scalers on training data and returns feature matrix X and labels y."""
        X_num = df[NUMERICAL_FEATURES].values
        X_cat = self.ohe.fit_transform(df[CATEGORICAL_FEATURES])

        X_num_scaled = self.scaler.fit_transform(X_num)
        X = np.hstack([X_num_scaled, X_cat])

        ohe_features = list(self.ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        self.feature_names = NUMERICAL_FEATURES + ohe_features
        self.fitted = True

        y = df["is_fraud"].values if "is_fraud" in df.columns else np.zeros(len(df))
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms unseen data for inference using fitted preprocessors."""
        if not self.fitted:
            raise RuntimeError("FeaturePipeline must be fitted before calling transform().")

        X_num = df[NUMERICAL_FEATURES].values
        X_cat = self.ohe.transform(df[CATEGORICAL_FEATURES])
        X_num_scaled = self.scaler.transform(X_num)
        return np.hstack([X_num_scaled, X_cat])

    def save(self, filepath=None):
        """Serializes the fitted pipeline for serving."""
        filepath = filepath or (MODELS_DIR / "feature_pipeline.joblib")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath=None) -> "FeaturePipeline":
        """Loads a pre-trained feature pipeline."""
        filepath = filepath or (MODELS_DIR / "feature_pipeline.joblib")
        return joblib.load(filepath)

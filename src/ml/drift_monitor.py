"""SentinX Data & Model Drift Monitoring Engine.

Detects covariate shift, feature distribution drift, and prediction drift
between reference baseline data and live incoming inference traffic using
Kolmogorov-Smirnov tests, Population Stability Index (PSI), and Wasserstein metrics.
"""

import json
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from config.settings import GOLD_DIR, REPORTS_DIR

logger = logging.getLogger("SentinX-DriftMonitor")


class DriftMonitor:
    """Computes statistical drift scores and generates operational alerts."""

    def __init__(self, psi_threshold: float = 0.20, ks_pvalue_threshold: float = 0.05):
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
        """Calculates Population Stability Index (PSI) between reference and current samples."""
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create quantile bins based on reference
        percentiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(expected, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        expected_pct = np.clip(expected_counts / len(expected), 1e-4, 1.0)
        actual_pct = np.clip(actual_counts / len(actual), 1e-4, 1.0)

        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi_val)

    def evaluate_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        numerical_features: List[str] = None
    ) -> Dict:
        """Evaluates feature drift across numerical distributions."""
        if numerical_features is None:
            numerical_features = [
                "amount",
                "distance_from_home_km",
                "hour_of_day",
                "account_age_days",
                "baseline_spend"
            ]

        drift_results = {}
        drifting_features = []

        for feature in numerical_features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue

            ref_vals = reference_df[feature].dropna().values
            curr_vals = current_df[feature].dropna().values

            # 1. Kolmogorov-Smirnov 2-sample test
            ks_stat, p_value = ks_2samp(ref_vals, curr_vals)

            # 2. Population Stability Index (PSI)
            psi_score = self.calculate_psi(ref_vals, curr_vals)

            # 3. Wasserstein Distance
            w_dist = wasserstein_distance(ref_vals, curr_vals)

            # Determine drift status
            is_drifting = bool(psi_score > self.psi_threshold or p_value < self.ks_pvalue_threshold)
            if is_drifting:
                drifting_features.append(feature)

            drift_results[feature] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 5),
                "psi_score": round(float(psi_score), 4),
                "wasserstein_distance": round(float(w_dist), 4),
                "drift_detected": is_drifting,
                "severity": "HIGH" if psi_score > 0.25 else ("MODERATE" if psi_score > 0.10 else "NEGLIGIBLE")
            }

        overall_status = "DRIFT_ALERT" if len(drifting_features) > 0 else "STABLE"
        report = {
            "overall_status": overall_status,
            "reference_sample_size": len(reference_df),
            "current_sample_size": len(current_df),
            "drifting_features_count": len(drifting_features),
            "drifting_features": drifting_features,
            "feature_metrics": drift_results
        }

        # Save drift report to disk
        out_path = REPORTS_DIR / "data_drift_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


if __name__ == "__main__":
    df_ref = pd.read_parquet(GOLD_DIR / "gold_fct_transactions.parquet")
    # Simulate a drifted batch with shifted amount and distance distributions
    df_curr = df_ref.sample(1500, random_state=42).copy()
    df_curr["amount"] = df_curr["amount"] * 1.8  # Significant amount shift
    
    monitor = DriftMonitor()
    report = monitor.evaluate_drift(df_ref, df_curr)
    print("Drift Evaluation Complete:", json.dumps(report, indent=2))

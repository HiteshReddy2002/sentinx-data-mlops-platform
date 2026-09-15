"""SentinX Synthetic FinTech Transaction Stream Generator.

Generates high-fidelity simulated streaming & batch financial transactions
with realistic consumer behavior, merchant networks, and sophisticated fraud attack vectors
(Velocity attacks, impossible travel, card testing/probing, and luxury crypto surges).
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

MERCHANT_CATEGORIES = {
    "grocery": {"avg_amt": 55.0, "std_amt": 30.0, "risk_tier": "Low"},
    "subscription": {"avg_amt": 18.0, "std_amt": 8.0, "risk_tier": "Low"},
    "restaurant": {"avg_amt": 42.0, "std_amt": 25.0, "risk_tier": "Low"},
    "electronics": {"avg_amt": 380.0, "std_amt": 250.0, "risk_tier": "Medium"},
    "travel": {"avg_amt": 540.0, "std_amt": 400.0, "risk_tier": "Medium"},
    "luxury_goods": {"avg_amt": 1250.0, "std_amt": 800.0, "risk_tier": "High"},
    "crypto_exchange": {"avg_amt": 1800.0, "std_amt": 1200.0, "risk_tier": "High"},
    "gaming_gambling": {"avg_amt": 220.0, "std_amt": 180.0, "risk_tier": "High"},
}

CHANNELS = ["mobile_app", "web", "in_store", "api"]
CARD_TYPES = ["credit", "debit", "virtual_card"]
DEVICES = ["ios", "android", "macos", "windows", "linux"]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "SG", "AU", "NG", "RU", "BR"]


class TransactionGenerator:
    """Generates synthetic users, merchants, and transactional event streams."""

    def __init__(self, num_users: int = 500, num_merchants: int = 60):
        self.num_users = num_users
        self.num_merchants = num_merchants
        self.users = self._generate_users()
        self.merchants = self._generate_merchants()

    def _generate_users(self) -> List[Dict]:
        users = []
        for i in range(1, self.num_users + 1):
            user_country = np.random.choice(["US", "CA", "GB", "DE"], p=[0.7, 0.1, 0.1, 0.1])
            account_age_days = random.randint(10, 1500)
            kyc_verified = random.random() > 0.15
            baseline_spend = random.uniform(20.0, 150.0)
            users.append({
                "user_id": f"USR_{i:05d}",
                "user_name": fake.name(),
                "email": fake.email(),
                "country": user_country,
                "account_age_days": account_age_days,
                "kyc_verified": kyc_verified,
                "baseline_spend": baseline_spend,
                "created_at": (datetime.now() - timedelta(days=account_age_days)).strftime("%Y-%m-%d %H:%M:%S")
            })
        return users

    def _generate_merchants(self) -> List[Dict]:
        merchants = []
        categories = list(MERCHANT_CATEGORIES.keys())
        for i in range(1, self.num_merchants + 1):
            cat = categories[i % len(categories)]
            meta = MERCHANT_CATEGORIES[cat]
            merchants.append({
                "merchant_id": f"MER_{i:04d}",
                "merchant_name": f"{fake.company()} {cat.capitalize()}",
                "category": cat,
                "risk_tier": meta["risk_tier"],
                "country": np.random.choice(["US", "GB", "CA", "DE", "SG"], p=[0.6, 0.15, 0.1, 0.1, 0.05]),
                "chargeback_history_rate": round(random.uniform(0.001, 0.04), 4),
            })
        return merchants

    def generate_batch(
        self,
        num_transactions: int = 10000,
        fraud_ratio: float = 0.035,
        start_time: datetime = None,
        duration_days: int = 30,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generates historical or batch transaction records with realistic fraud vectors."""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=duration_days)

        transactions = []
        num_fraud = int(num_transactions * fraud_ratio)
        num_legit = num_transactions - num_fraud

        # 1. Generate Legitimate Transactions
        for _ in range(num_legit):
            user = random.choice(self.users)
            merchant = random.choice(self.merchants)
            cat_meta = MERCHANT_CATEGORIES[merchant["category"]]

            # Normal amount around category baseline
            amount = max(2.0, np.random.normal(cat_meta["avg_amt"], cat_meta["std_amt"]))
            
            # Timestamp uniformly distributed over period with diurnal peak (day vs night)
            random_offset_sec = random.randint(0, duration_days * 86400)
            tx_time = start_time + timedelta(seconds=random_offset_sec)
            
            # Distance from home (usually close)
            distance_km = round(float(np.random.exponential(scale=12.0)), 2)
            
            tx = {
                "transaction_id": f"TX_{uuid.uuid4().hex[:12].upper()}",
                "user_id": user["user_id"],
                "merchant_id": merchant["merchant_id"],
                "amount": round(float(amount), 2),
                "currency": "USD",
                "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "channel": random.choice(CHANNELS),
                "card_type": random.choice(CARD_TYPES),
                "device_type": random.choice(DEVICES),
                "ip_address": fake.ipv4_public(),
                "country": user["country"],
                "distance_from_home_km": distance_km,
                "is_fraud": 0,
                "fraud_type": "NONE"
            }
            transactions.append(tx)

        # 2. Generate Fraudulent Transactions with Specific Attack Signatures
        fraud_patterns = ["VELOCITY_BURST", "IMPOSSIBLE_TRAVEL", "CARD_PROBING", "HIGH_RISK_SURGE"]
        for _ in range(num_fraud):
            pattern = random.choice(fraud_patterns)
            user = random.choice(self.users)
            random_offset_sec = random.randint(0, duration_days * 86400)
            tx_time = start_time + timedelta(seconds=random_offset_sec)

            if pattern == "VELOCITY_BURST":
                # Multiple large transactions in quick succession
                merchant = random.choice([m for m in self.merchants if m["risk_tier"] in ["Medium", "High"]])
                amount = random.uniform(800.0, 3500.0)
                distance_km = round(random.uniform(50.0, 400.0), 2)
                country = user["country"]

            elif pattern == "IMPOSSIBLE_TRAVEL":
                # Foreign country, massive distance
                merchant = random.choice(self.merchants)
                amount = random.uniform(250.0, 2000.0)
                distance_km = round(random.uniform(4000.0, 12000.0), 2)
                country = np.random.choice(["NG", "RU", "BR", "SG"])

            elif pattern == "CARD_PROBING":
                # Testing card with abnormal odd amount or sudden spike
                merchant = random.choice([m for m in self.merchants if m["category"] in ["gaming_gambling", "subscription"]])
                amount = random.choice([0.99, 1.49, 1899.99, 2999.00])
                distance_km = round(random.uniform(200.0, 1500.0), 2)
                country = np.random.choice(["RU", "BR", "US"])

            else:  # HIGH_RISK_SURGE
                # Crypto exchange or luxury goods spike by unverified or new user
                high_risk_merchants = [m for m in self.merchants if m["category"] in ["crypto_exchange", "luxury_goods"]]
                merchant = random.choice(high_risk_merchants) if high_risk_merchants else random.choice(self.merchants)
                amount = random.uniform(1500.0, 5000.0)
                distance_km = round(random.uniform(100.0, 2000.0), 2)
                country = np.random.choice(COUNTRIES)

            tx = {
                "transaction_id": f"TX_{uuid.uuid4().hex[:12].upper()}",
                "user_id": user["user_id"],
                "merchant_id": merchant["merchant_id"],
                "amount": round(float(amount), 2),
                "currency": "USD",
                "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "channel": random.choice(["web", "api", "mobile_app"]),
                "card_type": random.choice(["virtual_card", "credit"]),
                "device_type": random.choice(["windows", "linux", "android"]),
                "ip_address": fake.ipv4_public(),
                "country": country,
                "distance_from_home_km": distance_km,
                "is_fraud": 1,
                "fraud_type": pattern
            }
            transactions.append(tx)

        df_tx = pd.DataFrame(transactions)
        df_tx = df_tx.sort_values("timestamp").reset_index(drop=True)
        df_users = pd.DataFrame(self.users)
        df_merchants = pd.DataFrame(self.merchants)

        return df_tx, df_users, df_merchants


if __name__ == "__main__":
    generator = TransactionGenerator(num_users=200, num_merchants=30)
    txs, users, merchants = generator.generate_batch(num_transactions=500, fraud_ratio=0.04)
    print(f"Generated {len(txs)} transactions across {len(users)} users and {len(merchants)} merchants.")
    print(f"Fraud distribution:\n{txs['fraud_type'].value_counts()}")

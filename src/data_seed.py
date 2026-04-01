from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DRIFT_DATA_PATH, TRAIN_DATA_PATH


def _log(message: str) -> None:
    print(f"[data_seed] {message}")


def _base_dataset(seed: int = 42, n_rows: int = 1000) -> pd.DataFrame:
    _log(f"Generating base dataset with seed={seed}, rows={n_rows}")
    rng = np.random.default_rng(seed)
    fraud_prob = 0.25

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n_rows + 1),
            "age": rng.integers(20, 60, size=n_rows),
            "transactions": rng.integers(1, 25, size=n_rows),
            "fraud": rng.binomial(1, fraud_prob, size=n_rows),
            "event_timestamp": pd.Timestamp.utcnow(),
        }
    )


def _drift_dataset(seed: int = 123, n_rows: int = 400) -> pd.DataFrame:
    _log(f"Generating drift dataset with seed={seed}, rows={n_rows}")
    rng = np.random.default_rng(seed)
    fraud_prob = 0.45

    return pd.DataFrame(
        {
            "customer_id": np.arange(10001, 10001 + n_rows),
            "age": rng.integers(48, 80, size=n_rows),
            "transactions": rng.integers(30, 220, size=n_rows),
            "fraud": rng.binomial(1, fraud_prob, size=n_rows),
            "event_timestamp": pd.Timestamp.utcnow(),
        }
    )


def seed_data() -> None:
    _log("Starting data seeding flow.")
    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df = _base_dataset()
    drift_df = _drift_dataset()
    _log(f"Writing training parquet to {TRAIN_DATA_PATH}")
    train_df.to_parquet(TRAIN_DATA_PATH, index=False)
    _log(f"Writing drift parquet to {DRIFT_DATA_PATH}")
    drift_df.to_parquet(DRIFT_DATA_PATH, index=False)
    _log(f"Seeded training data shape={train_df.shape}, columns={list(train_df.columns)}")
    _log(f"Seeded drift data shape={drift_df.shape}, columns={list(drift_df.columns)}")


if __name__ == "__main__":
    seed_data()

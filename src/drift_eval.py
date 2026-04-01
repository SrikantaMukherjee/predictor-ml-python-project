from __future__ import annotations

import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.metrics import accuracy_score

from src.config import DRIFT_DATA_PATH, MODEL_ALIAS, REGISTERED_MODEL_NAME, TRAIN_DATA_PATH
from src.mlflow_utils import configure_mlflow


def evaluate_drift() -> tuple[float, float]:
    print("[drift_eval] Starting drift evaluation.")
    if not TRAIN_DATA_PATH.exists() or not DRIFT_DATA_PATH.exists():
        raise FileNotFoundError("Seed data first. Missing train or drift parquet files.")

    print(f"[drift_eval] Reading base data: {TRAIN_DATA_PATH}")
    base_df = pd.read_parquet(TRAIN_DATA_PATH)
    print(f"[drift_eval] Reading drift data: {DRIFT_DATA_PATH}")
    drift_df = pd.read_parquet(DRIFT_DATA_PATH)
    print(f"[drift_eval] Base shape={base_df.shape}, Drift shape={drift_df.shape}")

    configure_mlflow()
    print(f"[drift_eval] Loading model models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")

    x_base = base_df[["age", "transactions"]]
    y_base = base_df["fraud"]
    x_drift = drift_df[["age", "transactions"]]
    y_drift = drift_df["fraud"]

    base_pred = model.predict(x_base)
    drift_pred = model.predict(x_drift)
    base_accuracy = accuracy_score(y_base, base_pred)
    drift_accuracy = accuracy_score(y_drift, drift_pred)
    drop = base_accuracy - drift_accuracy

    with mlflow.start_run(run_name="drift_evaluation"):
        mlflow.log_metric("base_accuracy", base_accuracy)
        mlflow.log_metric("drift_accuracy", drift_accuracy)
        mlflow.log_metric("accuracy_drop", drop)
        mlflow.log_param("scenario", "synthetic_feature_drift")
        print("[drift_eval] Logged drift metrics to MLflow run 'drift_evaluation'.")

    print(f"Base accuracy: {base_accuracy:.4f}")
    print(f"Drift accuracy: {drift_accuracy:.4f}")
    print(f"Accuracy drop: {drop:.4f}")

    return base_accuracy, drift_accuracy


if __name__ == "__main__":
    evaluate_drift()

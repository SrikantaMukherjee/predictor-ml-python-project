from __future__ import annotations

import mlflow

from src.config import EXPERIMENT_NAME, MLARTIFACTS_DIR, MLFLOW_TRACKING_URI


def configure_mlflow() -> None:
    print("[mlflow_utils] Configuring MLflow tracking and registry.")
    print(f"[mlflow_utils] Tracking URI: {MLFLOW_TRACKING_URI}")
    MLARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"[mlflow_utils] Creating experiment '{EXPERIMENT_NAME}' at {MLARTIFACTS_DIR.as_uri()}")
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=MLARTIFACTS_DIR.as_uri(),
        )
    else:
        print(f"[mlflow_utils] Using existing experiment '{EXPERIMENT_NAME}'")
    mlflow.set_experiment(EXPERIMENT_NAME)


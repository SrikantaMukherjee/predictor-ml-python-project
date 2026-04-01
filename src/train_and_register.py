from __future__ import annotations

from typing import Tuple

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.config import MODEL_ALIAS, MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME, TRAIN_DATA_PATH
from src.mlflow_utils import configure_mlflow


def train_model(random_state: int = 42) -> Tuple[str, float]:
    print("[train_and_register] Starting train_model()")
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing training data file: {TRAIN_DATA_PATH}")

    print(f"[train_and_register] Reading training dataset: {TRAIN_DATA_PATH}")
    df = pd.read_parquet(TRAIN_DATA_PATH)
    print(f"[train_and_register] Loaded shape={df.shape}, columns={list(df.columns)}")
    x = df[["age", "transactions"]]
    y = df["fraud"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"[train_and_register] Model trained. Validation accuracy={accuracy:.4f}")

    configure_mlflow()

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI, registry_uri=MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_name="rf_train_register") as run:
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("features", "age,transactions")
        mlflow.log_metric("accuracy", accuracy)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        print(f"[train_and_register] Logged model artifact URI: {model_info.model_uri}")

        # Ensure the newest model version is available under a stable alias
        # so inference and drift checks can load it deterministically.
        new_version = getattr(model_info, "version", None)
        if new_version is None:
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            new_version = str(max(versions, key=lambda v: int(v.version)).version)

        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=MODEL_ALIAS,
            version=str(new_version),
        )
        print(
            f"[train_and_register] Alias '{MODEL_ALIAS}' now points to "
            f"{REGISTERED_MODEL_NAME} v{new_version}"
        )

        print(f"MLflow run_id: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")

    return run.info.run_id, accuracy


if __name__ == "__main__":
    run_id, acc = train_model()
    print(f"Training accuracy: {acc:.4f}")
    print(f"Completed run: {run_id}")

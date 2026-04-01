from __future__ import annotations

import json
import time
from subprocess import Popen

import requests

import os

from src.config import MODEL_ALIAS, MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME


def _wait_for_server(url: str, timeout_sec: int = 30) -> None:
    print(f"[inference_test] Waiting for model server health at {url} (timeout={timeout_sec}s)")
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code in {200, 404}:
                return
        except requests.RequestException:
            time.sleep(1)
    raise TimeoutError("MLflow model server did not start in time.")


def run_inference_test(port: int = 5001) -> dict:
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    scoring_url = f"http://127.0.0.1:{port}/invocations"
    print(f"[inference_test] Starting inference test with model_uri={model_uri}")
    print(f"[inference_test] Scoring endpoint will be {scoring_url}")

    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["MLFLOW_REGISTRY_URI"] = MLFLOW_TRACKING_URI
    print(f"[inference_test] MLFLOW_TRACKING_URI set to {MLFLOW_TRACKING_URI}")

    command = [
        "mlflow",
        "models",
        "serve",
        "-m",
        model_uri,
        "-p",
        str(port),
        "--no-conda",
    ]
    print(f"[inference_test] Launch command: {' '.join(command)}")

    server = Popen(command)
    try:
        _wait_for_server(f"http://127.0.0.1:{port}")
        payload = {
            "dataframe_records": [
                {"age": 35, "transactions": 10},
                {"age": 67, "transactions": 120},
            ]
        }
        print(f"[inference_test] Request payload: {json.dumps(payload)}")
        response = requests.post(scoring_url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"[inference_test] HTTP status: {response.status_code}")
        print(f"[inference_test] Response body: {json.dumps(result)}")
        return result
    finally:
        print("[inference_test] Stopping temporary MLflow model server.")
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    run_inference_test()

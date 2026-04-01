from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feast_online_fetch import fetch_online_features


def main() -> None:
    endpoint = "http://127.0.0.1:5001/invocations"
    print("[feast_inference_request] Starting Feast -> MLflow inference request flow.")
    print(f"[feast_inference_request] Endpoint: {endpoint}")

    start = time.perf_counter()
    features = fetch_online_features(customer_id=1)
    print(f"[feast_inference_request] Feast output: {features}")
    payload = {
        "dataframe_records": [
            {
                "age": features["age"][0],
                "transactions": features["transactions"][0],
            }
        ]
    }
    print(f"[feast_inference_request] Request payload: {json.dumps(payload)}")
    print("[feast_inference_request] Invocation method: HTTP POST /invocations")

    try:
        response = requests.post(endpoint, json=payload, timeout=10)
        duration_ms = (time.perf_counter() - start) * 1000
        print(
            f"[feast_inference_request] Response status: {response.status_code} "
            f"(elapsed={duration_ms:.1f}ms)"
        )
        response.raise_for_status()
        print(f"[feast_inference_request] Response body: {response.text}")
    except requests.RequestException as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        print(
            f"[feast_inference_request] Invocation failed after {duration_ms:.1f}ms: {exc}. "
            "Make sure model serving is running with `mlflow models serve` on port 5001."
        )
        raise


if __name__ == "__main__":
    main()

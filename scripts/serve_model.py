"""Start MLflow model server using this repo's SQLite tracking/registry DB.

Sets MLFLOW_TRACKING_URI and MLFLOW_REGISTRY_URI so `mlflow models serve` does not
fall back to ./mlruns or a different database.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    db_path = PROJECT_ROOT / "mlflow.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_REGISTRY_URI"] = uri

    port = sys.argv[1] if len(sys.argv) > 1 else "5001"
    model_ref = sys.argv[2] if len(sys.argv) > 2 else "models:/FraudModel@champion"

    print(f"[serve_model] Project root: {PROJECT_ROOT}")
    print(f"[serve_model] MLFLOW_TRACKING_URI = {uri}")
    print(f"[serve_model] MLFLOW_REGISTRY_URI = {uri}")
    print(f"[serve_model] Model: {model_ref}  Port: {port}")

    if not db_path.exists():
        print(
            "[serve_model] No mlflow.db yet. Run: python scripts\\run_pipeline.py "
            "(or src.train_and_register) first."
        )

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "models",
        "serve",
        "-m",
        model_ref,
        "-p",
        port,
        "--no-conda",
    ]
    print(f"[serve_model] Command: {' '.join(cmd)}")
    os.chdir(PROJECT_ROOT)
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()

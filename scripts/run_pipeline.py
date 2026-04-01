from __future__ import annotations

import sys
from pathlib import Path

# Allow running `python scripts/run_pipeline.py` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_seed import seed_data  # noqa: E402
from src.drift_eval import evaluate_drift  # noqa: E402
from src.inference_test import run_inference_test  # noqa: E402
from src.train_and_register import train_model  # noqa: E402


def main() -> None:
    print("[run_pipeline] Starting end-to-end ML lifecycle pipeline.")
    print("[run_pipeline] Step 1/4: seed data")
    seed_data()
    print("[run_pipeline] Step 2/4: train and register model")
    train_model()
    print("[run_pipeline] Step 3/4: inference smoke test")
    run_inference_test()
    print("[run_pipeline] Step 4/4: drift evaluation")
    evaluate_drift()
    print("[run_pipeline] Pipeline completed successfully.")


if __name__ == "__main__":
    main()

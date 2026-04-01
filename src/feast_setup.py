from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from src.config import FEAST_REPO_DIR
from src.data_seed import seed_data


def setup_and_materialize_feast() -> None:
    print("[feast_setup] Starting Feast setup/materialization flow.")
    seed_data()
    print(f"[feast_setup] Running `feast apply` in {FEAST_REPO_DIR}")
    subprocess.run(["feast", "apply"], cwd=str(FEAST_REPO_DIR), check=True)
    end_date_iso = datetime.now(timezone.utc).isoformat()
    print(f"[feast_setup] Running incremental materialization up to {end_date_iso}")
    subprocess.run(
        ["feast", "materialize-incremental", end_date_iso],
        cwd=str(FEAST_REPO_DIR),
        check=True,
    )

    print(f"[feast_setup] Feast apply and materialize completed for repo: {FEAST_REPO_DIR}")


if __name__ == "__main__":
    setup_and_materialize_feast()

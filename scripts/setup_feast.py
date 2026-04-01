from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feast_setup import setup_and_materialize_feast  # noqa: E402


if __name__ == "__main__":
    print("[setup_feast_script] Invoking Feast setup/materialization script.")
    setup_and_materialize_feast()

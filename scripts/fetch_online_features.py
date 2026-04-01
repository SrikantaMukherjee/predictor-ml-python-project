from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feast_online_fetch import fetch_online_features  # noqa: E402


if __name__ == "__main__":
    print("[fetch_online_features_script] Invoking Feast online feature fetch.")
    print(fetch_online_features(1))

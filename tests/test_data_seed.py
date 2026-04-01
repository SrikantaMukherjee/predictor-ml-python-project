import pandas as pd

from src.config import DRIFT_DATA_PATH, TRAIN_DATA_PATH
from src.data_seed import seed_data


def test_seed_data_creates_parquet_files():
    seed_data()
    assert TRAIN_DATA_PATH.exists()
    assert DRIFT_DATA_PATH.exists()

    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    drift_df = pd.read_parquet(DRIFT_DATA_PATH)

    assert {"age", "transactions", "fraud", "event_timestamp"}.issubset(train_df.columns)
    assert {"age", "transactions", "fraud", "event_timestamp"}.issubset(drift_df.columns)

import pandas as pd

from src.config import DRIFT_DATA_PATH, TRAIN_DATA_PATH
from src.data_seed import seed_data


def test_seed_data_creates_parquet_files():
    seed_data()
    assert TRAIN_DATA_PATH.exists()
    assert DRIFT_DATA_PATH.exists()

    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    drift_df = pd.read_parquet(DRIFT_DATA_PATH)

    assert {"age", "transactions", "fraud", "event_timestamp"}.issubset(train_df.columns)
    assert {"age", "transactions", "fraud", "event_timestamp"}.issubset(drift_df.columns)

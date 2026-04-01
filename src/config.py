from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
FEAST_REPO_DIR = PROJECT_ROOT / "feature_repo"

TRAIN_DATA_PATH = DATA_DIR / "train_data.parquet"
DRIFT_DATA_PATH = DATA_DIR / "drift_data.parquet"

EXPERIMENT_NAME = "fraud_lifecycle"
REGISTERED_MODEL_NAME = "FraudModel"
MODEL_ALIAS = "champion"

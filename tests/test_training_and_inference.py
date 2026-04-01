import mlflow.pyfunc
import pandas as pd

from src.config import MODEL_ALIAS, REGISTERED_MODEL_NAME, TRAIN_DATA_PATH
from src.data_seed import seed_data
from src.inference_test import run_inference_test
from src.mlflow_utils import configure_mlflow
from src.train_and_register import train_model


def test_train_register_and_inference_smoke():
    seed_data()
    train_model()
    configure_mlflow()

    # Load champion alias via pyfunc directly (no server) for a quick sanity check.
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")

    df = pd.read_parquet(TRAIN_DATA_PATH)
    X = df[["age", "transactions"]].head(5)
    preds = model.predict(X)
    assert len(preds) == 5

    # Validate HTTP inference path
    result = run_inference_test(port=5001)
    assert "predictions" in result
    assert len(result["predictions"]) >= 1


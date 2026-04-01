# Predictor ML Lifecycle (Synthetic Data + MLflow + Inference + Drift)

This project provides an end-to-end, runnable example that:
- seeds synthetic training + drift data
- creates a working Feast feature store on top of the seeded data
- trains a simple `RandomForestClassifier`
- logs and registers the model in **MLflow**
- serves the registered model via `mlflow models serve`
- tests inference through the HTTP inference layer
- evaluates drift by comparing accuracy on base vs drifted data

## Quick Start (Local)

```powershell
cd d:\predictor-ml-python-project
python scripts\setup_feast.py
python scripts\fetch_online_features.py
python scripts\run_pipeline.py
# optional: after model server is running on 5001
python scripts\feast_inference_request.py
pytest -q
```

## Visualize in MLflow UI

```powershell
cd d:\predictor-ml-python-project
mlflow ui --backend-store-uri "sqlite:///d:/predictor-ml-python-project/mlflow.db" --port 5000
```

Then open:
- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Main Entry Points

- `src/data_seed.py` : creates `data/train_data.parquet` and `data/drift_data.parquet`
- `feature_repo/` : Feast repository (`feature_store.yaml` + feature definitions)
- `src/feast_setup.py` : runs `feast apply` and `feast materialize-incremental`
- `src/feast_online_fetch.py` : fetches online features for inference
- `scripts/feast_inference_request.py` : sends Feast-fetched features to MLflow `/invocations`
- `src/train_and_register.py` : trains and registers to MLflow, sets model alias `champion`
- `src/inference_test.py` : starts `mlflow models serve` and calls `/invocations` over HTTP
- `src/drift_eval.py` : loads `models:/FraudModel@champion` and evaluates drift impact

## Notes

- MLflow tracking + registry are configured to use local SQLite: `sqlite:///.../mlflow.db`.
- Model artifacts are stored under `mlartifacts/`.
- The model expects exactly two numeric features: `age`, `transactions`.


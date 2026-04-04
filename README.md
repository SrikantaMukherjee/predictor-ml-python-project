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
MLFLow visualize in local - http://localhost:5000/


## Serve the model (local SQLite registry)

Use a **registry URI** (`models:/...`). Without `models:/`, MLflow treats the string as a **local folder** and fails.

**Recommended (sets SQLite URIs correctly from any directory):**

```powershell
cd d:\predictor-ml-python-project
python scripts\serve_model.py
```

Optional: `python scripts\serve_model.py 5001 "models:/FraudModel/1"`

**Manual PowerShell:**

```powershell
cd d:\predictor-ml-python-project
$env:MLFLOW_TRACKING_URI = "sqlite:///d:/predictor-ml-python-project/mlflow.db"
$env:MLFLOW_REGISTRY_URI = "sqlite:///d:/predictor-ml-python-project/mlflow.db"
mlflow models serve -m "models:/FraudModel@champion" -p 5001 --no-conda
```

Run `python scripts\run_pipeline.py` (or train once) first so the `champion` alias exists.

**Scoring:** POST JSON to [http://127.0.0.1:5001/invocations](http://127.0.0.1:5001/invocations). The model expects `age` and `transactions` (see `scripts/feast_inference_request.py`).

cURL you can paste into Postman (**Import → Raw text**):

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records":[{"age":35,"transactions":10},{"age":67,"transactions":120}]}'
```

One-liner from PowerShell (use `curl.exe` so it is not the `Invoke-WebRequest` alias):

```powershell
curl.exe -X POST http://127.0.0.1:5001/invocations -H "Content-Type: application/json" -d '{"dataframe_records":[{"age":35,"transactions":10}]}'
```

### If serve fails with Alembic (`Can't locate revision` / `1b5f0d9ad7c1`)

Your `mlflow.db` was created or migrated with a **different MLflow version** than the one you run now. The fix is to use a matching version and a fresh DB (you will lose old experiment rows in that file).

1. Align the package: `python -m pip install -r requirements.txt` (MLflow is pinned there).
2. From the project root, back up and remove the DB, then recreate runs:

   ```powershell
   cd d:\predictor-ml-python-project
   copy mlflow.db mlflow.db.bak
   del mlflow.db
   python scripts\run_pipeline.py
   ```

3. Start the server again with `python scripts\serve_model.py`.

Always use **`--no-conda`** (or this project’s `serve_model.py`, which passes it) so MLflow does not try to build a virtualenv via **pyenv** on Windows.

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


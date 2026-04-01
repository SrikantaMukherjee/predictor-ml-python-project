from src.data_seed import seed_data
from src.drift_eval import evaluate_drift


def test_drift_evaluation_runs():
    seed_data()
    base_acc, drift_acc = evaluate_drift()
    assert 0.0 <= base_acc <= 1.0
    assert 0.0 <= drift_acc <= 1.0


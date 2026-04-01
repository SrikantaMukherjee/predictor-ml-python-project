from __future__ import annotations

from feast import FeatureStore

from src.config import FEAST_REPO_DIR


def fetch_online_features(customer_id: int = 1) -> dict:
    print(f"[feast_online_fetch] Fetching online features for customer_id={customer_id}")
    store = FeatureStore(repo_path=str(FEAST_REPO_DIR))
    result = store.get_online_features(
        features=[
            "customer_features:age",
            "customer_features:transactions",
        ],
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()
    print(f"[feast_online_fetch] Raw feature response: {result}")
    return result


if __name__ == "__main__":
    print(fetch_online_features(1))

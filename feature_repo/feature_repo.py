from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64
from feast.value_type import ValueType

customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    value_type=ValueType.INT64,
)

customer_source = FileSource(
    path="../data/train_data.parquet",
    timestamp_field="event_timestamp",
)

customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=7),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="transactions", dtype=Int64),
    ],
    source=customer_source,
    online=True,
)

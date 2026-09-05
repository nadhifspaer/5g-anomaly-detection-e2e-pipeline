# Kafka producer: replays the held-out test split as a simulated live KPI feed.

import argparse
import json
import sys
import time
from pathlib import Path

# `python streaming/producer.py` sets sys.path[0] to this script's own directory, not the cwd, so the sibling src/ package needs this
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from kafka import KafkaProducer

from src import feature_engineering as fe

DATA_PATH = "data/5g_kpi_dataset.csv"
TOPIC = "kpi-stream"
PAYLOAD_COLS = ["timestamp", "cell_id", "cell_type", "slice_type"] + fe.KPI_COLS


def load_test_rows():
    # reuses the same feature-engineering split functions as training, never a redefinition
    df = fe.load_raw(DATA_PATH)
    X, meta = fe.build_feature_matrix(df.copy())
    _, _, _, meta_test = fe.temporal_train_test_split(X, meta)
    # identity match back to raw columns, meta_test carries no raw KPI values
    test_raw = meta_test[["timestamp", "cell_id", "slice_type"]].merge(
        df, on=["timestamp", "cell_id", "slice_type"], how="left"
    )
    # global timestamp order also preserves per-(cell_id, slice_type) order as a consequence
    return test_raw.sort_values("timestamp").reset_index(drop=True)


def run(rate_per_sec: float, bootstrap_servers: str, topic: str) -> None:
    """Publish the held-out test split to `topic` at `rate_per_sec` rows/sec, in global timestamp order."""
    rows = load_test_rows()
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )
    delay = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0

    for _, row in rows.iterrows():
        payload = {col: row[col] for col in PAYLOAD_COLS}
        payload["timestamp"] = str(payload["timestamp"])
        # partition key = (cell_id, slice_type): Kafka's per-partition ordering guarantee then holds per group
        key = f"{payload['cell_id']}|{payload['slice_type']}"
        producer.send(topic, key=key, value=payload)
        if delay:
            time.sleep(delay)

    producer.flush()
    producer.close()
    print(f"published {len(rows)} rows to topic {topic!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay the held-out test split to Kafka.")
    parser.add_argument("--rate", type=float, default=10.0, help="rows/sec to publish")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default=TOPIC)
    args = parser.parse_args()
    run(args.rate, args.bootstrap_servers, args.topic)

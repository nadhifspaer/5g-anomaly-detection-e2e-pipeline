# Kafka consumer: scores each streamed KPI row via /score, writes results to SQLite.

import argparse
import json
import sqlite3

import requests
from kafka import KafkaConsumer

TOPIC = "kpi-stream"
DB_PATH = "streaming/scored_results.db"
API_URL = "http://127.0.0.1:8000/score"
# fixed group_id: restarts resume from the last committed offset instead of re-reading from earliest
GROUP_ID = "kpi-scoring-consumer"


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scored_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cell_id TEXT NOT NULL,
            slice_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            anomaly_score REAL NOT NULL,
            alert INTEGER NOT NULL,
            top_kpis TEXT NOT NULL,
            scored_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    return conn


def run(bootstrap_servers: str, topic: str, api_url: str, db_path: str, group_id: str = GROUP_ID) -> None:
    """Consume `topic`, call `api_url` per row, write cell_id/timestamp/anomaly score/alert/top KPIs to SQLite."""
    conn = init_db(db_path)
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    for message in consumer:
        row = message.value
        try:
            response = requests.post(api_url, json=row, timeout=10.0)
            response.raise_for_status()
        except requests.RequestException as exc:
            # insufficient backward history (422) or a transient API error: skip, don't crash the stream
            print(f"score failed for {row.get('cell_id')}/{row.get('slice_type')} at {row.get('timestamp')}: {exc}")
            continue

        result = response.json()
        conn.execute(
            "INSERT INTO scored_results (cell_id, slice_type, timestamp, anomaly_score, alert, top_kpis) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                result["cell_id"],
                result["slice_type"],
                result["timestamp"],
                result["anomaly_score"],
                int(result["alert"]),
                json.dumps(result["root_cause"]),
            ),
        )
        conn.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score streamed KPI rows and persist results to SQLite.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default=TOPIC)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--group-id", default=GROUP_ID)
    args = parser.parse_args()
    run(args.bootstrap_servers, args.topic, args.api_url, args.db_path, args.group_id)

# Single stable-interface scoring function for api/main.py and the dashboard.
# See architecture.md Section 5 Step 4, Section 9: no routing logic, one path.

import pandas as pd

from src import evaluate as ev
from src import feature_engineering as fe
from src import root_cause as rc

# Fixed absolute score cutoff approximating the K=2% production choice
# (src/evaluate.py's select_production_k; see reports/model_evaluation.md,
# "Capacity-and-Severity K Selection"): the 98th percentile of train-side
# anomaly scores (architecture.md Section 2/Stage 0.1: temporal split, no
# leakage). This is an approximation, not identical to K=2%: the report's
# capacity-and-severity table ranks each row against the REST OF THAT SAME
# DAY's rows, which live single-row scoring cannot do before the day ends.
# A fixed cutoff applied to the temporal test split gives a close but not
# identical operating point (24 alerts, precision 0.333, recall 0.148 vs.
# the per-day-ranked table's 30.4 alerts/day, precision 0.333, recall 0.130).
# Re-derive if the model, contamination, or picked production K changes.
ANOMALY_SCORE_ALERT_THRESHOLD = 0.026575134021403518


def score_row(
    raw_row: dict,
    history: pd.DataFrame,
    model,
    alert_threshold: float = ANOMALY_SCORE_ALERT_THRESHOLD,
    top_n: int = 5,
) -> dict:
    """Raw KPI row -> Stage 2 feature engineering -> Stage 3 Isolation Forest -> Stage 4.1 root cause; single stable interface, no routing logic."""
    new_row = pd.DataFrame([raw_row])
    new_row["timestamp"] = pd.to_datetime(new_row["timestamp"])
    # empty history (a genuinely new cell/slice) needs no concat, avoids an empty-frame dtype warning
    combined = new_row if history.empty else pd.concat([history, new_row], ignore_index=True)

    featured = fe.add_time_features(fe.add_rolling_zscore_features(combined))

    # identity match (timestamp, cell_id, slice_type), never positional order or array
    # index: add_rolling_zscore_features sorts by (cell_id, slice_type, timestamp) and
    # resets the index, so the new row's position in `featured` is unrelated to its
    # position in `combined`. Same pattern verified safe in notebooks/02's Stage 4 example.
    identity = (
        (featured["timestamp"] == new_row["timestamp"].iloc[0])
        & (featured["cell_id"] == raw_row["cell_id"])
        & (featured["slice_type"] == raw_row["slice_type"])
    )
    matched = featured.index[identity]
    if len(matched) != 1:
        raise ValueError(
            f"expected exactly one feature row for (timestamp={raw_row['timestamp']}, "
            f"cell_id={raw_row['cell_id']}, slice_type={raw_row['slice_type']}), found {len(matched)}"
        )
    row_id = matched[0]

    if featured.loc[row_id, fe.ZSCORE_COLS].isna().any():
        raise ValueError(
            f"insufficient backward history for ({raw_row['cell_id']}, {raw_row['slice_type']}): "
            f"need at least {fe.ROLL_MIN_PERIODS} prior rows in `history`"
        )

    # identity-matched row_id passed through unchanged, same row `rank_deviation_contributions` scores below
    X_row = featured.loc[[row_id], fe.FEATURE_COLS]
    score = float(ev.anomaly_score(model, X_row)[0])
    root_cause = rc.rank_deviation_contributions(featured, row_id, top_n=top_n)

    return {
        "timestamp": raw_row["timestamp"],
        "cell_id": raw_row["cell_id"],
        "slice_type": raw_row["slice_type"],
        "anomaly_score": score,
        "alert": bool(score >= alert_threshold),
        "alert_threshold": alert_threshold,
        "root_cause": root_cause.to_dict(orient="records"),
    }

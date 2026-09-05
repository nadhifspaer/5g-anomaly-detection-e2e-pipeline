# Single stable-interface scoring function for api/main.py and the dashboard.

import pandas as pd

from src import evaluate as ev
from src import feature_engineering as fe
from src import root_cause as rc

# Fixed score cutoff approximating the production K
ANOMALY_SCORE_ALERT_THRESHOLD = 0.026575134021403518


def score_row(
    raw_row: dict,
    history: pd.DataFrame,
    model,
    alert_threshold: float = ANOMALY_SCORE_ALERT_THRESHOLD,
    top_n: int = 5,
) -> dict:
    """Raw KPI row -> feature engineering -> Isolation Forest -> root cause; single stable interface, no routing logic."""
    new_row = pd.DataFrame([raw_row])
    new_row["timestamp"] = pd.to_datetime(new_row["timestamp"])
    # empty history needs no concat, avoids an empty-frame dtype warning
    combined = new_row if history.empty else pd.concat([history, new_row], ignore_index=True)

    featured = fe.add_time_features(fe.add_rolling_zscore_features(combined))

    # identity match (timestamp, cell_id, slice_type), never positional: add_rolling_zscore_features re-sorts and resets the index
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

# Backward-window (cell_id, slice_type) rolling z-score features, temporal split.

from typing import Tuple

import pandas as pd

KPI_COLS = [
    "throughput_mbps",
    "latency_ms",
    "packet_loss_pct",
    "handover_count",
    "rsrp_dbm",
    "rsrq_db",
    "prb_utilization_pct",
    "active_users",
]
GROUP_KEYS = ["cell_id", "slice_type"]

# window=10, min_periods=3 
ROLL_WINDOW = 10
ROLL_MIN_PERIODS = 3

ZSCORE_COLS = [f"{c}_zscore" for c in KPI_COLS]
TIME_COLS = ["hour_of_day", "day_of_week"]
FEATURE_COLS = ZSCORE_COLS + TIME_COLS
META_COLS = ["timestamp", "cell_id", "slice_type", "cell_type", "sla_compliant"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df


def _backward_rolling_mean_std(df: pd.DataFrame, col: str) -> Tuple[pd.Series, pd.Series]:
    # shift(1) excludes the current row from its own baseline window
    shifted = df.groupby(GROUP_KEYS, observed=True)[col].shift(1)
    grouped_shifted = shifted.groupby([df["cell_id"], df["slice_type"]], observed=True)
    mean = grouped_shifted.rolling(window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).mean()
    std = grouped_shifted.rolling(window=ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).std()
    mean.index = mean.index.droplevel(GROUP_KEYS)
    std.index = std.index.droplevel(GROUP_KEYS)
    return mean.reindex(df.index), std.reindex(df.index)


def add_rolling_zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    # backward-window (cell_id, slice_type) z-score
    df = df.sort_values(GROUP_KEYS + ["timestamp"]).reset_index(drop=True)
    for col in KPI_COLS:
        mean, std = _backward_rolling_mean_std(df, col)
        z = (df[col] - mean) / std
        # std=0 within window -> no deviation, z-score set to 0
        z = z.where(std != 0, 0.0)
        df[f"{col}_zscore"] = z
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # calendar features derived from timestamp, not copied from the second Kaggle file
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # X: model-ready numeric features only, no raw KPI values, no categoricals, no label
    df = add_rolling_zscore_features(df)
    df = add_time_features(df)
    # drop rows with insufficient backward history (< min_periods)
    df = df.dropna(subset=ZSCORE_COLS).reset_index(drop=True)
    X = df[FEATURE_COLS].copy()
    # sla_compliant held out for evaluation only, never a feature
    meta = df[META_COLS].copy()
    return X, meta


def temporal_train_test_split(
    X: pd.DataFrame, meta: pd.DataFrame, train_frac: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # cutoff by timestamp range, never random_state/shuffle
    ts = meta["timestamp"]
    cutoff = ts.min() + train_frac * (ts.max() - ts.min())
    train_mask = ts <= cutoff
    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[~train_mask].reset_index(drop=True)
    meta_train = meta[train_mask].reset_index(drop=True)
    meta_test = meta[~train_mask].reset_index(drop=True)
    return X_train, X_test, meta_train, meta_test

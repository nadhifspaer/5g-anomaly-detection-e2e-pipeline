# Path A evaluation, capacity-and-severity K selection. See architecture.md
# Section 7/8: Path A resolved at Stage 0.2, not re-litigated; no Path B code
# path here. sla_compliant is read here only as a label, never as a feature.

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_curve

from src import feature_engineering as fe
from src import model as m

# swept alert-review capacity, percent of each day's scored volume (architecture.md Section 7)
K_PCT_CANDIDATES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15]

# below this sla_compliant==0 rate, Precision@K/PR-AUC estimates are judged too noisy for Path A
MIN_LABEL_RATE = 0.01
# accuracy edge over base rate above which a single KPI column is judged a label restatement
TRIVIAL_SINGLE_COLUMN_EDGE_PP = 5.0

# confirmed ~60 rows/hour x 24h, uniform across the file (architecture.md Section 2 / notebooks/01_eda.ipynb)
DAILY_ROW_VOLUME = 1_440

# review-capacity assumption (architecture.md Section 7), not a dollar figure:
# 10 min/alert, 2h/shift dedicated to this system's alerts, 3 shifts covering 24h
# -> (2*60)/10 = 12 alerts/analyst-shift, x3 shifts = 36 alerts/day capacity
# -> 36 / 1440 daily rows = 0.025 (2.5%), the capacity cap expressed as K
ANALYST_MINUTES_PER_ALERT = 10
SHIFT_HOURS_FOR_ALERTS = 2
SHIFTS_PER_DAY = 3
DAILY_ALERT_CAPACITY = (SHIFT_HOURS_FOR_ALERTS * 60 // ANALYST_MINUTES_PER_ALERT) * SHIFTS_PER_DAY
CAPACITY_K_PCT = DAILY_ALERT_CAPACITY / DAILY_ROW_VOLUME


def _best_single_column_accuracy(values: np.ndarray, is_positive: np.ndarray) -> float:
    # audit-only: best-possible single-threshold accuracy, both directions, O(n log n).
    # Not a model evaluation metric, see select_evaluation_path.
    order = np.argsort(values)
    sorted_labels = is_positive[order]
    n = len(sorted_labels)
    total_pos = sorted_labels.sum()
    total_neg = n - total_pos
    cum_pos = np.cumsum(sorted_labels)
    cum_neg = np.arange(1, n + 1) - cum_pos
    correct_high_positive = (total_pos - cum_pos) + cum_neg
    correct_low_positive = cum_pos + (total_neg - cum_neg)
    return float(max(correct_high_positive.max(), correct_low_positive.max()) / n)


def select_evaluation_path(
    meta_train: pd.DataFrame,
    df_full_raw: pd.DataFrame,
    kpi_cols=fe.KPI_COLS,
    label_col: str = "sla_compliant",
) -> dict:
    """Confirms Evaluation Path A (architecture.md Section 7/8, resolved at
    Stage 0.2, not re-litigated), reproducing the Stage 0.1 label audit
    (architecture.md Section 2) on train-side data for transparency in
    reports/model_evaluation.md. This project implements Path A only; there
    is no Path B code path to dispatch to.

    meta_train is Stage 2's actual train split (src/feature_engineering.py's
    temporal_train_test_split output): rate0 is computed by calling
    src/model.py's derive_contamination_path_a(meta_train) directly, the same
    call Stage 3.1 uses for `contamination`, so both report one shared number.
    df_full_raw supplies the raw KPI columns for the single-column check
    (meta_train carries no raw KPI values), row-matched to meta_train's exact
    (timestamp, cell_id, slice_type) keys, not an independent timestamp cutoff,
    so it does not silently reintroduce the min_periods rows Stage 2 dropped.

    Returns rate0, base_rate, max_single_column_edge_pp, single_column_edges_pp,
    degenerate_rate, trivial_single_column (the last two per the thresholds
    documented in MIN_LABEL_RATE / TRIVIAL_SINGLE_COLUMN_EDGE_PP), reported
    for transparency only; they do not change which evaluation runs.
    """
    rate0 = m.derive_contamination_path_a(meta_train)
    base_rate = max(rate0, 1 - rate0)

    # row-match df_full_raw to meta_train's exact population, not a timestamp cutoff
    raw = meta_train[["timestamp", "cell_id", "slice_type"]].merge(
        df_full_raw, on=["timestamp", "cell_id", "slice_type"], how="left"
    )
    label = raw[label_col]
    edges = {
        col: (_best_single_column_accuracy(raw[col].to_numpy(), (label == 0).to_numpy()) - base_rate) * 100
        for col in kpi_cols
    }
    max_edge = max(edges.values())
    return {
        "rate0": rate0,
        "base_rate": base_rate,
        "max_single_column_edge_pp": max_edge,
        "single_column_edges_pp": edges,
        "degenerate_rate": rate0 < MIN_LABEL_RATE or rate0 > 1 - MIN_LABEL_RATE,
        "trivial_single_column": max_edge > TRIVIAL_SINGLE_COLUMN_EDGE_PP,
    }


def anomaly_score(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    # higher = more anomalous; decision_function is higher = more normal
    return -model.decision_function(X)


def _alert_mask(scores: np.ndarray, dates: np.ndarray, k_pct: float) -> np.ndarray:
    # top k_pct of EACH day's scored rows by score, pooled across days
    frame = pd.DataFrame({"score": scores, "date": dates})
    alerted = np.zeros(len(frame), dtype=bool)
    for _, idx in frame.groupby("date").groups.items():
        day_scores = frame.loc[idx, "score"]
        n_alerts = max(1, int(np.ceil(len(day_scores) * k_pct)))
        alerted[day_scores.nlargest(n_alerts).index] = True
    return alerted


def precision_recall_at_k(
    y_true: np.ndarray, scores: np.ndarray, dates: np.ndarray, k_pct: float
) -> Tuple[float, float, int]:
    alerted = _alert_mask(scores, dates, k_pct)
    tp = int((alerted & (y_true == 1)).sum())
    n_alerted = int(alerted.sum())
    n_pos = int((y_true == 1).sum())
    precision = tp / n_alerted if n_alerted else float("nan")
    recall = tp / n_pos if n_pos else float("nan")
    return precision, recall, n_alerted


def evaluate_path_a(
    model: IsolationForest,
    X_test: pd.DataFrame,
    meta_test: pd.DataFrame,
    k_pcts=K_PCT_CANDIDATES,
    target_recall: float = 0.9,
) -> dict:
    """Path A metrics (architecture.md Section 7): PR-AUC, Precision@K/Recall@K
    swept over k_pcts (percent of each day's scored volume, never a fixed
    absolute K), False Positive Rate at target_recall (0.9 default, an
    adjustable operational choice, not derived from the data). sla_compliant
    is read from meta_test only, never passed to the model.
    """
    y_true = (meta_test["sla_compliant"] == 0).astype(int).to_numpy()
    scores = anomaly_score(model, X_test)
    dates = meta_test["timestamp"].dt.date.to_numpy()

    pr_auc = average_precision_score(y_true, scores)

    k_table = pd.DataFrame(
        [
            dict(zip(("k_pct", "precision", "recall", "n_alerts"), (k, *precision_recall_at_k(y_true, scores, dates, k))))
            for k in k_pcts
        ]
    )

    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, scores)
    hit = int(np.argmax(tpr_arr >= target_recall))
    reached = bool(tpr_arr[hit] >= target_recall)

    return {
        "pr_auc": float(pr_auc),
        "precision_recall_at_k": k_table,
        "target_recall": target_recall,
        "fpr_at_target_recall": float(fpr_arr[hit]) if reached else float("nan"),
        "score_threshold_at_target_recall": float(thresh_arr[hit]) if reached else float("nan"),
    }


def capacity_and_severity_table(
    model: IsolationForest,
    X_test: pd.DataFrame,
    meta_test: pd.DataFrame,
    k_pcts=K_PCT_CANDIDATES,
) -> pd.DataFrame:
    """Capacity-and-severity K-selection table (architecture.md Section 7),
    replacing the dollar-cost alert-budget metric: this dataset carries no
    financial data to ground a cost figure, and the dollar-cost curve never
    converged to an interior minimum (always favored higher K).

    Per swept K: alerts/day (n_alerts scaled from the test window's actual
    row count to DAILY_ROW_VOLUME, since the test split covers a partial
    day, not a full one), precision, recall, and the severity of missed
    anomalies, mean and max of the mean absolute z-score across
    src/feature_engineering.py's ZSCORE_COLS for every sla_compliant == 0
    row NOT alerted at that K. Severity uses the KPI z-score columns only,
    not hour_of_day/day_of_week (calendar context, not a deviation measure),
    even though both are in the feature set the model was trained on.

    within_capacity flags K values at or under CAPACITY_K_PCT, the stated
    review-capacity assumption (see DAILY_ALERT_CAPACITY above), not a
    dollar figure.
    """
    y_true = (meta_test["sla_compliant"] == 0).astype(int).to_numpy()
    scores = anomaly_score(model, X_test)
    dates = meta_test["timestamp"].dt.date.to_numpy()
    severity = X_test[fe.ZSCORE_COLS].abs().mean(axis=1).to_numpy()
    day_scale = DAILY_ROW_VOLUME / len(X_test)

    rows = []
    for k in k_pcts:
        alerted = _alert_mask(scores, dates, k)
        precision, recall, n_alerts = precision_recall_at_k(y_true, scores, dates, k)
        fn_severity = severity[(y_true == 1) & ~alerted]
        rows.append(
            {
                "k_pct": k,
                "alerts_per_day": n_alerts * day_scale,
                "precision": precision,
                "recall": recall,
                "n_false_negatives": int(len(fn_severity)),
                "mean_severity_missed": float(fn_severity.mean()) if len(fn_severity) else float("nan"),
                "max_severity_missed": float(fn_severity.max()) if len(fn_severity) else float("nan"),
                "within_capacity": bool(k <= CAPACITY_K_PCT),
            }
        )
    return pd.DataFrame(rows)


def select_production_k(table: pd.DataFrame) -> dict:
    """Production K (architecture.md Section 7): among swept candidates at or
    under CAPACITY_K_PCT (the stated review-capacity assumption, not a
    dollar-cost minimum), the one with the highest recall. Recall is
    non-decreasing in K under top-K alerting, so this is equivalently the
    largest tested K that still fits the capacity cap.
    """
    under_cap = table[table["within_capacity"]]
    if under_cap.empty:
        raise ValueError("no swept K stays within CAPACITY_K_PCT; widen K_PCT_CANDIDATES downward")
    return under_cap.loc[under_cap["recall"].idxmax()].to_dict()


# ---------------- runtime entry point ----------------


def run_evaluation(
    df_full_raw: pd.DataFrame,
    meta_train: pd.DataFrame,
    X_test: pd.DataFrame,
    meta_test: pd.DataFrame,
    models_by_contamination: Dict[float, IsolationForest],
    k_pcts=K_PCT_CANDIDATES,
    target_recall: float = 0.9,
) -> dict:
    """Runs Path A evaluation (architecture.md Section 7/8, the only
    evaluation path this project implements). select_evaluation_path's
    diagnostics are attached for transparency, reproducing the Stage 0.1
    label audit on train-side data at runtime; they report on the
    resolved decision, they do not select or change which evaluation runs.
    """
    diagnostics = select_evaluation_path(meta_train, df_full_raw)

    contamination = next(iter(models_by_contamination))
    model = models_by_contamination[contamination]
    result = evaluate_path_a(model, X_test, meta_test, k_pcts=k_pcts, target_recall=target_recall)
    result["capacity_severity_table"] = capacity_and_severity_table(model, X_test, meta_test, k_pcts=k_pcts)
    result["production_k"] = select_production_k(result["capacity_severity_table"])

    result["path"] = "A"
    result["path_selection_diagnostics"] = diagnostics
    return result

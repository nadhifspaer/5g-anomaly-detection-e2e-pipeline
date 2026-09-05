# Evaluation metrics and capacity-and-severity K selection for the Isolation Forest model.

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_curve

from src import feature_engineering as fe
from src import model as m

# swept alert-review capacity, percent of each day's scored volume
K_PCT_CANDIDATES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15]

# below this sla_compliant==0 rate, Precision@K/PR-AUC estimates are judged too noisy for Path A
MIN_LABEL_RATE = 0.01
# accuracy edge over base rate above which a single KPI column is judged a label restatement
TRIVIAL_SINGLE_COLUMN_EDGE_PP = 5.0

# ~60 rows/hour x 24h, uniform across the file (see notebooks/01_eda.ipynb)
DAILY_ROW_VOLUME = 1_440

# review-capacity assumption, not a dollar figure; see reports/model_evaluation.md, Capacity-and-Severity K Selection
ANALYST_MINUTES_PER_ALERT = 10
SHIFT_HOURS_FOR_ALERTS = 2
SHIFTS_PER_DAY = 3
DAILY_ALERT_CAPACITY = (SHIFT_HOURS_FOR_ALERTS * 60 // ANALYST_MINUTES_PER_ALERT) * SHIFTS_PER_DAY
CAPACITY_K_PCT = DAILY_ALERT_CAPACITY / DAILY_ROW_VOLUME


def _best_single_column_accuracy(values: np.ndarray, is_positive: np.ndarray) -> float:
    # audit-only best-possible single-threshold accuracy, both directions; not a model evaluation metric
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
    """Reproduces the sla_compliant label audit on train-side data; see reports/model_evaluation.md, Evaluation Label."""
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
    # higher = more anomalous. decision_function is higher = more normal
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
    """PR-AUC, Precision@K/Recall@K swept over k_pcts, and FPR at target_recall against sla_compliant."""
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
    """Alerts/day, precision/recall, and missed-anomaly severity per swept K; see reports/model_evaluation.md, Capacity-and-Severity K Selection."""
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
    """Picks the highest-recall swept K at or under CAPACITY_K_PCT."""
    under_cap = table[table["within_capacity"]]
    if under_cap.empty:
        raise ValueError("no swept K stays within CAPACITY_K_PCT; widen K_PCT_CANDIDATES downward")
    return under_cap.loc[under_cap["recall"].idxmax()].to_dict()


# runtime entry point


def run_evaluation(
    df_full_raw: pd.DataFrame,
    meta_train: pd.DataFrame,
    X_test: pd.DataFrame,
    meta_test: pd.DataFrame,
    models_by_contamination: Dict[float, IsolationForest],
    k_pcts=K_PCT_CANDIDATES,
    target_recall: float = 0.9,
) -> dict:
    """Runs evaluation and attaches label-selection diagnostics for reporting."""
    diagnostics = select_evaluation_path(meta_train, df_full_raw)

    contamination = next(iter(models_by_contamination))
    model = models_by_contamination[contamination]
    result = evaluate_path_a(model, X_test, meta_test, k_pcts=k_pcts, target_recall=target_recall)
    result["capacity_severity_table"] = capacity_and_severity_table(model, X_test, meta_test, k_pcts=k_pcts)
    result["production_k"] = select_production_k(result["capacity_severity_table"])

    result["path"] = "A"
    result["path_selection_diagnostics"] = diagnostics
    return result

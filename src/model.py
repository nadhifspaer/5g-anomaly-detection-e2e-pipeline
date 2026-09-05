# Isolation Forest training + comparison-only baseline. See architecture.md
# Section 3, Section 5 Step 3, Section 7. src/model.py owns these two
# functions only; no routing logic, no scoring interface (src/score_pipeline.py).

import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import feature_engineering as fe

# bump when src/feature_engineering.py's FEATURE_COLS changes
FEATURE_SET_VERSION = "v1"

# three-sigma rule, comparison baseline only, never wired into serving
BASELINE_ZSCORE_THRESHOLD = 3.0

EXPERIMENT_NAME = "isolation-forest-5g-kpi"


def derive_contamination_path_a(meta_train: pd.DataFrame) -> float:
    # measured sla_compliant == 0 rate on train-side data, not a guess
    return float((meta_train["sla_compliant"] == 0).mean())


def _mlruns_uri() -> str:
    # local file-based MLflow store under mlruns/, per architecture.md Section 6
    return (Path(__file__).resolve().parent.parent / "mlruns").as_uri()


def train_isolation_forest(
    X_train: pd.DataFrame,
    meta_train: pd.DataFrame,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict:
    """Primary model: Isolation Forest on Stage 2's train-side feature matrix.

    Evaluation Path A only (architecture.md Section 7/8, resolved at Stage
    0.2, not re-litigated): contamination is the measured sla_compliant == 0
    rate on meta_train, a single value, stated here as measured, not assumed.

    Logs the run to the local MLflow store under mlruns/: contamination,
    n_estimators, feature_set_version, training_seconds.

    Returns {"path": "A", "runs": {contamination: {"model", "contamination",
    "training_seconds"}}}.
    """
    contamination = derive_contamination_path_a(meta_train)

    mlflow.set_tracking_uri(_mlruns_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    start = time.perf_counter()
    model.fit(X_train)
    training_seconds = time.perf_counter() - start

    with mlflow.start_run(run_name=f"iforest_pathA_c{contamination:.4f}"):
        mlflow.log_param("evaluation_path", "A")
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("feature_set_version", FEATURE_SET_VERSION)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("training_seconds", training_seconds)
        mlflow.sklearn.log_model(model, "model")

    return {
        "path": "A",
        "runs": {contamination: {"model": model, "contamination": contamination, "training_seconds": training_seconds}},
    }


def rolling_zscore_threshold_baseline(
    X: pd.DataFrame, threshold: float = BASELINE_ZSCORE_THRESHOLD
) -> pd.Series:
    # comparison baseline only, never wired into serving (architecture.md Section 2)
    max_abs_z = X[fe.ZSCORE_COLS].abs().max(axis=1)
    return (max_abs_z > threshold).astype(int).rename("baseline_flag")

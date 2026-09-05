# Streamlit dashboard, APP_MODE=cloud (default) or local. See architecture.md
# Section 4, Section 5 Step 6, Section 9. Single codebase, two modes.
#
# cloud: model + src/score_pipeline.py embedded in-process, sample-picker
# over held-out Stage 2 test data. No Kafka, no FastAPI.
# local: polls streaming/scored_results.db (Stage 5.2 consumer output),
# live-updating alert table + summary panel.

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
# `streamlit run` (unlike `python -m streamlit run`) does not add the current
# working directory to sys.path, only this script's own dashboard/ directory,
# so a bare `streamlit run dashboard/app.py` from the project root cannot
# find the sibling src/ package without this. Confirmed via diagnostic:
# bare `streamlit run` sys.path has no project-root entry at all; `python -m
# streamlit run` does (Python's -m semantics insert cwd). Insert unconditionally
# so this holds regardless of invocation style, cwd, or OS path separators.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from src import feature_engineering as fe
from src import model as m
from src import score_pipeline as sp

DATA_PATH = ROOT_DIR / "data" / "5g_kpi_dataset.csv"
# SCORED_DB_PATH override: docker-compose's consumer writes to a shared named
# volume, not the bind-mounted streaming/ directory (docker/docker-compose.yml)
DB_PATH = Path(os.environ.get("SCORED_DB_PATH", ROOT_DIR / "streaming" / "scored_results.db"))
LOCAL_POLL_SECONDS_DEFAULT = 5
LOCAL_ROW_LIMIT = 200
TOP_CELLS_SHOWN = 5
# 300 rows at 1/sec fits a ~5 minute viewing session while still spanning the
# entire test window; evenly-spaced striding (not the first 300 rows) avoids
# biasing the replay toward only the earliest part of that window
LIVE_REPLAY_ROWS = 300
LIVE_REPLAY_INTERVAL_SECONDS = 1

st.set_page_config(page_title="5G KPI Anomaly Detection", layout="wide")


# ---------------- shared layout / scoring-display components ----------------


def render_header(mode: str) -> None:
    st.title("5G Network KPI Anomaly Detection")
    st.caption(f"APP_MODE={mode} | Isolation Forest, per-(cell_id, slice_type) rolling baseline")


def render_score_panel(cell_id: str, slice_type: str, timestamp, anomaly_score: float, alert: bool) -> None:
    cols = st.columns(4)
    cols[0].metric("Cell", cell_id)
    cols[1].metric("Slice", slice_type)
    cols[2].metric("Anomaly score", f"{anomaly_score:.4f}")
    cols[3].metric("Alert", "YES" if alert else "no")
    st.caption(f"Timestamp: {timestamp}")


def render_root_cause(root_cause: list) -> None:
    st.subheader("Root cause: top deviating KPIs (z-score)")
    rc_df = pd.DataFrame(root_cause)
    if rc_df.empty:
        st.write("No root-cause ranking available.")
        return
    st.bar_chart(rc_df.set_index("kpi")["zscore"])
    st.dataframe(rc_df[["kpi", "zscore"]], hide_index=True, use_container_width=True)


def render_raw_kpi(raw_row: dict) -> None:
    # actual measured values, distinct from render_root_cause's z-score deviations
    st.subheader("Raw KPI values (actual, not z-score)")
    kpi_row = {col: [raw_row[col]] for col in fe.KPI_COLS}
    st.dataframe(pd.DataFrame(kpi_row), hide_index=True, use_container_width=True)


def _highlight_alerts(row: pd.Series) -> list:
    color = "background-color: #ffcccc" if row.get("alert") else ""
    return [color] * len(row)


# ---------------- cloud mode ----------------


@st.cache_data
def load_stage2_split():
    # Stage 2 temporal split; meta_test rows already passed build_feature_matrix's
    # dropna, so every one has sufficient (cell_id, slice_type) backward history
    df = fe.load_raw(str(DATA_PATH))
    X, meta = fe.build_feature_matrix(df.copy())
    X_train, _, meta_train, meta_test = fe.temporal_train_test_split(X, meta)
    return df, X_train, meta_train, meta_test


@st.cache_resource
def load_model_cloud():
    """Trains fresh in-process via src/model.py's train_isolation_forest, never
    mlflow.sklearn.load_model(runs:/...): a run's artifact_uri is an absolute
    path baked in at training time, tied to the filesystem that trained it.
    That already broke twice moving this project between a Windows host and a
    Linux container (in both directions); Streamlit Community Cloud's own
    server filesystem is a fourth, unrelated absolute-path environment this
    can never assume. Cached so training runs once per app session, not once
    per interaction. Local mode and api/main.py still load via MLflow, correctly,
    since docker-compose controls that path to stay consistent end to end."""
    _, X_train, meta_train, _ = load_stage2_split()
    result = m.train_isolation_forest(X_train, meta_train)
    contamination = next(iter(result["runs"]))
    return result["runs"][contamination]["model"]


def _lookup_raw_row(df: pd.DataFrame, cell_id: str, slice_type: str, timestamp) -> dict:
    # identity match (timestamp, cell_id, slice_type), never positional
    return (
        df[(df["cell_id"] == cell_id) & (df["slice_type"] == slice_type) & (df["timestamp"] == timestamp)]
        .iloc[0]
        .to_dict()
    )


def _score_test_row(df: pd.DataFrame, model, cell_id: str, slice_type: str, timestamp) -> dict:
    raw_row = _lookup_raw_row(df, cell_id, slice_type, timestamp)
    # backward-only: same-group rows strictly before the picked timestamp, per
    # src/feature_engineering.py's rolling window (no forward leakage)
    history = df[
        (df["cell_id"] == cell_id) & (df["slice_type"] == slice_type) & (df["timestamp"] < timestamp)
    ]
    return sp.score_row(raw_row, history, model)


def _verify_picker_coverage(meta_test: pd.DataFrame) -> dict:
    """Sums row counts across every (cell_id, slice_type) combination present
    in meta_test and confirms the total equals len(meta_test): every row is
    reachable through some cell/slice/timestamp selectbox combination in
    render_sample_picker, none pooled away or hidden by a dropdown that can't
    reach it."""
    combos = meta_test[["cell_id", "slice_type"]].drop_duplicates()
    total = sum(
        int(((meta_test["cell_id"] == c) & (meta_test["slice_type"] == s)).sum())
        for c, s in combos.itertuples(index=False)
    )
    return {"total": total, "expected": len(meta_test), "combos": len(combos), "matches": total == len(meta_test)}


def render_sample_picker() -> None:
    model = load_model_cloud()
    df, _, _, meta_test = load_stage2_split()

    st.sidebar.header("Sample picker (held-out test data)")

    coverage = _verify_picker_coverage(meta_test)
    st.sidebar.caption(
        f"Picker coverage check: {coverage['total']}/{coverage['expected']} meta_test rows reachable "
        f"across {coverage['combos']} (cell_id, slice_type) combinations "
        f"({'OK' if coverage['matches'] else 'MISMATCH, see console'})."
    )
    if not coverage["matches"]:
        st.sidebar.error("Sample picker coverage mismatch: not every meta_test row is reachable. See CLAUDE.md checklist.")

    cell_id = st.sidebar.selectbox("Cell", sorted(meta_test["cell_id"].unique()))
    slice_options = sorted(meta_test.loc[meta_test["cell_id"] == cell_id, "slice_type"].unique())
    slice_type = st.sidebar.selectbox("Slice type", slice_options)
    ts_options = sorted(
        meta_test.loc[
            (meta_test["cell_id"] == cell_id) & (meta_test["slice_type"] == slice_type), "timestamp"
        ]
    )
    timestamp = st.sidebar.selectbox("Timestamp", ts_options)

    result = _score_test_row(df, model, cell_id, slice_type, timestamp)
    raw_row = _lookup_raw_row(df, cell_id, slice_type, timestamp)

    render_score_panel(
        result["cell_id"], result["slice_type"], result["timestamp"], result["anomaly_score"], result["alert"]
    )
    render_raw_kpi(raw_row)
    render_root_cause(result["root_cause"])


@st.cache_resource
def build_live_replay_rows():
    """Scores LIVE_REPLAY_ROWS samples evenly strided across meta_test's full
    timestamp-sorted range (architecture.md-style global order, matching
    streaming/producer.py's own sort_values("timestamp") convention), not the
    first LIVE_REPLAY_ROWS rows in sequence. Computed once per process."""
    model = load_model_cloud()
    df, _, _, meta_test = load_stage2_split()
    sorted_meta = meta_test.sort_values("timestamp").reset_index(drop=True)

    n = len(sorted_meta)
    count = min(LIVE_REPLAY_ROWS, n)
    positions = np.unique(np.round(np.linspace(0, n - 1, count)).astype(int))
    sampled_meta = sorted_meta.iloc[positions].reset_index(drop=True)

    results = [
        _score_test_row(df, model, row["cell_id"], row["slice_type"], row["timestamp"])
        for _, row in sampled_meta.iterrows()
    ]
    return results, sorted_meta["timestamp"].min(), sorted_meta["timestamp"].max()


def render_live_replay() -> None:
    results, test_min_ts, test_max_ts = build_live_replay_rows()
    total = len(results)

    if "live_replay_idx" not in st.session_state:
        st.session_state.live_replay_idx = 0
    if "live_replay_playing" not in st.session_state:
        st.session_state.live_replay_playing = False
    if "live_replay_autopause" not in st.session_state:
        st.session_state.live_replay_autopause = True
    # revealed count we last auto-paused at, so a resumed Play doesn't
    # immediately re-trigger the pause on the same still-current alert row
    if "live_replay_autopaused_at" not in st.session_state:
        st.session_state.live_replay_autopaused_at = None

    revealed = st.session_state.live_replay_idx
    finished = revealed >= total
    current = results[revealed - 1] if (not finished and revealed > 0) else None

    # auto-pause detection runs before the sidebar controls render below, so
    # the Play/Pause button label reflects the post-auto-pause state on the
    # same rerun it happens, not one rerun late. Only triggers on a freshly
    # revealed row during active playback, never re-triggers on the same row
    # when Play is pressed again to resume past it.
    if (
        current is not None
        and st.session_state.live_replay_autopause
        and current["alert"]
        and st.session_state.live_replay_playing
        and st.session_state.live_replay_autopaused_at != revealed
    ):
        st.session_state.live_replay_playing = False
        st.session_state.live_replay_autopaused_at = revealed

    st.sidebar.header("Live replay controls")
    st.sidebar.checkbox("Auto-pause on alert", key="live_replay_autopause")
    if finished:
        if st.sidebar.button("Reset", use_container_width=True):
            st.session_state.live_replay_idx = 0
            st.session_state.live_replay_playing = False
            st.session_state.live_replay_autopaused_at = None
    else:
        play_label = "Pause" if st.session_state.live_replay_playing else "Play"
        ctrl_cols = st.sidebar.columns(2)
        if ctrl_cols[0].button(play_label, use_container_width=True):
            st.session_state.live_replay_playing = not st.session_state.live_replay_playing
        if ctrl_cols[1].button("Reset", use_container_width=True):
            st.session_state.live_replay_idx = 0
            st.session_state.live_replay_playing = False
            st.session_state.live_replay_autopaused_at = None
            revealed = 0
            finished = False

    st.caption(
        f"Sampled {total} rows, evenly spaced across the full temporal test "
        f"split ({test_min_ts} to {test_max_ts}), not the first {total} rows "
        "in sequence, so the replay stays spread across the whole window."
    )

    if finished:
        st.success(f"Replay finished: all {total} sampled rows played. Press Reset to play again.")
    elif revealed == 0:
        st.info("Press Play to start the replay.")
    else:
        st.write(f"Row {revealed} of {total} (sampled from full test period: {test_min_ts} to {test_max_ts})")
        if st.session_state.live_replay_autopaused_at == revealed and not st.session_state.live_replay_playing:
            st.warning(
                f"Paused: anomaly detected at Row {revealed} "
                f"({current['cell_id']}/{current['slice_type']}, {current['timestamp']}) "
                "— press Play to continue."
            )
        render_score_panel(
            current["cell_id"], current["slice_type"], current["timestamp"],
            current["anomaly_score"], current["alert"],
        )
        render_root_cause(current["root_cause"])

    played = results[:revealed]
    if played:
        st.subheader("Replay feed")
        feed_df = pd.DataFrame(
            [
                {
                    "timestamp": r["timestamp"],
                    "cell_id": r["cell_id"],
                    "slice_type": r["slice_type"],
                    "anomaly_score": r["anomaly_score"],
                    "alert": r["alert"],
                }
                for r in played
            ]
        )
        st.dataframe(feed_df.style.apply(_highlight_alerts, axis=1), hide_index=True, use_container_width=True)

    if st.session_state.live_replay_playing and not finished:
        time.sleep(LIVE_REPLAY_INTERVAL_SECONDS)
        st.session_state.live_replay_idx += 1
        st.rerun()
    elif st.session_state.live_replay_playing and finished:
        st.session_state.live_replay_playing = False


def render_cloud_mode() -> None:
    render_header("cloud")
    view = st.sidebar.radio("View", ["Sample Picker", "Live Replay"])
    st.sidebar.divider()
    if view == "Sample Picker":
        render_sample_picker()
    else:
        render_live_replay()


# ---------------- local mode ----------------


def _read_scored_results(limit: int = LOCAL_ROW_LIMIT) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query(
            "SELECT id, cell_id, slice_type, timestamp, anomaly_score, alert, top_kpis, scored_at "
            "FROM scored_results ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    finally:
        conn.close()
    return df


def render_local_mode() -> None:
    render_header("local")

    poll_seconds = st.sidebar.slider("Poll interval (seconds)", 2, 30, LOCAL_POLL_SECONDS_DEFAULT)
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    if st.sidebar.button("Refresh now"):
        st.rerun()

    df = _read_scored_results()
    if df.empty:
        st.info(f"No scored rows yet in {DB_PATH}. Start the docker-compose producer/consumer/API stack.")
    else:
        df["scored_at_dt"] = pd.to_datetime(df["scored_at"])
        today = pd.Timestamp.now().normalize()
        alerts_today = int(((df["alert"] == 1) & (df["scored_at_dt"] >= today)).sum())

        summary_cols = st.columns(3)
        summary_cols[0].metric("Alerts scored today", alerts_today)
        summary_cols[1].metric(f"Rows in feed (last {LOCAL_ROW_LIMIT})", len(df))
        summary_cols[2].metric(f"Alert rate (last {LOCAL_ROW_LIMIT})", f"{df['alert'].mean() * 100:.1f}%")

        st.subheader(f"Top affected cells (last {LOCAL_ROW_LIMIT} rows)")
        top_cells = (
            df[df["alert"] == 1]["cell_id"]
            .value_counts()
            .head(TOP_CELLS_SHOWN)
            .rename_axis("cell_id")
            .reset_index(name="alerts")
        )
        if top_cells.empty:
            st.write("No alerts in the current window.")
        else:
            st.dataframe(top_cells, hide_index=True, use_container_width=True)

        st.subheader("Recently scored rows")
        display_df = df[["timestamp", "cell_id", "slice_type", "anomaly_score", "alert", "scored_at"]].copy()
        display_df["alert"] = display_df["alert"].astype(bool)
        st.dataframe(
            display_df.style.apply(_highlight_alerts, axis=1),
            hide_index=True,
            use_container_width=True,
        )

        alerted = df[df["alert"] == 1].head(10)
        if not alerted.empty:
            st.subheader("Root cause for recent alerts")
            for _, row in alerted.iterrows():
                with st.expander(f"{row['cell_id']} / {row['slice_type']} @ {row['timestamp']}"):
                    render_root_cause(json.loads(row["top_kpis"]))

    if auto_refresh:
        time.sleep(poll_seconds)
        st.rerun()


# ---------------- entry point ----------------


def main() -> None:
    app_mode = os.environ.get("APP_MODE", "cloud").strip().lower()
    if app_mode == "local":
        render_local_mode()
    elif app_mode == "cloud":
        render_cloud_mode()
    else:
        st.error(f"Unknown APP_MODE={app_mode!r}; expected 'cloud' or 'local'.")


if __name__ == "__main__":
    main()

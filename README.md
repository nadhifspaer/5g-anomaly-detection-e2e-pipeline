# End-to-End ML Pipeline for 5G Network Anomaly Detection

**Live demo: https://5g-anomaly-detection-e2e-pipeline-vmg4drna4c2aadbpzdizpv.streamlit.app/** (Usually takes about ~3 minutes to load)

## Introduction & Goals

This project detects anomalous behavior in 5G radio access network KPIs using a single unsupervised Isolation Forest, scored in near real time against each cell's own recent history rather than a fixed global threshold. A network operations team reviewing raw KPI dashboards across a heterogeneous cell population, macro, micro, and pico cells, four different network slices, cannot rely on one threshold for everything: a latency value that is normal for a URLLC slice is a serious problem for an eMBB slice on the same physical cell. The system scores every KPI reading against that specific cell-and-slice pair's own recent history, flags the readings that deviate most from that pair's own normal range, and ranks which KPIs drove the flag, so an analyst opening an alert sees a plain-language answer for what looked wrong, not just a bare score.

The dataset is 5,000 rows of 5G network KPI readings at 1-minute granularity, spanning `2024-01-01 00:00:00` to `2024-01-04 11:19:00`, across 20 cells and 4 network slices (`eMBB`, `mMTC`, `URLLC`, `HC`), 80 `(cell_id, slice_type)` groups in total, ranging from 18 to 122 rows per group. Each row carries eight raw KPI columns (`throughput_mbps`, `latency_ms`, `packet_loss_pct`, `handover_count`, `rsrp_dbm`, `rsrq_db`, `prb_utilization_pct`, `active_users`) plus `sla_compliant`, a binary business-rule flag derived from those same KPIs, used only to evaluate the model and never as a training feature.

### Goals

- **Catch real SLA-relevant anomalies well above chance, without labeled training data.** **How I know it worked:** PR-AUC 0.181 on the held-out test split, against a no-skill baseline of about 0.07.
- **Keep daily alerts inside what an operations team can actually review.** **How I know it worked:** the production threshold (top 2% of daily volume) yields 30.4 alerts/day, precision 0.333, recall 0.130, within a stated 36 alerts/day capacity (10 min/alert, 2h/shift, 3 shifts/day).
- **Justify a multivariate, tunable model over a simple threshold rule.** **How I know it worked:** the three-sigma comparison baseline fires at 208.6 alerts/day, about 5.8x that same capacity, with no way to dial it down; only Isolation Forest's continuous score can be matched to the capacity budget directly.
- **Prove the real-time serving path (Kafka to FastAPI to SQLite to the dashboard) works end to end, not just in a notebook.** **How I know it worked:** a full `docker-compose` run scored the entire held-out test split (994 rows) through the live API and produced exactly 24 alerts, matching the documented real-time threshold's expected result exactly (precision 0.333, recall 0.148). This is a fixed score cutoff approximating the top-2% production threshold above for single-row scoring, not a conflicting number.

## Approach

**Isolation Forest** (`scikit-learn`) is the single model driving every score, unsupervised and multivariate, trained on rolling z-score deviation features rather than raw KPI values. Chosen for concrete reasons: no distributional assumptions about the feature space, fast training without a GPU, a `contamination` parameter that maps directly onto an empirically measured anomaly rate, and a continuous score that Precision@K/Recall@K alert-budget tuning needs directly. Every KPI is converted to a rolling z-score computed strictly backward from each row's own `timestamp`, grouped by `(cell_id, slice_type)`, not `cell_id` alone: `slice_type` changes the KPI expectation profile structurally (URLLC targets low latency and high reliability, mMTC targets high device density at low throughput, eMBB targets high throughput), and pooling slice types into one cell's baseline would blend different operating ranges together and produce false positives whenever the slice mix shifts. `cell_type` (`micro`/`macro`/`pico`) varies per row within a `cell_id` in this dataset, so it is kept for exploratory segmentation only, never a grouping key. The trained model is wrapped by one stable-interface scoring function (`src/score_pipeline.py`), shared unchanged by the API, the streaming consumer, and both dashboard modes, and every flagged row is paired with a per-feature deviation-contribution ranking (`src/root_cause.py`), so an alert comes with a plain-language reason attached.

**FastAPI** wraps that same scoring function behind a `/score` endpoint, returning the anomaly score, alert flag, and root-cause ranking for one submitted row. It was chosen as a fast, async-capable Python framework that lets the streaming consumer and any other caller share the exact scoring logic the dashboard's cloud mode calls in-process, with no reimplementation elsewhere. It loads the latest MLflow-tracked run at startup and runs only in the local, full-stack deployment mode, never in the public demo.

**Kafka** stands in for a live network telemetry feed: a producer replays held-out test-period KPI rows in simulated real time, and a consumer reads each one, calls the FastAPI `/score` endpoint, and writes the scored result to a shared store. It was chosen to demonstrate a genuine produce-consume-score-store streaming architecture without needing an actual live 5G network connection, which this project has no access to. `streaming/producer.py` and `streaming/consumer.py` run only inside the local docker-compose stack.

**Streamlit** is the one codebase behind both modes, `dashboard/app.py`, switching on an `APP_MODE` environment variable. It was chosen because a single lightweight, pure-Python app can serve both a self-contained public demo, Streamlit Community Cloud hosts only one process and has no room for Kafka or a separate backend, and a full live-polling dashboard, without maintaining two frontends. Cloud mode embeds the model in-process with a sample picker (994 held-out rows, every one reachable through the cascading picker) and a Live Replay view (300 samples spread across the test window, with an auto-pause-on-alert toggle); local mode polls the Kafka-fed shared store instead of scoring in-process.

**Docker and Docker Compose** containerize and orchestrate the full local stack, the API, Kafka, the consumer, the dashboard, Prometheus, and Grafana, as one `docker-compose up` command. This was chosen to demonstrate a production-style, multi-service architecture reproducibly on one machine, without every reviewer installing Kafka, Prometheus, and Grafana separately by hand. The API container reads the local MLflow tracking store through a read-only bind mount rather than baking a specific model artifact into the image, so the served model always matches whatever run was most recently trained.

**Prometheus and Grafana** monitor the serving layer itself, not the network: Prometheus scrapes metrics from the FastAPI service, and Grafana visualizes them. This pairing was chosen as the standard open-source combination for system health monitoring, kept deliberately separate from the network-operations view the Streamlit dashboard provides, since API latency and error rate answer a different question than network health. Both run only in the local stack and are never exposed publicly.

## Results Highlights

Full detail, methodology, and every number's derivation live in `reports/model_evaluation.md`. `sla_compliant == 0` is a workable minority-class label (6.44% whole-file rate, 6.80% on the train-side split), and it is not reproducible from any single raw KPI column at a level meaningfully above chance, confirming it reflects a genuine multivariate SLA determination rather than a single-column proxy.

Isolation Forest (`contamination=0.0680`, the measured train-side `sla_compliant == 0` rate, `n_estimators=200`), scored on the temporal test split (994 rows):

- **PR-AUC: 0.181**, against a no-skill baseline of about 0.07 on this test split's base rate.
- **Production alert threshold: top 2% of daily scored volume**, selected by a stated review-capacity assumption (10 minutes/alert, 2 hours/shift, 3 shifts/day = 36 alerts/day = 2.5% of daily volume), never a dollar-cost figure and never a fixed alert count asserted without basis. At this threshold: 30.4 alerts/day, precision 0.333, recall 0.130.
- **Comparison baseline** (three-sigma rule on any single KPI's rolling z-score, comparison only, never served): fires on 14.49% of test rows, 208.6 alerts/day, about 5.8 times the stated capacity, with no way to tune that volume down without changing the rule itself. Isolation Forest's advantage over this baseline is not raw detection accuracy at every volume, it is that only a continuous, rankable score can be matched to a review budget the team can actually staff.

## Business Impact

Full narrative, assumptions, and every derived figure live in `reports/business_impact.md`. At the production operating point, the system uses about 84% of the stated daily review-time budget and surfaces roughly 10 confirmed SLA violations a day alongside roughly 20 false alarms, catching about 13% of real violations in the test window and concentrating that coverage on the more severe ones first. The remaining violations go undetected at this capacity-constrained operating point. No dollar net-value figure is given anywhere in this project: this dataset carries no financial data to ground one. Net value is instead stated in review-time and detection-coverage terms, all traceable to the numbers above.

## Repository Structure

```
data/                          # 5g_kpi_dataset.csv (not committed)
notebooks/                     # 01_eda.ipynb, 02_feature_engineering_and_model.ipynb
src/
  feature_engineering.py       # per-(cell_id, slice_type) rolling deviation features, temporal split
  model.py                     # Isolation Forest training + comparison baseline
  score_pipeline.py            # single stable-interface scoring function
  root_cause.py                # deviation-contribution ranking
  evaluate.py                  # evaluation metrics, capacity-and-severity K selection
api/
  main.py                      # FastAPI app, /score endpoint
streaming/
  producer.py
  consumer.py
dashboard/
  app.py                       # APP_MODE=cloud or local
monitoring/
  prometheus.yml
  grafana/dashboards/
docker/
  Dockerfile.api
  Dockerfile.dashboard
  docker-compose.yml
mlruns/                        # MLflow local tracking store
reports/
  model_evaluation.md
  business_impact.md
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run cloud mode locally:

```
APP_MODE=cloud streamlit run dashboard/app.py
```

Run the full local stack:

```
docker-compose -f docker/docker-compose.yml up
```

## Further Reading

- `reports/model_evaluation.md`: the complete evaluation methodology, the capacity-and-severity threshold-selection derivation, the baseline comparison, and the real-time alert threshold approximation and its limitation.
- `reports/business_impact.md`: the full business-impact narrative behind the summary above.

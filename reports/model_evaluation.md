# Model Evaluation

## Evaluation Label: `sla_compliant`

- Class balance (whole file): 322 rows with `sla_compliant == 0` (6.44%), 4,678 rows with `sla_compliant == 1`.
- Single-column triviality check: the best single-column threshold (`packet_loss_pct`) clears the majority-class base rate by only 0.24 points. Point-biserial correlations against every raw KPI are weak: `latency_ms` 0.153, `active_users` 0.144, `packet_loss_pct` -0.132, all others under 0.05.
- Conclusion: `sla_compliant` reflects a multivariate SLA determination, not a single-KPI proxy, and is usable as the evaluation label. Metrics used: PR-AUC (primary), Precision@K / Recall@K, FPR at a fixed target recall. Plain accuracy is never reported, given the 6.44% minority rate.

## Data Integrity Checks

- 5,000 rows, no nulls in any column, 0 fully duplicated rows, 0 duplicate `(timestamp, cell_id, slice_type)` tuples, 0 duplicate timestamps overall.
- `timestamp` parses cleanly and is monotonic increasing both globally and within every one of the 80 `(cell_id, slice_type)` groups.
- Row volume is flat at 60 rows/hour (one row per minute) across the full range, dropping only at the final partial hour. Confirms a single interleaved stream, not 80 independently sampled per-group streams.
- `cell_type` non-compliance rate is flat across categories (macro 6.6%, micro 6.6%, pico 5.8%), no meaningful separation. `cell_type` stays EDA segmentation only, never a grouping key or feature.

## Feature Design

- Every KPI (`throughput_mbps`, `latency_ms`, `packet_loss_pct`, `handover_count`, `rsrp_dbm`, `rsrq_db`, `prb_utilization_pct`, `active_users`) is converted to a rolling z-score computed within its own `(cell_id, slice_type)` pair, never per `cell_id` alone and never against a global statistic across all cells.
- Grouping rationale: `slice_type` changes the KPI expectation profile structurally (URLLC targets low latency and high reliability, mMTC targets high device density at low throughput, eMBB targets high throughput). Grouping by `cell_id` alone would blend these different operating ranges into one baseline and produce false positives whenever the slice mix on a cell shifts, independent of any real degradation.
- `cell_type` is excluded from the grouping key because it varies per row within a `cell_id` in this dataset and was never the driver of the KPI expectation shift the grouping key needs to capture.
- Backward-only window: `add_rolling_zscore_features` applies `shift(1)` within each `(cell_id, slice_type)` group before computing the rolling mean and standard deviation, so a row's baseline uses only rows strictly earlier in that group's history, never the row itself and never a later row.
- Verification (`CELL_015`/`eMBB`, row at `2024-01-01 09:14:00`, `throughput_mbps = 723.63`): backward window spans `2024-01-01 00:40:00` to `2024-01-01 08:20:00`, confirmed not to include `09:14:00` itself. Manual z-score from that window's 10 raw values (`mean = 875.666`, `std = 95.008`) matches the pipeline's computed z-score to floating-point precision (both `-1.60024`).
- Window size: `window=10, min_periods=3`. Group sizes range 18 to 122 rows (mean 62.5); the `HC` slice is sparsest at every cell (18 to 35 rows, mean 25.6), `CELL_016`/`HC` the global minimum at 18 rows. With `min_periods=3`, the first 3 rows of every group are dropped (3 x 80 groups = 240 rows dropped, 4,760 of 5,000 raw rows survive). The smallest group (`CELL_016`/`HC`, 18 rows) still yields 15 rows with a valid feature after the drop. A longer window would leave `HC` rows with too little backward history for a meaningful baseline across most of their range.

## Contamination Rate

- Measured on the temporal train split (`meta_train`, 3,766 rows): `sla_compliant == 0` rate **6.80%** (256 of 3,766 rows). Used as the model's `contamination` value, not the whole-file rate (6.44%) and not a guess.
- Best single-column edge over base rate on the train split: 0.27pp (`packet_loss_pct`), within the required bounds (1% floor, 5pp triviality ceiling), confirming the label is usable on train-side data as well as whole-file data.

## Isolation Forest Results

Model: `n_estimators=200`, `contamination=0.0680`, `random_state=42`. Scored on the temporal test split (994 rows, `2024-01-03 18:46` to `2024-01-04 11:19`).

- **PR-AUC: 0.181** (test base rate ~7%, so well above the no-skill baseline of ~0.07).
- **FPR at target recall 0.90: 0.713.** Reaching 90% recall requires alerting on roughly 71% of true-negative rows.

| K (% of daily volume) | n_alerts | precision | recall |
|---|---|---|---|
| 0.5% | 6 | 0.500 | 0.056 |
| 1% | 11 | 0.364 | 0.074 |
| 2% | 21 | 0.333 | 0.130 |
| 5% | 50 | 0.260 | 0.241 |
| 10% | 100 | 0.180 | 0.333 |
| 15% | 150 | 0.160 | 0.444 |

At review-budget-realistic K (0.5-5%), precision stays in the 0.26-0.50 range while recall stays low (5-24%). A 90%-recall operating point is not compatible with a small daily alert budget on this test window.

## Capacity-and-Severity K Selection

- No financial data grounds a dollar-cost metric for this dataset, and a cost curve (missed-anomaly cost minus false-alert cost) never converges to an interior minimum; it keeps favoring higher K all the way to the edge of any swept range. K selection instead uses two stated assumptions:
- **Review-capacity assumption:** 10 minutes of analyst time per alert, 2 hours per shift dedicated to this system's alerts, 3 shifts covering 24 hours. Arithmetic: (2 x 60) / 10 = 12 alerts/analyst-shift x 3 shifts = **36 alerts/day capacity**. Daily row volume is ~1,440 rows (60 rows/hour x 24h), so 36 / 1,440 = **2.5% capacity cap** on K.
- **Severity of missed anomalies:** for every `sla_compliant == 0` row not alerted at a given K, the mean absolute rolling z-score across the 8 KPI z-score features (excluding `hour_of_day`/`day_of_week`, which are calendar context, not deviation measures).

| K (% of daily volume) | alerts/day | precision | recall | false negatives | mean severity missed | max severity missed | within 2.5% cap |
|---|---|---|---|---|---|---|---|
| 0.5% | 8.7 | 0.500 | 0.056 | 51 | 1.142 | 1.910 | yes |
| 1% | 15.9 | 0.364 | 0.074 | 50 | 1.128 | 1.910 | yes |
| 2% | 30.4 | 0.333 | 0.130 | 47 | 1.086 | 1.800 | yes |
| 5% | 72.4 | 0.260 | 0.241 | 41 | 1.027 | 1.425 | no |
| 10% | 144.9 | 0.180 | 0.333 | 36 | 0.995 | 1.425 | no |
| 15% | 217.3 | 0.160 | 0.444 | 30 | 0.937 | 1.267 | no |

- **Production K: 2%** (30.4 alerts/day, precision 0.333, recall 0.130): the largest swept candidate at or under the 2.5% capacity cap, and (since recall is non-decreasing in K under top-K alerting) also the highest-recall candidate under that cap. The next candidate up, K=5%, already exceeds capacity at 72.4 alerts/day, roughly double the 36/day assumption.
- Mean severity of missed anomalies decreases monotonically as K grows (1.142 at K=0.5% down to 0.937 at K=15%): the highest-severity rows are caught first as more alerts are allowed. At K=2%, the 47 still-missed rows average 1.086 mean absolute z-score, max 1.800, both moderate rather than extreme.

## Baseline Comparison

- `rolling_zscore_threshold_baseline` (three-sigma rule: flags a row if any single KPI's rolling z-score exceeds 3 in absolute value) exists only as a comparison, never wired into serving.
- Flag rate: train 655 of 3,766 rows (17.39%), test 144 of 994 rows (14.49%).
- Test performance against `sla_compliant == 0`: precision 0.208, recall 0.556 (30 of 54 true anomalies caught, 144 total alerts). Scaled to alerts/day: ~208.6 alerts/day, roughly 5.8x the 36/day capacity assumption and roughly 6.9x the picked Isolation Forest operating point (K=2%, 30.4 alerts/day).
- Matched-volume comparison: the baseline's 144 test-window alerts sit close to Isolation Forest's K=15% volume (150 alerts). At that matched volume, the baseline has higher precision (0.208 vs. 0.160) and higher recall (0.556 vs. 0.444) than Isolation Forest at K=15%. Isolation Forest does not dominate the baseline in raw detection performance on this test split.
- Isolation Forest is still the production choice because it produces a continuous, rankable anomaly score that can be tuned to any capacity level, including the 2.5% cap the review-capacity assumption supports. The baseline is a fixed binary rule with exactly one operating point (~208.6 alerts/day, already ~5.8x over capacity) and no way to dial down without changing the rule itself.

## Real-Time Alert Threshold

- Batch K=2% ranks each row's anomaly score against the rest of that day's rows, so it is only known once the day's scores are complete. The `/score` endpoint scores one row at a time in real time, before the day is over, so it cannot rank against an incomplete day.
- `src/score_pipeline.py` uses a fixed absolute score cutoff instead: `ANOMALY_SCORE_ALERT_THRESHOLD = 0.026575134021403518`, the 98th percentile of anomaly scores on train-side data (matching K=2%'s implied percentile).
- Gap versus batch K=2%, measured on the test split: fixed cutoff gives 24 alerts, precision 0.333, recall 0.148; batch K=2% gives 30.4 alerts/day, precision 0.333, recall 0.130. Precision matches; the fixed cutoff alerts on slightly fewer rows and catches slightly more of them, since it does not adapt to day-to-day shifts in the score distribution the way per-day ranking does. Close enough to treat as the same operating point for this dataset, not exact.
- **Staleness limitation:** `ANOMALY_SCORE_ALERT_THRESHOLD` is computed once from train-side data at build time and is not recalibrated automatically as new data arrives. Network KPI patterns can drift over time (new cell deployments, changing traffic patterns), so the fixed threshold can become stale. Periodic recalibration (for example weekly, using a trailing window of recent data) is the natural next step; it is not implemented here.

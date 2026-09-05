# Business Impact

## Summary

- The system flags about 30 cell/slice readings per day for analyst review.
- Of those, about 10 are real SLA violations and about 20 are false alarms.
- That review load uses about 84% of the daily review budget.
- It catches about 13% of the SLA violations present in the test window. The remaining 87% are missed.
- The violations it does catch skew toward the more severe ones, since alerts are ranked by anomaly score and the highest-severity cases are the first to be caught as more alerts are allowed.

## Review Capacity Assumptions

- 10 minutes of analyst time per alert reviewed.
- 2 hours per shift dedicated to this system's alerts, 3 shifts covering 24 hours.
- Review-time budget: 2 hours x 3 shifts = 6 hours/day, 360 minutes/day.
- Alert-review capacity: 360 minutes / 10 minutes per alert = 36 alerts/day.
- No dollar figure is assumed for a missed anomaly, a false alert, or an hour of analyst time. These are time-budget assumptions, stated explicitly.

## Net Value at the Production Operating Point (K=2%)

| Quantity | Value |
|---|---|
| Alerts/day | 30.4 |
| True positives/day (real violations caught) | ~10.1 |
| False positives/day (false alarms) | ~20.3 |
| Real violations/day (total, caught and missed) | ~77.9 |
| Real violations missed/day | ~67.7 |
| Analyst review time/day | ~304 minutes (~5.1 hours) |
| Share of the 6-hour/day capacity budget used | ~84.4% |

- The system uses most, not all, of the budgeted review time.
- It converts that time into roughly 10 confirmed problems surfaced per day, alongside roughly 20 false alarms the analyst rules out and dismisses.
- Recall at this operating point is 0.130: most real violations are not caught at this capacity level. This is a direct consequence of the capacity cap, not a hidden limitation.

## Versus No Automated Monitoring

**Versus zero proactive review:**
- With no system in place, zero analyst-hours are spent and zero violations are caught proactively. Every violation surfaces only through some other channel: a customer complaint, a billing dispute, or a manual audit triggered after the fact.
- Against that baseline, catching roughly 10 real violations a day, concentrated on the more severe ones, is a genuine gain from a starting point of zero.

**Versus full manual review of every row:**
- Reviewing all ~1,440 rows/day at 10 minutes each requires 14,400 minutes/day (240 hours/day), about 40 times the 6-hour/day review budget.
- This is not viable under the stated staffing assumption. The system's role is to convert an infeasible full audit into a bounded, capacity-matched review list.

## A Third Comparison Point: The Simple Per-KPI Threshold Rule

- A simpler three-sigma rule on individual KPIs was also considered as an alternative to the model.
- That rule would require about 34.8 hours of review time per day to work through its alert volume, far beyond the 6-hour/day capacity.

## Limitations

- At current review capacity, 87% of real SLA violations in the test window go undetected.
- Every number in this report depends on the stated review-capacity assumptions (10 minutes/alert, 2 hours/shift, 3 shifts/day). Changing those assumptions changes every figure here.
- The `sla_compliant` label used to measure catch rate is a business-rule flag, not an independently confirmed ground-truth anomaly log.
- The alert threshold is fixed at build time and is not recalibrated automatically as network conditions change over time.

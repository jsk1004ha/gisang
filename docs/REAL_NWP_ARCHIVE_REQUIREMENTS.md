# Real NWP Archive Requirements

The current local prepared forecast CSV has 48 rows, 2 stations, and horizons 1-24. It is enough for schema/smoke validation only, not honest training/evaluation.

Minimum for operational validation:

- Multiple stations and regions, initially at least 20 ASOS stations.
- Multiple issue times per day or at least daily issue cycles.
- Several weeks minimum; several months preferred for time-ordered train/val/test.
- Horizon coverage for 1-24h for each station and issue time.
- UTC-normalized `forecast_init_time`/`issue_time`, `valid_time`, `horizon_step`.
- Station-id + issue-time + horizon uniqueness.
- No synthetic/smoke/generated/fixture provenance for operational scoring.

For a 72h encoder and 24h prediction window, each evaluated station needs enough observation history before each issue time and enough future observations after it. The archive must support separate calibration/selection/test periods without leakage.

# V3 Real-Data Results

Date: 2026-05-25

## Summary

Real-data run evidence should be taken from generated experiment artifacts and the unified report:

- `data/artifacts/v3_experiments/`
- `data/artifacts/leaderboard*.csv`
- `reports/experiment_summary.csv`
- `reports/experiment_report.html`

A full V3 temperature MOS run was attempted with:

```bash
PYTHONPATH=src .venv312/Scripts/python.exe -m weather_korea_forecast.v2.train \
  --config configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml \
  --update-report --report-title "Gisang V1-V3 Experiment Report"
```

The run path now detects stale cached training tables and rebuilds them when the
configured V3 feature columns are missing. Results from the latest successful run
are summarized below when present.

## Required experiment sequence

```bash
scripts/run_v3_all_core_experiments.sh
```

or run the temperature and humidity tracks separately:

```bash
scripts/run_v3_temp_mos_experiments.sh
scripts/run_v3_humidity_experiments.sh
```

## Result table

After runs complete, regenerate and inspect:

```bash
PYTHONPATH=src .venv312/Scripts/python.exe -m weather_korea_forecast.reporting.generate_report \
  --experiments-root data/artifacts --output-dir reports --title "Gisang V1-V3 Experiment Report"
```

Then read:

- `reports/experiment_summary.csv` for all V1/V2/V3 runs.
- `reports/best_models.csv` for target/track top 5.
- `reports/failed_or_incomplete_experiments.csv` for failed/incomplete artifacts.

## Goal interpretation

- Temperature NWP-assisted MOS goal: `rmse <= 1.0`, worst horizon `<= 1.2`, worst station `<= 1.4`, absolute bias `<= 0.2`.
- Temperature observation-only goal: `rmse <= 2.0`.
- Humidity goal: `rh_rmse/rmse <= 10.0%p`, good target `<= 8.0%p`, absolute bias `<= 3.0%p`.

`rmse_goal`, `rmse_goal_met`, and `rmse_gap_to_goal` are written automatically to experiment summaries, leaderboards, and the unified report.

## Current limitations

- Live GFS/ECMWF/KMA downloader integration remains a V4/operations task; prepared forecast CSV is the validated interface.
- CatBoost remains optional and is not run unless the dependency is installed.
- RMSE <= 1.0°C can only be claimed from completed full-data artifacts, not from synthetic tests.

## 2026-05-25 observed artifacts

The unified report currently discovers 66 completed experiment summaries under
`data/artifacts`.

Best non-oracle temperature NWP-assisted/MOS artifact in the current report:

| experiment | RMSE | MAE | Bias | RMSE goal met |
|---|---:|---:|---:|---|
| `v3_temp_mos_residual_ridge_72to24` | 1.0641 | 0.8042 | 0.0623 | false |

This is close to, but still above, the 1.0°C target by about `+0.064°C`.

The attempted rerun on 2026-05-25 rebuilt the stale cached V3 training table
because `nwp_temp_c` / `obs_minus_nwp_temp_*` columns were missing. The rerun was
manually stopped after several minutes in this interactive session before a new
`experiment_summary.json` was written. The existing completed full-data artifacts
above remain the current real-data evidence. Re-run with the scripts when longer
wall-clock execution is available.

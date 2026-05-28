# G024 Operational Performance Sprint

G024 is the first operational-valid MOS benchmark path that trains and tests on real forecast NWP plus real KMA ASOS observations. It is for performance improvement and gate evidence after a prepared forecast archive has passed the G023 archive quality gate.

## Scope

Official operational baselines are:

- Temperature: `raw_gfs_t2m`, `operational_residual_lgbm_temp`
- Humidity: `raw_gfs_rh`, `operational_residual_lgbm_humidity`

Ridge residual models are kept in the debug section only until their operational residual behavior is trustworthy. They do not count as official operational baselines when they underperform raw GFS.

## CLI

```bash
PYTHONPATH=src .venv312/Scripts/python.exe -m weather_korea_forecast.v4.operational_performance \
  --nwp-archive data/raw/nwp/archive/prepared_forecast_archive.csv \
  --archive-quality-report data/raw/nwp/archive/archive_quality_report.json \
  --observations data/raw/asos/asos_hourly.csv \
  --station-metadata data/raw/metadata/stations.csv \
  --output-dir data/artifacts/g024_operational_performance
```

The command refuses to run unless `archive_quality_report.forecast_archive_adequate` is the boolean `true`.

## Method

For each target, the runner:

1. Joins prepared forecast rows to hourly ASOS observations by `station_id` and UTC `valid_time`.
2. Uses a strict time-ordered cycle split: latest 30 cycles as 20 train / 5 validation / 5 test.
3. Reports raw GFS target-column metrics on the test split.
4. Fits residual LightGBM models (`actual - nwp_baseline`) with a small hyperparameter grid.
5. Evaluates no-patch, `patch3`, and `patch5` station-neighborhood feature modes.
6. Fits calibration candidates on the first validation subset and selects them on validation holdout only.
7. Applies calibration to test predictions only when validation holdout selection beats the uncalibrated candidate; test metrics are never used to choose calibration.
8. Writes ridge residual sign/formula diagnostics separately.

`patch3` and `patch5` currently use a station-neighborhood proxy from real GFS station-nearest values. True GRIB-grid patch extraction remains a separate V4-B/V4-C extension.

## Always-written artifacts

Every successful run writes:

- `experiment_summary.json`
- `operational_performance_report.html`
- `operational_performance_report.md`
- `summary_snapshot.json`
- `predictions_test.csv`
- `joined_training_frame.csv`
- `models.pkl`
- `<target>_patch_ablation_results.csv`
- `<target>_<mode>_lgbm_grid_search_results.csv`
- `<target>_<mode>_calibration_candidate_results.csv`
- `<target>_best_lgbm_params.json`
- `<target>_feature_importance.csv`
- `<target>_ridge_residual_debug_report.md`
- `<target>_ridge_residual_debug_rows.csv`
- ridge residual diagnostic plots when matplotlib is available

The HTML report is generated automatically as part of training/evaluation output, so there is always a visual summary for completed G024 runs.

## Benchmark reliability

G024 labels archive strength separately from adequacy:

| Label | Condition |
| --- | --- |
| `smoke` | fewer than 30 forecast cycles |
| `short` | at least 30 cycles |
| `medium` | at least 90 cycles |
| `strong` | at least 180 cycles |
| `seasonal` | at least 365 cycles, at least 365 days, and all 4 seasons |

A 30-cycle real benchmark is useful evidence but remains `short`; it should not be described as final production performance.

## Gates

V4-C remains `FAIL` unless all required conditions are met:

- operational-valid temperature model exists
- temperature operational RMSE `<= 1.5°C`
- humidity operational RMSE `<= 10%p`
- benchmark reliability is at least `medium`
- patch ablation completed
- report generated

Site readiness is `WARN` when humidity passes but temperature or benchmark reliability is still short of beta criteria.

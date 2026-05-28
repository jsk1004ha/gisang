# G025 Operational Model Accuracy Sprint

G025 extends the G024 operational-valid benchmark from a 30-cycle short archive to a medium archive target and adds true GFS grid-patch evidence, stronger LGBM tuning, optional CatBoost, calibration enhancements, and ensemble artifacts.

## Goals

- Expand real forecast archive coverage to at least 90 forecast cycles when public data is available.
- Keep all archive adequacy gates honest: a short archive must not pass V4-C/site beta gates as medium.
- Compare station-neighborhood proxy patch features against true GRIB-grid 3x3/5x5 patch summaries.
- Keep `raw_gfs_t2m` / `raw_gfs_rh` and residual LightGBM as the official operational baselines.
- Keep Ridge in debug/reference unless it beats raw GFS on operational data.
- Attempt to reduce temperature RMSE below `1.5°C` while keeping humidity RMSE below `10%p`.

## Medium archive build evidence

The local G025 run used real NOAA GFS files and real KMA ASOS observations:

- Older forecast cycles came from NOAA public GFS S3 full-message GRIB files for 2026-05-04 through 2026-05-18.
- Recent forecast cycles came from the existing G023/G024 NOMADS regional archive for 2026-05-19 through 2026-05-26 06Z.
- KMA ASOS observations were downloaded for 30 stations through 2026-05-27.

Ignored local outputs:

```text
data/raw/nwp/archive/prepared_forecast_archive_g025_medium.csv
data/raw/nwp/archive/archive_quality_report_g025.json
data/raw/nwp/archive/archive_quality_report_g025.md
data/raw/asos/asos_g025_medium_20260504_20260527_30stations.csv
```

Quality report target values:

```text
forecast_cycle_count = 90
station_count = 30
horizon_1_24_coverage = 1.0
forecast_archive_adequate = true
benchmark_reliability = medium
```

The older S3 slice intentionally guarantees `nwp_t2m`, `nwp_humidity`, and `nwp_dew_point`; wind/pressure/precip fields remain richer in the recent NOMADS slice. This is adequate for a medium temperature/humidity MOS benchmark but should be expanded to the full operational variable set before treating the archive as a mature production benchmark.

## True GFS grid patch extraction

Use `weather_korea_forecast.v4.gfs_grid_patch` to extract station-centered patch summaries directly from GRIB files without committing generated data:

```bash
PYTHONPATH=src .venv312/Scripts/python.exe -m weather_korea_forecast.v4.gfs_grid_patch \
  --grib-glob 'data/raw/nwp/g025_medium/s3_gfs_global_cache_20260504_20260518_rh2m_t2m_d2m/*.grib2' \
  --station-metadata data/raw/metadata/stations_kma_datawiki_30_live.csv \
  --output-csv data/raw/nwp/g025_medium/gfs_true_grid_patch3_s3_20260504_20260518_60cycles.csv \
  --patch-size 3 \
  --variables nwp_t2m,nwp_humidity,nwp_dew_point
```

Patch summary columns include center, mean, std, min, max, range, sum, gradient_x, gradient_y, and upwind_mean for each requested variable. Rows carry `patch_feature_mode=true_gfs_grid_patch` so reports can distinguish them from the station-neighborhood proxy mode.

## Operational performance command

```bash
PYTHONPATH=src .venv312/Scripts/python.exe -m weather_korea_forecast.v4.operational_performance \
  --nwp-archive data/raw/nwp/archive/prepared_forecast_archive_g025_medium.csv \
  --archive-quality-report data/raw/nwp/archive/archive_quality_report_g025.json \
  --observations data/raw/asos/asos_g025_medium_20260504_20260527_30stations.csv \
  --station-metadata data/raw/metadata/stations_kma_datawiki_30_live.csv \
  --grid-patch-features data/raw/nwp/g025_medium/gfs_true_grid_patch3_patch5_g025_medium_90cycles.csv \
  --output-dir data/artifacts/g025_operational_accuracy_medium
```

The runner uses all available cycles for medium-or-larger archives with a strict time-ordered split. For 90 cycles the split is 62 train / 14 validation / 14 test cycles. For exactly 30 cycles it preserves the G024 20 / 5 / 5 split.

Every successful run writes both machine-readable artifacts and a visual HTML report:

```text
experiment_summary.json
operational_performance_report.html
operational_performance_report.md
predictions_test.csv
joined_training_frame.csv
models.pkl
*_patch_ablation_results.csv
*_lgbm_grid_search_results.csv
*_calibration_candidate_results.csv
*_ensemble_weights.json
*_ensemble_component_metrics.csv
*_ensemble_predictions_test.csv
```


## Medium run result (local G025 evidence)

The medium run completed with `data/artifacts/g025_operational_accuracy_medium/operational_performance_report.html` and `experiment_summary.json`.

| Target | Model / mode | RMSE | MAE | Bias | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Temp | Raw GFS | 2.637°C | 2.043°C | -1.330°C | real GFS baseline |
| Temp | Residual LGBM true_patch5 | 1.703°C | 1.314°C | -0.181°C | best selected LGBM mode |
| Temp | Ensemble constrained least squares | 1.683°C | 1.296°C | -0.171°C | best validation-selected ensemble |
| Humidity | Raw GFS RH | 12.051%p | 9.411%p | 0.593%p | real GFS baseline |
| Humidity | Residual LGBM true_patch5 | 11.781%p | 9.347%p | -6.169%p | validation-selected official LGBM mode did not generalize |
| Humidity | Residual LGBM no_patch | 10.406%p | 8.182%p | -2.503%p | best test ablation, still above 10%p |

Patch ablation evidence:

- Temperature true_patch5 improved no_patch RMSE by about `0.125°C`, late-horizon RMSE by about `0.102°C`, and worst-station RMSE by about `0.542°C`.
- Humidity true patches did not beat no_patch on the test split; the medium run invalidates the short-benchmark humidity PASS as not yet stable.

Gate status:

```text
V4-C: FAIL (temp_rmse <= 1.5 not met; humidity_rmse <= 10 not met on medium)
Site readiness: WARN (medium archive is present, but temp and humidity gates are not both met)
```

## Gate interpretation

Operational beta can pass only when all of the following are true:

- temperature operational-valid RMSE `<= 1.5°C`
- humidity operational-valid RMSE `<= 10%p`
- benchmark reliability is at least `medium`
- patch ablation completed
- visual report generated
- diagnostic/oracle/smoke/generated exports remain blocked

A medium archive alone does not pass V4-C; the temperature RMSE gate remains decisive.

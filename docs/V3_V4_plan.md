# V3 / V4 Plan

## V3 definition

V3 turns the V2 experiment pipeline into an operationally separated station-level NWP-assisted/MOS forecasting system for Korea.

V3 is not a new model-only layer. It is a production-readiness layer over the existing station pipeline:

- keep observation-only and NWP-assisted/MOS experiments on separate tracks;
- mark ERA5 reanalysis decoder covariates as backtest-only unless replaced by forecast NWP inputs;
- train temperature as station-level NWP residual correction;
- keep humidity as a separate redesign track using RH, dew point, and dew-point depression targets;
- require station/region/horizon/daily reports for every run.

## V3 first-pass scope

The first V3 pass is intentionally compatible with the existing V2 CLI:

```bash
python -m weather_korea_forecast.v2.train --config configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml
```

The config is versioned as `v3` and writes to `data/artifacts/v3_experiments/`, but it reuses the V2 training/evaluation/inference code path while V3 stabilizes.

### Required V3 metadata

Every V3 experiment artifact and leaderboard row must preserve:

- `forecast_track`
- `uses_future_weather_features`
- `future_feature_source`
- `operational_valid`
- `backtest_only`
- `target_name`
- `model_family`

Stable umbrella leaderboards are:

- `leaderboard_observation_only.csv`
- `leaderboard_nwp_assisted.csv`
- `leaderboard_humidity.csv` / `leaderboard_temp.csv`

Exact track-specific leaderboards, such as `leaderboard_nwp_assisted_mos.csv`, may also be written.

### Temperature MOS baseline

The first V3 temperature MOS config is:

```text
configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml
```

It trains:

```text
baseline = era5_t2m_c   # future NWP/ERA5-style 2m temperature
model target = observed_temp - baseline
prediction = baseline + predicted_residual
```

It includes station-level NWP bias-history features such as:

- `obs_minus_era5_temp_lag_1`
- `obs_minus_era5_temp_lag_6`
- `obs_minus_era5_temp_lag_24`
- `obs_minus_era5_temp_lag_48`
- `obs_minus_era5_temp_roll_mean_24`
- `obs_minus_era5_temp_roll_mean_72`
- `obs_minus_era5_temp_roll_std_24`

Because the starter source is ERA5 reanalysis, the config is explicitly:

```text
forecast_track: nwp_assisted_mos
future_feature_source: era5_reanalysis
operational_valid: false
backtest_only: true
```

Operational inference must replace ERA5 reanalysis with a forecast NWP source supplied through `--future-weather-csv`.

### Honest humidity NWP-MOS path

Humidity `RMSE < 1` cannot be claimed from observed-target oracle or
target-derived future dew point.  The honest path is now separated into an
issue-time-aligned V3 runner:

```text
python -m weather_korea_forecast.data.gfs_surface_forecast
python -m weather_korea_forecast.v3.nwp_mos --config configs/v3/experiments/v3_humidity_nwp_mos_lgbm_72to24.yaml
```

The forecast archive must contain `station_id`, `issue_time`, `valid_time`,
`lead_hour`, and future NWP moisture columns such as
`nwp_relative_humidity_2m` / `nwp_dew_point_2m_c`.  The MOS trainer joins
observations by `(station_id, issue_time)` for init-time history and
`(station_id, valid_time)` for the target, then learns a residual from the NWP
RH baseline.  This avoids using a forecast run issued after the sample's
forecast-init time.  The NOAA GFS extractor is optional and depends on the
`nwp` extra (`cfgrib`/`eccodes`).

### Diagnostic oracle ceiling checks

The V3 diagnostic config below is intentionally not an operational model:

```text
configs/v3/experiments/diagnostic/v3_temp_observed_oracle_decoder_feature_72to24.yaml
configs/v3/experiments/diagnostic/v3_humidity_observed_oracle_decoder_feature_72to24.yaml
```

They copy future observed `target_value` from the decoder slice and therefore
use actual future observations. Use them only to verify artifact/evaluator
plumbing or to establish an RMSE ceiling; they must remain marked
`forecast_track: observed_target_oracle`, `operational_valid: false`, and
`backtest_only: true`.

### Humidity V3 track

Humidity remains a separate V3 track. Compare at least:

1. direct RH prediction;
2. dew point prediction with RH restoration;
3. dew-point depression prediction with RH restoration.

Starter configs:

```text
configs/v3/experiments/v3_humidity_direct_lgbm_72to24.yaml
configs/v3/experiments/v3_humidity_dewpoint_lgbm_72to24.yaml
configs/v3/experiments/v3_humidity_dewpoint_depression_lgbm_72to24.yaml
```

The dew point/depression configs reuse the V2 target-transform implementation while all three configs write V3 artifacts and leaderboards.

The preferred V3 humidity chain is:

```text
temp MOS forecast -> predicted_temp feature -> dew point/depression model -> RH restoration -> clip 0..100
```

### Stronger prediction ensemble

V3 can strengthen point forecasts by ensembling already-generated experiment artifacts:

```bash
python -m weather_korea_forecast.v2.ensemble \
  --experiment-dir <direct-rh-experiment-dir> \
  --experiment-dir <dew-point-experiment-dir> \
  --experiment-dir <dew-point-depression-experiment-dir> \
  --output-root data/artifacts/v3_experiments \
  --name v3_humidity_mean_ensemble_72to24 \
  --method mean \
  --clip-min 0 --clip-max 100 \
  --leaderboard-path data/artifacts/v3_experiments/leaderboard.csv
```

The ensemble CLI aligns component `predictions_test.csv` files by forecast key,
verifies shared actuals, writes ensemble predictions/components, evaluates with
the standard V2 report stack, and updates V3 leaderboards.

### Web status dashboard

V3 status can be monitored locally without adding a web dependency:

```bash
python -m weather_korea_forecast.dashboard.app \
  --artifact-root data/artifacts/v3_experiments \
  --host 127.0.0.1 \
  --port 8765
```

The dashboard reads artifact files only. It shows latest/best aliases, best
experiments by target/track, full leaderboard rows, and comparison report links.
Use `--write-html data/artifacts/v3_experiments/dashboard.html` for a static
snapshot.

## V4 definition

V4 extends beyond station-level MOS into national, spatial, probabilistic, and climate-generalized forecasting.

V4 adds:

- NWP/ERA5 grid patch extraction around stations (`3x3`, `5x5`, `9x9`);
- spatial encoders such as PatchCNN, ConvLSTM, or graph encoders;
- ERA5-Land, DEM, slope/aspect, land-sea mask, land cover, urban fraction;
- nationwide ASOS/AWS training with station quality scores;
- probabilistic forecasts (`P10`, `P50`, `P90`, mean, uncertainty);
- 24h, 72h, and optionally 120h forecast horizons;
- multi-year and seasonal rolling validation;
- physical consistency checks across temperature, humidity, and dew point;
- scheduled inference, automatic evaluation, and model/feature versioning.

## Completion criteria

### V3 complete when

1. observation-only and NWP-assisted/MOS leaderboards are separated;
2. ERA5 backtest and operational forecast-NWP modes are distinct;
3. temperature residual MOS exists and is validated;
4. humidity supports direct RH plus dew point/depression target tracks;
5. station metadata avoids collapsed `unknown` region reporting for supported stations;
6. at least 20 ASOS stations can run through train/evaluate/inference;
7. leakage/backtest warnings are written into artifacts;
8. standard station/region/horizon/daily reports are generated.

### V4 complete when

1. national ASOS/AWS training is supported;
2. NWP patch extraction and at least one spatial encoder are implemented;
3. temperature/humidity/dew point forecasts pass physical consistency checks;
4. 24h and 72h forecasts are supported;
5. probabilistic forecast evaluation is available;
6. seasonal rolling validation and operational automatic evaluation are in place.

## 2026-05-25 V3 MOS implementation update

Temperature NWP-assisted/MOS now has a fuller experiment matrix under `configs/v3/experiments/`:

1. `v3_temp_mos_residual_ridge_72to24`
2. `v3_temp_mos_residual_ridge_168to24`
3. `v3_temp_mos_horizonwise_residual_ridge_72to24`
4. `v3_temp_mos_horizonwise_residual_ridge_168to24`
5. `v3_temp_mos_residual_lgbm_72to24`
6. `v3_temp_mos_residual_lgbm_168to24`
7. `v3_temp_mos_horizonwise_residual_lgbm_72to24`
8. `v3_temp_mos_residual_catboost_72to24` (optional dependency)
9. `v3_temp_mos_ensemble_72to24` (artifact-level ensemble run spec)

All ridge V3 MOS configs use `alpha_grid = logspace(-4, 4, 33)` and select only on validation-derived loss.  The generic MOS bias feature family is now named around `nwp_temp_c` / `obs_minus_nwp_temp_*` so the same configs can use ERA5 reanalysis in backtests and later swap to GFS/ECMWF/KMA forecast fields through the future-weather adapter.

Additional artifact contract fields are now written to summaries and leaderboards:

- `track` as an alias for `forecast_track`
- `uses_future_weather_features`
- `leakage_risk_note`
- `worst_station_rmse`

Residual target-transform runs also write MOS component columns in `predictions_test.csv`: `baseline_prediction`, `predicted_residual`, `actual_residual`, `prediction_raw`, `prediction_corrected`, `error`, and `abs_error`.

The forecast-source adapter entry point is:

```python
load_future_weather_features(source, start_time, horizon, stations, path=..., config=...)
```

Supported source names are `era5_reanalysis_backtest`, `gfs_forecast`, `ecmwf_forecast`, and `kma_forecast`.  Live download implementations remain future work; prepared CSV tables are supported now.  Operational inference can be guarded with `--operational`, which rejects backtest-only ERA5/reanalysis future features.

New V3 plots include:

- `daily_diurnal_range_error.png`
- `worst_station_timeseries.png`
- `worst_horizon_samples.png`
- `residual_scatter.png`
- `baseline_vs_final_scatter.png`

## 2026-05-25 real-run and reporting update

Implemented for V3 before V4 handoff:

- sequential runner scripts for the core V3 temp MOS and humidity experiment order;
- validation-only LightGBM grid-search artifacts;
- RMSE goal fields in summaries/leaderboards;
- stricter prepared forecast CSV adapter validation;
- stale cached training-table detection/rebuild when configured V3 features are absent;
- standalone unified HTML/CSV experiment reporting across V1/V2/V3;
- train/evaluate `--update-report` integration.

Still considered skeleton or V4 work:

- live operational GFS/ECMWF/KMA downloaders beyond prepared CSV ingestion;
- broad observation-only tuning beyond the added V3 horizon-wise baseline config;
- CatBoost production use unless the optional dependency is installed and verified;
- claiming RMSE <= 1.0°C without a completed full real-data V3 run artifact.

Before V4, the preferred gate is:

1. run `scripts/run_v3_all_core_experiments.sh` on the real data snapshot;
2. inspect `reports/experiment_report.html` and `reports/experiment_summary.csv`;
3. confirm best temp NWP-assisted MOS goal status and worst-horizon/station constraints;
4. replace ERA5 reanalysis future covariates with prepared forecast CSV or a live forecast adapter for operational validation.

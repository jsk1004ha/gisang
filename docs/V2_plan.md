# V2 Plan

## Current V1 Status

### Implemented in V1

- Observation ingestion for ASOS and optional AWS/local CSV supplementation.
- ERA5 extraction at stations with `nearest` and `bilinear` modes.
- UTC-normalized training table generation.
- Sliding-window dataset generation for direct sequence modeling.
- Baselines: `persistence`, `seasonal_persistence`, `ridge`.
- TFT wrapper with `auto` backend resolution.
- Train / evaluate / predict CLIs with experiment artifacts.
- `latest/` and `best/` experiment aliases.

### Gaps That V2 Must Address

- V1 is still oriented around prototype-scale station runs.
- V1 defaults are not organized around single-target temp/humidity experiments.
- Feature engineering is still light for production-like weather forecasting.
- Experiment tracking exists, but V2 needs stricter summaries and leaderboard updates.
- TFT exists, but V2 should treat real TFT as the main sequence path.

## V2 Scope

- Preserve V1 behavior.
- Add a separate V2 pipeline under `src/weather_korea_forecast/v2/`.
- Use unified V2 experiment configs under `configs/v2/experiments/`.
- Default to direct `24h` forecasting and single-target experiments.
- Support multi-station inputs with station metadata and geographic features.
- Preserve config-driven ASOS primary plus optional supplementary observation sources; `prefer_columns` can prioritize selected AWS variables without replacing the primary target column.

## Files That Needed Refactoring

- `src/weather_korea_forecast/data/station_metadata.py`
- `src/weather_korea_forecast/features/geo_features.py`
- `src/weather_korea_forecast/models/baselines.py`
- `src/weather_korea_forecast/models/registry.py`
- `src/weather_korea_forecast/models/tft_model.py`
- `pyproject.toml`

## New V2 Files

- `src/weather_korea_forecast/v2/data.py`
- `src/weather_korea_forecast/v2/dataset.py`
- `src/weather_korea_forecast/v2/scaling.py`
- `src/weather_korea_forecast/v2/train.py`
- `src/weather_korea_forecast/v2/evaluate.py`
- `src/weather_korea_forecast/v2/predict.py`
- `src/weather_korea_forecast/v2/prepare_data.py`
- `src/weather_korea_forecast/v2/artifacts.py`
- `configs/v2/experiments/*.yaml`
- `configs/v2/templates/station_metadata_template.csv`
- `tests/test_v2_pipeline.py`
- `scripts/run_v2_*.sh`

## V2 Data Flow

```text
observations
-> UTC normalization
-> station metadata join
-> ERA5 join/extraction
-> time features
-> lag / rolling / delta features
-> split assignment
-> train-only scaler fit
-> direct multi-horizon datasets
-> train / evaluate / predict
```

## V2 Defaults

- Single-target experiments:
  - `v2_temp_*`
  - `v2_humidity_*`
- Direct forecast horizon: `24h`
- Recommended encoder lengths:
  - `72h`
  - `168h`
- Geographic static features:
  - `lat`
  - `lon`
  - `elevation`
  - `coastal_distance_km`
  - `region_class`

## Standard V2 Artifacts

- `experiment_config.yaml`
- `model.pt`
- `scaler.json`
  - supports `global`, `station_wise`/`stationwise`, `region_wise`/`regionwise`, and `none`
  - group-wise modes store train-split-only group means/stds
- `training_history.json`
- `predictions_test.csv`
- `metrics_test.json`
- `metrics_summary.json`
- `metrics_target_name.csv`
- `metrics_target_name_horizon_step.csv`
- `metrics_target_name_station_id.csv`
- `metrics_target_name_region.csv`
- `metrics_target_name_season.csv`
- `metrics_raw_target_name.csv`
- `metrics_raw_target_name_horizon_step.csv`
- `metrics_raw_target_name_station_id.csv`
- `metrics_raw_target_name_region.csv`
- `metrics_raw_target_name_season.csv`
- `forecast_vs_actual.png`
- `horizon_error.png`
- `prediction_scatter.png`
- `raw_vs_corrected.png`
- `horizon_station_heatmap.png`
- `station_rmse_bar.png`
- `region_rmse_bar.png`
- `daily_max_min_error.png`
- `extreme_target_scatter.png`
- `bias_correction.json`
  - includes calibration/holdout selection evidence when `apply_when: improves_on_holdout`
  - defaults to station+horizon guarded mean-bias correction for multi-station V2 configs
- `future_feature_metadata.json`
  - marks `forecast_track`, `uses_future_nwp_features`, `future_feature_source`, and `operational_valid`
  - keeps ERA5 reanalysis decoder-covariate backtests separate from operational NWP-assisted forecasts
  - supports `method: affine` for guarded horizon-wise `slope * prediction + intercept` calibration
  - supports additional modes: `per_station`, `per_season_horizon`, `per_region_horizon`
  - supports `method: quantile` for empirical quantile correction candidates
  - supports `mode: auto` / `method: auto` candidate comparison for research runs
- Ridge configs can set `model.alpha_selection.metric: bias_corrected_holdout_mse` so `alpha_grid` is selected against the same guarded station+horizon correction objective used in evaluation, instead of raw validation MSE only.
- `worst_case_samples.csv`
- `worst_case_summary.json`
- `feature_importance.csv` for supported baselines
- `horizon_model_metrics.csv` for horizon-wise ridge/lightgbm and residual components
- `predictions_test_components.csv` for residual experiments
- `daily_target_errors.csv`
- `metrics_daily_target.csv`
- `metrics_humidity_extremes.csv` for humidity dry/humid event diagnostics
- `metrics_target_name_rolling_origin_fold.csv`
- `experiment_summary.json`
- `experiment_summary.md`
- `leaderboard.csv`
- `leaderboard_<target_name>.csv`
- `leaderboard_<forecast_track>.csv`

NWP-assisted / MOS configs must set `data.future_features.source`. ERA5 reanalysis sources are treated as backtest upper-bound experiments (`operational_valid: false`) unless replaced by forecast NWP inputs. Temperature residual MOS configs can use `data.target_transform.type: residual_from_feature` with `baseline_column: era5_t2m_c` so the learner predicts station-level correction residuals rather than absolute temperature directly.

`decoder_feature_baseline` is available for forecast-model baselines and explicit oracle/backtest ceiling checks: it copies configured `model.target_source_features` from decoder-known covariates into the target. Any config that uses observed-target-derived future decoder features must remain marked `operational_valid: false` and documented as non-canonical.

Operational inference passes prepared forecast covariates with:

```bash
python -m weather_korea_forecast.v2.predict \
  --experiment-dir data/artifacts/v2_experiments/latest \
  --station-id 108 \
  --forecast-init-time 2025-01-03T00:00:00Z \
  --future-weather-csv data/raw/nwp/latest_station_forecast.csv
```

The forecast CSV contract is `station_id`, `valid_time`/`datetime`, optional `issue_time`, and weather columns mapped by `data.future_features.column_mapping`. If `issue_time` exists, prediction selects the latest issue not later than the requested forecast init time. Decoder target lag features such as `target_value_lag_24/48/72` are rebuilt from historical encoder data so operational inference does not require future observation rows.

## Remaining Follow-ups

- V3 handoff now starts with `configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml`, a V2-CLI-compatible config that formalizes temperature MOS residual learning under `forecast_track: nwp_assisted_mos`.
- V3 humidity starter configs also exist under `configs/v3/experiments/`: `v3_humidity_direct_lgbm_72to24.yaml`, `v3_humidity_dewpoint_lgbm_72to24.yaml`, and `v3_humidity_dewpoint_depression_lgbm_72to24.yaml`. They compare direct RH against V2 target-transform restoration while keeping V3 artifacts separate.
- V3 can now build stronger artifact-level ensembles with `python -m weather_korea_forecast.v2.ensemble`; this reuses component `predictions_test.csv` files, writes a new evaluated experiment directory, and updates V3 leaderboards.
- V3 artifact status can be inspected locally with `python -m weather_korea_forecast.dashboard.app --artifact-root data/artifacts/v3_experiments`.
- V2 temperature now has a strong official ridge baseline; follow-up experiments should compare against the corrected RMSE/MAE/Bias leaderboard row rather than treating persistence as the main target.
- `horizon_wise_ridge` is available for horizon-specific intercept/alpha learning, with `72->24` and `168->24` configs.
- The generic `residual` wrapper supports baseline + residual learner experiments for baseline/ridge/lightgbm/fallback-torch style models and writes component predictions.
- `fallback_torch` now has a residual shortcut from recent observed targets; true-pytorch-forecasting residual dataset support remains a follow-up item.
- Rolling-origin reporting exists, but true rolling-origin retraining is not yet the default path.
- The canonical 12-20 station benchmark still depends on user-provided local ASOS/ERA5 data files.
- Local real-data bootstrap configs currently cover Seoul smoke runs, not the full multi-station benchmark.
- Multi-station setup now has config-driven raw data bootstrap files, but still depends on successful KMA/CDS downloads in the user environment.

## V3 compatibility note (2026-05-25)

V3 MOS configs continue to execute through the V2 train/evaluate/predict CLIs, but their artifacts are track-separated.  Leaderboard rows now carry `track`, `uses_future_weather_features`, `future_feature_source`, `operational_valid`, `backtest_only`, `leakage_risk_note`, and worst station/horizon fields.  ERA5 future decoder features must be interpreted as backtest-only unless replaced by prepared forecast NWP features via the future-weather adapter.

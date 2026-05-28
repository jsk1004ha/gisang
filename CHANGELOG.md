## G025 - Operational Model Accuracy Sprint

### Added
- Added true GFS GRIB-grid patch extraction utilities for station-centered 3x3/5x5 patch summaries.
- Added operational runner support for `--grid-patch-features`, explicit `patch_feature_mode` reporting, optional CatBoost residual baselines, and ensemble artifacts.
- Added medium-archive time-ordered splitting so 90-cycle benchmarks use all cycles instead of silently falling back to the 30-cycle short split.
- Added G025 operational accuracy documentation and tests for true patch extraction, optional CatBoost skip, ensemble artifacts, and medium reliability gates.

### Changed
- Residual LightGBM remains the official operational MOS baseline, while Ridge stays in debug/reference unless it beats raw GFS.
- V4-C and site-readiness gates now consume best operational model metrics while still requiring benchmark reliability at least `medium`.

### Verified
- `PYTHONPATH=src .venv312/Scripts/python.exe -m ruff check .`
- `PYTHONPATH=src .venv312/Scripts/python.exe -m compileall src tests`
- `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q` (212 passed, 64 warnings)

# Changelog

## 2026-05-28

### Added
- Added the G024 operational performance runner for real forecast NWP + real ASOS benchmarks, with official raw-GFS and residual-LightGBM baselines, calibration selection, station-neighborhood patch ablation, LightGBM grid artifacts, ridge residual diagnostics, V4-C/site-readiness gate summaries, benchmark reliability labels, and always-generated HTML reports.
- Added G023 real NWP forecast archive acquisition adapters for KMA local forecast CSVs, GFS/NOMADS local/extracted tables, and an ECMWF local-table skeleton.
- Added richer prepared forecast archive quality reports with station/cycle/horizon coverage tables, per-station adequacy, expected-column missing rates, archive SHA-256 binding, blocking reasons, and humidity/dew-point/weather-code sanity checks.
- Added a G022 operational training gate that requires strict boolean `forecast_archive_adequate=true`, matching archive SHA-256, and configured forecast feature columns before model fitting.
- Added NWP Archive Status integration to the HTML report and documentation for KMA/GFS archive build workflows and quality gates.

## 2026-05-14

### Added
- Added `artifacts.profile: minimal` for V3.5/V4-style runs that should keep only report-critical CSV/JSON artifacts while relying on unified CSV/HTML reports instead of per-run plot bundles.
- Added the V3.5 diagnostic humidity ERA5-RH residual LightGBM config, `configs/v3/experiments/v3_5_humidity_era5_rh_residual_lgbm_72to24.yaml`, marked backtest-only/non-operational because the current local ERA5 station CSV lacks real dew-point/RH forecast inputs.
- Added diagnostic-aware unified reporting: oracle/decoder-feature sanity checks are separated from main KPI/best rankings, `best`/`latest` aliases are excluded from representative leaderboards, `included_in_main_leaderboard` records the exact main-row decision, target/track-specific best cards replace mixed-unit overall best RMSE, and HTML image embedding now supports `full`, `thumbnail`, and `external-assets` modes.
- Added humidity feature sanity reporting for dew-point Celsius ranges and required `hour_sin/hour_cos/doy_sin/doy_cos` covariates.
- Added an issue-time-aligned V3 humidity NWP-MOS runner (`weather_korea_forecast.v3.nwp_mos`) plus `configs/v3/experiments/v3_humidity_nwp_mos_lgbm_72to24.yaml` for honest residual learning from forecast RH/dew-point archives.
- Added `weather_korea_forecast.data.gfs_surface_forecast`, an optional NOAA GFS surface forecast extractor for station-nearest RH/T/DPT/etc. covariates, and the `nwp` optional dependency extra for `cfgrib`/`eccodes`.
- Added V3 diagnostic temperature and humidity oracle configs under `configs/v3/experiments/diagnostic/` for observed-target RMSE ceiling checks; they are marked backtest-only and non-operational.
- Added V2 `decoder_feature_baseline`, a config-driven baseline that copies configured future-known decoder covariates into the target for forecast-model baselines or explicit oracle/backtest ceiling checks.
- Added a local Seoul humidity backtest-only oracle config under `configs/v2/experiments/real/`; it is documented as non-canonical and `operational_valid: false` because the local raw ERA5 file lacks dew point and the derived decoder dew point is target-derived.
- Added the first V3 config, `configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml`, for temperature NWP-assisted MOS residual ridge experiments on the existing V2 CLI path.
- Added V3 humidity starter configs for direct-RH, dew-point, and dew-point-depression LightGBM tracks, writing to the V3 artifact root while reusing RH restoration where applicable.
- Added `docs/V3_V4_plan.md` to separate V3 station-level MOS work from V4 national/spatial/probabilistic forecasting work.
- Added V3 leaderboard metadata fields `model_family` and `backtest_only`, plus stable umbrella track leaderboards for observation-only and NWP-assisted experiments.
- Added `weather_korea_forecast.v2.ensemble`, an artifact-level prediction ensemble CLI that aligns component prediction files, writes evaluated ensemble artifacts, and updates V3 leaderboards.
- Added `weather_korea_forecast.dashboard.app`, a dependency-free local web dashboard for artifact/leaderboard prediction status.

## 2026-05-12

### Added
- Added V2 NWP-assisted/MOS metadata with `future_feature_metadata.json`, per-track leaderboards, and leaderboard columns for `uses_future_nwp_features`, `future_feature_source`, and `operational_valid`.
- Added `residual_from_feature` target transform for temperature MOS runs, enabling `observed_temp - future_era5_t2m_c` training with absolute-temperature restoration at evaluation and inference time.
- Added station-level ERA5 temperature-bias feature support (`obs_minus_era5_temp`) and residual ridge configs for `v2_temp_future_era5_residual_ridge_72to24` and `168to24`.
- Added fallback region/coastal/terrain metadata enrichment so missing `region_class` values do not collapse all stations into `unknown`.
- Added operational NWP-assisted inference support via `--future-weather-csv`, including issue-time selection, common GFS/NWP column aliases, wind derived features, and live reconstruction of decoder target lags from history.
- Added a forecast weather CSV template under `configs/v2/templates/`.

### Changed
- Future-feature metadata now honors an explicit `data.future_features.backtest_only` override so non-weather oracle diagnostics are not mislabeled as operational.
- V2 inference now refuses NWP-assisted decoder weather runs when future-valid weather covariates are missing for requested horizons instead of silently carrying forward the last encoder value.
- Temperature daily max/min/diurnal-range metrics are flattened into summary/leaderboard fields, including the weighted `daily_score`.
- Ridge closed-form solving now falls back to a jittered least-squares solve if a horizon-wise design matrix is singular.

## 2026-05-11

### Added
- Added humidity LightGBM research configs for fixed humidity features, 168->24 encoders, horizon-wise LightGBM, logit-RH, dew-point, dew-point-depression, extreme-weighted, and quantile-calibrated experiments.
- Added V2 humidity diagnostics: dry/humid event hit rates, low/high RH MAE, and target-neutral daily min/max/range reports.
- Added target-transform helpers for `logit_rh`, `dew_point`, and `dew_point_depression`, including RH restoration from temperature context.
- Added ERA5 dew-point unit sanity summaries and Kelvin-to-Celsius normalization for ERA5 temperature/dew-point inputs.

### Changed
- V2 daily and extreme plots/reports now use target-neutral names (`daily_target_errors.csv`, `metrics_daily_target.csv`, `extreme_target_scatter.png`) so humidity runs are not mislabeled as temperature reports.
- Humidity feature engineering can now include `is_daytime`, vapor-pressure/absolute-humidity features, ERA5 wind-speed/direction features, and optional predicted-temperature merge features.
- LightGBM baselines consume optional sequence sample weights for dry/humid extreme emphasis.

## 2026-05-10

### Added
- Added V2 `horizon_wise_ridge` with independent ridge heads, horizon-specific alpha search, and `horizon_model_metrics.csv`.
- Added explicit `horizon_wise_lightgbm` and a generic residual forecasting wrapper that trains `baseline + residual` models and writes component predictions.
- Added 168h encoder, horizon-wise ridge, horizon-wise LightGBM, residual LightGBM-on-ridge, station-wise scaling, and region-wise scaling temperature configs.
- Added V2 daily max/min temperature metrics, diurnal range metrics, horizon-station heatmap, station/region bar plots, extreme-temperature scatter, and worst-case summary artifacts.
- Added `docs/V2_STATUS.md` to record the current temperature baseline, diagnosis, experiment order, artifacts, and limitations.
- Added leakage-oriented synthetic tests for lag/rolling/delta features and train-only group scaling.

### Changed
- V2 temp ridge is documented as the official 72->24 temperature baseline and keeps alpha-grid selection aligned with guarded holdout bias correction.
- V2 bias correction supports additional grouped modes (`per_station`, `per_season_horizon`, `per_region_horizon`) and quantile correction candidates.
- V2 scaler supports `stationwise` and `regionwise` aliases and persists group statistics for station/region normalization.
- V2 leaderboard rows now include scaling mode, station count, train/val/test periods, raw/corrected metrics, and best/worst horizons.

### Known Limitations
- Residual forecasting does not yet build true `pytorch_forecasting` residual datasets; use fallback/baseline-style residual learners for now.

## 2026-03-26

### Added
- V2 single-target experiment pipeline under `src/weather_korea_forecast/v2/`.
- Direct multi-horizon dataset flow for station-level `24h` forecasting.
- Geographic metadata support with `lat`, `lon`, `elevation`, `coastal_distance_km`, and `region_class`.
- Stronger V2 baselines: `persistence`, `seasonal_persistence`, `ridge`, and optional `lightgbm`.
- V2 artifact and leaderboard management with config snapshot, scaler metadata, summaries, plots, and alias directories.
- V2 synthetic regression coverage in `tests/test_v2_pipeline.py`.
- Local real-data bootstrap configs under `configs/v2/experiments/real/` for Seoul `Q4 2024 -> Q1 2025` smoke runs.
- KMA station metadata downloader and multi-station raw-data bootstrap configs for V2.
- KMA station metadata downloader now supports a public DataWiki source so multi-station setup does not depend on API Hub auth.
- Multi-station ERA5 bootstrap now uses split `2024Q4` and `2025Q1` download configs plus a concat utility to stay under CDS cost limits.

### Changed
- README now documents V2 single-target temp/humidity strategy, artifact structure, raw vs corrected metrics, and real-data bootstrap execution.
- `docs/V2_plan.md` now records the implemented V2 scope, remaining gaps, and prioritized follow-up items.
- V2 evaluation now exports raw/corrected breakdowns, rolling-origin slice reports, worst-case samples, and additional plots.
- Humidity V2 configs now include dew-point-derived features, humidity-specific lag/rolling/delta features, and prediction clipping.
- V2 LightGBM prediction now keeps consistent feature names to avoid repeated sklearn warning noise during evaluation and inference.
- V2 leaderboard writing now also emits per-target leaderboard files such as `leaderboard_temp.csv` and `leaderboard_humidity.csv`.
- V2 multi-station LightGBM default configs were reduced to lighter tree settings so the direct 24h baseline completes in practical runtime on the current environment.
- V2 multi-station LightGBM configs now also set `n_jobs` explicitly for better runtime on CPU-heavy direct multi-horizon runs.
- The `fallback_torch` TFT substitute now supports residual forecasting from recent observed targets and applies configured gradient clipping during training.
- ASOS/AWS observation merging now supports `prefer_columns` so supplementary AWS variables can improve feature quality without replacing the primary ASOS target column.
- Forecast plots now use a headless matplotlib backend for stable training/evaluation tests in GUI-less environments.
- V2 bias correction now supports calibration/holdout selection so correction is applied only when it improves a held-out validation slice.
- V2 bias correction now supports guarded affine calibration, and ridge baselines can select `alpha` from an `alpha_grid` by validation loss.
- V2 default correction now uses guarded station+horizon mean-bias calibration after evaluation showed it lowered temp ridge RMSE more than the prior horizon-only correction.
- V2 temp ridge now supports bias-corrected holdout `alpha_grid` selection, and the default temp ridge config is strengthened to the best high-regularization alpha from the latest local sweep, lowering the temp benchmark RMSE further.

### Known Limitations
- The canonical multi-station V2 benchmark still depends on user-provided local ASOS/ERA5/metadata files.
- True-TFT-specific residual variants and true rolling-origin retraining remain follow-up items.

## 2026-05-25

### Added
- Added the V3 temperature MOS residual experiment matrix for 72h/168h ridge, horizon-wise ridge, LightGBM, horizon-wise LightGBM, optional CatBoost, and artifact-level ensemble specs.
- Added generic `nwp_temp_c` / `obs_minus_nwp_temp_*` MOS bias features with past-only lag/rolling construction for ERA5 backtests and future forecast-NWP adapters.
- Added V3 future-weather adapter entry point `load_future_weather_features(...)` with source options for ERA5 backtest, GFS, ECMWF, and KMA prepared forecast tables.
- Added residual MOS component columns to `predictions_test.csv` and additional V3 plots for residual, baseline-vs-final, worst-station, worst-horizon, and diurnal-range diagnostics.
- Added optional CatBoost baseline model routing for V3 MOS configs.

### Changed
- V3/V2 experiment summaries and leaderboards now include `track`, `uses_future_weather_features`, `leakage_risk_note`, and `worst_station_rmse` alongside existing operational-validity fields.
- Operational inference can now fail fast with `--operational` when a saved experiment depends on backtest-only ERA5/reanalysis future covariates.
- Station metadata loading now fills the V3 metadata schema (`station_name`, `region_class`, `terrain_class`, `coastal_class`, `urban_class`) to avoid collapsed `unknown` region reports for supported stations.

## 2026-05-25

### Added
- Added the unified reporting package `weather_korea_forecast.reporting` with CSV/JSON collection, standalone HTML dashboard generation, base64 plot embedding, best-model extraction, and failed/incomplete experiment reporting.
- Added default unified report refresh after V1/V2 train/evaluate runs so `reports/experiment_report.html` and `reports/experiment_summary.csv` are regenerated automatically; use `--no-update-report` only to skip it.
- Added V3 core runner scripts for temperature MOS, humidity experiments, and all-core sequential execution with per-run logs under `logs/v3_runs/`.
- Added LightGBM validation-only grid-search support with `lgbm_grid_search_results.csv` and `best_lgbm_params.json` artifacts.
- Added prepared forecast CSV validation/conversion support for operational future-weather features, including horizon/station completeness checks and Kelvin/Pa normalization.
- Added V3 real-run status and result-tracking docs plus reporting documentation.

### Changed
- V3/V2 leaderboard and experiment summary rows now include RMSE goal fields (`rmse_goal`, `rmse_goal_met`, `rmse_gap_to_goal`) and worst-horizon/station goal flags where available.
- Cached V2/V3 training tables are now rebuilt automatically when configured feature columns are missing, avoiding stale pre-V3 tables during real-data runs.
- Artifact-level ensembles can be invoked from an ensemble-only config through the V2 train CLI and can learn validation inverse-RMSE or horizon-wise weights when validation predictions are available.

### Verified
- `PYTHONPATH=src .venv312/Scripts/python.exe -m compileall -q src tests`
- `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q` after the reporting/V3 guard additions.

## G021 - Forecast Web Service MVP foundation

- Added production forecast artifact exporter and schema documentation.
- Added rule-based weather code and forecast confidence helpers.
- Added forecast API helpers with FastAPI-compatible app creation and no-dependency test fallback.
- Added static web MVP skeleton with operational warning banner.
- Documented real NWP archive requirements and site deployment plan.

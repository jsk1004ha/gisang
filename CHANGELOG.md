# Changelog

## 2026-03-26

### Added
- V2 single-target experiment pipeline under `src/weather_korea_forecast/v2/`.
- Direct multi-horizon dataset flow for station-level `24h` forecasting.
- Geographic metadata support with `lat`, `lon`, `elevation`, `coastal_distance_km`, and `region_class`.
- Stronger V2 baselines: `persistence`, `seasonal_persistence`, `ridge`, and optional `lightgbm`.
- V2 artifact and leaderboard management with config snapshot, scaler metadata, summaries, plots, and alias directories.
- V2 synthetic regression coverage in `tests/test_v2_pipeline.py`.
- Local real-data bootstrap configs under `configs/v2/experiments/real/` for Seoul `Q4 2024 -> Q1 2025` smoke runs.

### Changed
- README now documents V2 single-target temp/humidity strategy, artifact structure, raw vs corrected metrics, and real-data bootstrap execution.
- `docs/V2_plan.md` now records the implemented V2 scope, remaining gaps, and prioritized follow-up items.
- V2 evaluation now exports raw/corrected breakdowns, rolling-origin slice reports, worst-case samples, and additional plots.
- Humidity V2 configs now include dew-point-derived features, humidity-specific lag/rolling/delta features, and prediction clipping.
- V2 LightGBM prediction now keeps consistent feature names to avoid repeated sklearn warning noise during evaluation and inference.
- V2 leaderboard writing now also emits per-target leaderboard files such as `leaderboard_temp.csv` and `leaderboard_humidity.csv`.

### Known Limitations
- The canonical multi-station V2 benchmark still depends on user-provided local ASOS/ERA5/metadata files.
- Residual forecasting and true rolling-origin retraining remain follow-up items.

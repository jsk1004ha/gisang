# Forecast Output Schema

Production forecast artifacts are separate from research artifacts.

## Research artifacts

- `predictions_test.csv`
- `metrics_summary.json`
- `experiment_report.html`
- `training_history.json`

## Production artifacts

- `data/forecasts/runs/{forecast_run_id}/forecast_run.json`
- `data/forecasts/runs/{forecast_run_id}/forecast_points.json`
- `data/forecasts/runs/{forecast_run_id}/forecast_points.csv`
- `data/forecasts/latest/*`

## Schema version

The frozen G030 production manifest binds the site/export contract to `forecast_points.v2-beta-sources`.

`production_model_manifest.json` records:

- `forecast_schema_version: "forecast_points.v2-beta-sources"`
- target-specific model keys and artifact paths
- `temperature_status`, `humidity_status`, and `humidity_bias_status`
- `site_readiness`, `operational_valid`, and `site_caveats`
- beta labels required for temperature, humidity, and rule-based weather code

`production_model_freeze_record.json` stores the manifest checksum, model artifact checksums, and the source benchmark summary checksum. The benchmark checksum points to the frozen source benchmark summary, not the final summary that embeds the freeze record.

## `forecast_run.json`

`forecast_run.json` records:

- `forecast_run_id`
- `model_version`
- `forecast_init_time`
- `created_at`
- `source`
- `operational_valid`
- `backtest_only`
- `horizon_hours`
- `targets`
- `beta_targets`
- `notes`
- `warnings`

Warnings must surface when a run is not operational, when it is backtest-only, when humidity is beta, or when weather code is rule-based beta.

## `forecast_points.v2-beta-sources`

Each forecast point contains:

- station metadata: `station_id`, `station_name`, `lat`, `lon`, `region_class`
- time metadata: `valid_time`, `horizon_step`
- weather values: `temperature_c`, `humidity_percent`, `dew_point_c`, `feels_like_c`, `precip_probability`, `precip_mm`, `wind_speed_ms`, `wind_direction_deg`, `cloud_cover_percent`
- rule-based weather output: `weather_code`, `weather_label_ko`, `weather_icon`
- confidence/output metadata: `confidence`, `model_version`, `data_source`, `operational_valid`, `backtest_only`
- source/status fields for beta/direct variables:
  - `precip_probability_source`, `precip_probability_status`, `precip_probability_confidence`
  - `wind_source`, `wind_status`, `wind_confidence`
  - `cloud_source`, `cloud_status`, `cloud_confidence`
  - `weather_code_source`, `weather_code_status`, `weather_code_confidence`

Temperature uses `°C`; humidity uses `%p` in reports and percent values in forecast points. G031 must consume this schema without changing frozen model selection.

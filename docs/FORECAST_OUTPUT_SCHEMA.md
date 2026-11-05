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
- `data/forecasts/latest/current.json`
- `data/forecasts/latest/forecast_run.json`, `forecast_points.json`, `forecast_points.csv` (legacy compatibility copies)

### Publication and snapshot reads

Each completed `runs/{forecast_run_id}/` is immutable. Reusing an existing run ID is an error; automatically generated IDs include a random suffix so simultaneous exports do not share a directory. The exporter writes all run artifacts and the archive metadata before atomically replacing `latest/current.json` on the same filesystem:

```json
{"forecast_run_id": "forecast_20261105T000000Z_example"}
```

Pipeline invocations also use separate temporary work directories under `pipeline_work/`. Concurrent invocations targeting the same output directory cannot overwrite each other's raw/merged intermediate CSVs. These temporary files are removed after export or failure.

Read this pointer once, then read all required files from `runs/{forecast_run_id}/`. Python consumers can use `weather_korea_forecast.service.forecast_snapshot.resolve_forecast_dir(Path("data/forecasts/latest"))`. The API and static web fallback follow this contract. A failed run write or pointer replacement leaves the previous pointer and forecast available. Older run directories must remain available while readers may still reference them.

The flat files in `latest/` are retained for older/offline consumers and refreshed individually with atomic file replacement after the pointer commits. They are best-effort compatibility copies: multiple flat files are **not** one atomic snapshot, and concurrent publishers or a compatibility-copy failure can leave them stale or from different runs. Consumers requiring consistency must use `current.json` or the API. A compatibility-copy failure emits a warning without undoing the committed complete run. Legacy directories without a pointer remain readable; when the first pointer appears during a legacy read, the API/web reader switches to that complete run.

Atomic replacements preserve each existing destination file's permission mode. A new `current.json` inherits the run metadata file's mode; new compatibility files inherit their respective run files' modes. This preserves the configured POSIX read bits for an API/static-server account without changing the process umask or forcing broader permissions.

### Pipeline input provenance

`run_forecast_pipeline --operational` requires explicit `--predictions` from model inference as well as the existing trusted metadata gates. It never silently substitutes raw forecast values for model output.

In `--research` mode, omitting `--predictions` still permits direct CSV inspection. Temperature and humidity are labeled `nwp_direct` (or the explicit `gfs_direct`/`kma_direct` provider), with `direct` status and low target confidence. Missing values remain `unavailable`. This path uses `model_version=direct_forecast_csv`, forces `operational_valid=false` and `backtest_only=true`, and does not reuse the experiment's model RMSE. Explicitly supplied prediction values and source metadata retain their existing merge behavior.

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

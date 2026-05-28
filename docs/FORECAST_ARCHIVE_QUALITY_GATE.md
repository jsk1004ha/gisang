# Forecast Archive Quality Gate

`weather_korea_forecast.data.nwp_archive` merges prepared forecast CSVs and writes the archive quality report used by G022 operational training and HTML reports.

## Required core schema

```text
station_id
forecast_init_time
valid_time
horizon_step
source
issue_time
```

Source-specific weather columns are optional but should use canonical `nwp_*` names where possible.

## Report fields

`archive_quality_report.json` includes:

- `source`
- `station_count`
- `forecast_cycle_count`
- `min_forecast_init_time`
- `max_forecast_init_time`
- `horizon_min`, `horizon_max`
- `horizon_1_24_coverage`
- `missing_rate`
- `duplicate_count`
- `station_coverage_table`
- `horizon_coverage_table`
- `issue_time_coverage_table`
- `has_train_val_test_split`
- `forecast_archive_adequate`
- `blocking_reasons`
- `warnings`
- `archive_content_sha256` when an output archive CSV is written
- `expected_columns` and `expected_column_missing_rates` when operational feature columns are required

## Sanity checks

The gate records warnings and blocks adequacy for invalid humidity/dew-point/weather-code fields:

- `nwp_humidity` or `nwp_relative_humidity`: 0–100, with a tiny `1e-3` numerical tolerance for GRIB interpolation/packing noise
- dew point: plausible Celsius range and `dew_point <= temperature`, with a `0.1°C` numerical tolerance for near-saturated GRIB values
- precipitation probability: 0–100
- sky code: KMA valid values 1, 3, 4
- precipitation type: integer range 0–7

## Training behavior

G022 operational prepared-forecast training requires `forecast_archive_adequate=true`. The JSON value must be a strict boolean `true`; string values such as `"false"` are rejected instead of coerced. When `--archive-quality-report` is supplied with `--future-weather-csv`, the report must include `archive_content_sha256` matching the archive CSV bytes, so a stale quality report cannot green-light a different file.

Use `--expected-columns` on the archive builder when preparing a source for a specific model feature contract. Expected columns are reported with missing rates and block adequacy if more than 5% missing by default.

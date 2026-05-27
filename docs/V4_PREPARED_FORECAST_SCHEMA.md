# V4 Prepared Forecast CSV Schema

> This file is the explicit prepared-forecast schema entrypoint requested for V4-A. It mirrors `docs/V4_FORECAST_SCHEMA.md` so downstream agents can link either name.

# V4 Prepared Forecast CSV Schema

`prepared_forecast_csv` is the V4 station-level operational forecast input. It is valid only when each row is tied to a forecast issued at or before the prediction issue time.

## Required columns

| Column | Meaning |
| --- | --- |
| `station_id` | Station key matching observation and metadata tables. |
| `issue_time` | Forecast issue/runtime used for the prediction sample. |
| `valid_time` | Target valid timestamp. |
| `lead_hour` | Integer forecast lead from `issue_time` to `valid_time`. |
| `nwp_temp_2m_c` | 2m temperature in Celsius. |
| `nwp_relative_humidity_2m` | 2m relative humidity in percent. |
| `nwp_dew_point_2m_c` | 2m dew point in Celsius when available or derived from temp/RH. |
| `nwp_dew_point_depression` | `nwp_temp_2m_c - nwp_dew_point_2m_c`. |
| `nwp_surface_pressure` | Surface pressure in hPa. |
| `nwp_u10`, `nwp_v10` | 10m wind components. |
| `nwp_wind_speed` | 10m wind speed. |
| `nwp_total_precipitation` | Lead-aligned precipitation amount. |
| `source` | Forecast provider/model identifier, for example `gfs`, `ecmwf`, or `kma`. |

## Validation contract

A run may write `forecast_schema_valid=true` only after checking:

1. required columns are present;
2. timestamps parse as timezone-aware UTC or are normalized before join;
3. `valid_time >= issue_time` and `lead_hour` matches that interval;
4. no missing horizons in the configured prediction window for operational mode;
5. physical ranges are plausible (`0 <= RH <= 100`, pressure positive, precipitation non-negative);
6. rows do not use ERA5/reanalysis values as future forecasts.

## Reporting fields

Experiment summaries or copied `experiment_config.yaml` may expose:

```yaml
v4:
  stage: v4_operational_candidate
forecast_schema:
  version: v4-prepared-forecast-v1
  valid: true
```

The unified report propagates these to `experiment_summary.csv`, `best_models.csv`, and the HTML leaderboard/detail views.

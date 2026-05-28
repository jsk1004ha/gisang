# KMA Forecast Adapter

`weather_korea_forecast.data.kma_forecast_archive` converts KMA short-term forecast category rows into the prepared forecast archive schema.

## Input

Local CSV/API-shaped records must include KMA short-term forecast fields:

```text
baseDate/baseTime/fcstDate/fcstTime/nx/ny/category/fcstValue
```

Snake-case aliases such as `base_date` and `fcst_value` are also accepted. Station metadata must include `station_id` and either KMA grid coordinates (`grid_x/grid_y`, `nx/ny`, or `forecast_grid_x/forecast_grid_y`) or lat/lon plus a forecast grid table for nearest-grid mapping.

## Category mapping

| KMA category | Prepared column |
| --- | --- |
| TMP | nwp_t2m |
| REH | nwp_humidity |
| WSD | nwp_wind_speed |
| VEC | nwp_wind_direction |
| SKY | nwp_sky_code, nwp_cloud_cover |
| PTY | nwp_precip_type |
| POP | nwp_precip_probability |
| PCP | nwp_precip_amount |
| SNO | nwp_snow_amount |

KMA native `baseDate/baseTime` and `fcstDate/fcstTime` values are interpreted as Korea Standard Time (`Asia/Seoul`) and converted to UTC for `forecast_init_time` and `valid_time`; `horizon_step` is derived after conversion. PCP/SNO text values such as `강수없음`/`적설없음` map to `0.0`; numeric ranges such as `30.0~50.0mm` or `5.0~10.0cm` are converted to their midpoint.

## CLI

```bash
python -m weather_korea_forecast.data.kma_forecast_archive \
  --input data/raw/nwp/kma_forecast_raw.csv \
  --station-metadata data/raw/metadata/stations.csv \
  --output data/raw/nwp/archive/kma_prepared_forecast.csv \
  --mapping-output data/raw/nwp/archive/station_forecast_grid_mapping.csv
```

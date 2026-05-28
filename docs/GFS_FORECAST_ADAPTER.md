# GFS/NOMADS Forecast Adapter

`weather_korea_forecast.data.gfs_forecast_archive` converts local GFS/NOMADS extracted tables into the prepared forecast archive schema.

## Input modes

1. Station-level local CSV containing `station_id`, `forecast_init_time` or `issue_time`, `horizon_step` or `lead_hour`, and GFS/NWP variables.
2. Grid local CSV containing `lat/lon` instead of `station_id`; the adapter selects the nearest grid point for each station metadata row.

Accepted variable aliases include `gfs_temp_2m_c`, `t2m`, `gfs_surface_pressure`, `sp`, `gfs_u10`, `gfs_v10`, `gfs_total_precipitation`, `gfs_dew_point_2m_c`, and `gfs_cloud_cover`. Temperatures are converted from Kelvin when needed; pressure is converted from Pa to hPa when needed.

## NOMADS URL skeleton

The adapter exposes `gfs_nomads_filter_url(...)` for constructing a grib-filter URL. It is a download skeleton only; tests and normal conversion use local files to avoid hidden network or dependency requirements.

## CLI

```bash
python -m weather_korea_forecast.data.gfs_forecast_archive \
  --input data/raw/nwp/gfs_forecast_raw.csv \
  --station-metadata data/raw/metadata/stations.csv \
  --output data/raw/nwp/archive/gfs_prepared_forecast.csv
```

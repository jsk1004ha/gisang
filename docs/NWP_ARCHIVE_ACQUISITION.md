# G023 Real NWP Forecast Archive Acquisition

G023 turns real forecast-source exports into an operational-valid prepared forecast archive. The sprint is data-first: it does not claim RMSE improvement unless the archive quality gate passes.

## Workflow

```text
KMA/GFS/ECMWF raw or extracted local files
→ source adapter (`*_prepared_forecast.csv`)
→ archive builder (`prepared_forecast_archive.csv`)
→ quality report (`archive_quality_report.json/.md`)
→ G022 operational training only if adequate=true
```

Default artifact paths:

```text
data/raw/nwp/archive/kma_prepared_forecast.csv
data/raw/nwp/archive/gfs_prepared_forecast.csv
data/raw/nwp/archive/prepared_forecast_archive.csv
data/raw/nwp/archive/archive_quality_report.json
data/raw/nwp/archive/archive_quality_report.md
data/raw/nwp/archive/station_forecast_grid_mapping.csv
```

## Build commands

```bash
python -m weather_korea_forecast.data.kma_forecast_archive \
  --input data/raw/nwp/kma_forecast_raw.csv \
  --station-metadata data/raw/metadata/stations.csv \
  --output data/raw/nwp/archive/kma_prepared_forecast.csv

python -m weather_korea_forecast.data.gfs_forecast_archive \
  --input data/raw/nwp/gfs_forecast_raw.csv \
  --station-metadata data/raw/metadata/stations.csv \
  --output data/raw/nwp/archive/gfs_prepared_forecast.csv

python -m weather_korea_forecast.data.nwp_archive \
  --inputs data/raw/nwp/archive/kma_prepared_forecast.csv data/raw/nwp/archive/gfs_prepared_forecast.csv \
  --output data/raw/nwp/archive/prepared_forecast_archive.csv \
  --quality-report data/raw/nwp/archive/archive_quality_report.json \
  --expected-columns nwp_t2m,nwp_sp,nwp_u10,nwp_v10,nwp_tp,nwp_dew_point
```

## Operational training trigger

Only run operational G022 training after `forecast_archive_adequate=true`:

```bash
python -m weather_korea_forecast.v2.train \
  --config configs/g022/experiments/g022_temp_operational_residual_ridge_72to24.yaml \
  --future-weather-csv data/raw/nwp/archive/prepared_forecast_archive.csv \
  --archive-quality-report data/raw/nwp/archive/archive_quality_report.json
```

If the report is missing or inadequate, the train CLI raises a clear `ValueError` with blocking reasons.

## Adequacy target

The default minimum useful archive is:

- `station_count >= 20`
- `forecast_cycle_count >= 30`
- horizon 1–24 coverage `>= 0.95`
- missing rate `<= 0.05`
- train/val/test split possible
- 72h encoder + 24h prediction window possible
- source is not synthetic/smoke/generated/fixture/oracle/diagnostic
- humidity/dew-point/weather-code sanity checks pass
- all configured operational forecast feature columns exist and stay below the missing-rate threshold
- every station independently satisfies minimum cycle and horizon coverage thresholds

The report includes `archive_content_sha256`; the V2 train gate compares it with `--future-weather-csv` before model fitting.

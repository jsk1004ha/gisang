# G031 Website Finalization

G031 is site-only. It consumes the frozen G030 manifest and `forecast_points.v2-beta-sources` contract; it must not add training candidates, change model selection, or rerun model-improvement experiments.

## Frozen inputs

- Production manifest: `data/artifacts/g030_production_freeze_final/production_model_manifest.json`
- Freeze record: `data/artifacts/g030_production_freeze_final/production_model_freeze_record.json`
- Forecast schema: `forecast_points.v2-beta-sources`
- Temperature: frozen with `temperature_accuracy_caveat`
- Humidity: frozen as beta/NEAR_PASS
- Weather code: `rule_based_beta`

## API endpoints

The web MVP consumes these endpoints:

- `GET /api/forecast/latest`
- `GET /api/forecast/station/{station_id}`
- `GET /api/forecast/station/{station_id}/hourly`
- `GET /api/forecast/station/{station_id}/daily`
- `GET /api/model/status`

`/api/model/status` prefers the frozen G030 `production_model_manifest.json` when present and exposes temperature/humidity RMSE, status badges, benchmark reliability, site readiness, operational validity, caveats, beta label requirements, and the frozen schema version.

## UI coverage

The static `web/` MVP includes:

- region selector and station selector
- current forecast card with weather icon/label
- 24-hour temperature, humidity beta, and precipitation charts
- daily max/min temperature summary
- hourly forecast table with precipitation, wind, cloud, weather status, source, and confidence
- model status card with benchmark reliability, site readiness, V4-C status, humidity beta, and rule-based weather-code warning

## Required warnings

The site must show warnings for:

- `humidity_beta`
- `weather_code_rule_based_beta`
- `operational_valid=false` or non-PASS site readiness
- benchmark reliability below medium
- any manifest `site_caveats`

## Stop rule

G031 changes are limited to web/API/schema-consumption/docs. If model quality looks insufficient, the site must show caveats instead of reopening training.

## Freeze guard

Use the executable guard before treating post-G030 site work as complete:

```bash
python -m weather_korea_forecast.service.freeze_guard \
  --freeze-record data/artifacts/g030_production_freeze_final/production_model_freeze_record.json \
  --changed-files .omx/reports/g031-site-only-changed-files.txt \
  --json
```

The guard fails if G031 changes include training, model-selection, or experiment-candidate paths.

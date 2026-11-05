# Forecast API

Run with FastAPI if installed:

```bash
PYTHONPATH=src uvicorn weather_korea_forecast.api.main:app --reload
```

The module also provides pure-Python helpers for tests when FastAPI is not installed.

Endpoints:

- `GET /health`
- `GET /api/stations`
- `GET /api/forecast/latest`
- `GET /api/forecast/station/{station_id}`
- `GET /api/forecast/station/{station_id}/hourly`
- `GET /api/forecast/station/{station_id}/daily`
- `GET /api/model/status`
- `GET /api/evaluation/summary`

Responses include warnings when `operational_valid=false` or `backtest_only=true`.

Forecast endpoints resolve `data/forecasts/latest/current.json` once per request and read metadata and points from the same immutable `data/forecasts/runs/{forecast_run_id}/` directory. A concurrent publication may be visible to the next request but cannot mix two runs within one response. `create_app(forecast_dir=...)` still accepts a legacy directory with flat forecast files or an explicit run directory. If a legacy directory receives its first pointer during a read, the request re-reads the newly published complete run.

The static web fallback also resolves the pointer before fetching run metadata and points. When serving static artifacts, expose both `latest/current.json` and the referenced `runs/` directories; publish a complete run before replacing the pointer. See [Forecast Output Schema](FORECAST_OUTPUT_SCHEMA.md) for legacy-copy and retention requirements.

For G031, `/api/model/status` also reads the frozen G030 `production_model_manifest.json` when available and returns benchmark reliability, site readiness, temperature/humidity RMSE, target status, beta label requirements, rule-based weather-code status, manifest caveats, and the frozen `forecast_points.v2-beta-sources` schema version.

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

# Deployment Plan

## Scope

Deploy the G031 forecast web MVP using the frozen G030 production manifest. This plan intentionally excludes model improvement work.

## Inputs

- Frozen manifest: `data/artifacts/g030_production_freeze_final/production_model_manifest.json`
- Forecast schema: `forecast_points.v2-beta-sources`
- Forecast artifacts: `data/forecasts/latest/current.json` and the complete referenced `data/forecasts/runs/{forecast_run_id}/` directory
- Static frontend: `web/`
- API app: `weather_korea_forecast.api.main:create_app`

## Steps

1. Generate forecast artifacts using `weather_korea_forecast.service.export_forecast` or `weather_korea_forecast.service.run_forecast_pipeline`. Operational pipeline runs require `--predictions` from model inference. The exporter completes an immutable run before atomically replacing `latest/current.json`.
2. Keep generated forecast artifacts under `data/forecasts/` outside git.
3. Serve the API app and confirm:
   - `/api/forecast/latest`
   - `/api/forecast/station/{station_id}`
   - `/api/forecast/station/{station_id}/hourly`
   - `/api/forecast/station/{station_id}/daily`
   - `/api/model/status`
4. Serve `web/` as static files and set `window.API_BASE` if the API is on another origin. For static artifact fallback, serve the `runs/` directories together with `latest/current.json`. When copying to another host, upload the complete run first and replace the pointer last; retain older runs for in-flight readers. Do not cache the current pointer indefinitely.
5. Confirm the UI shows humidity beta, weather-code rule-based beta, benchmark reliability, operational validity, and site readiness warnings.

## Production guardrails

- Do not promote diagnostic, oracle, smoke, synthetic, fixture, or generated benchmark artifacts as official model evidence.
- Do not change frozen model selection during site deployment.
- If `site_readiness` is `WARN` or `FAIL`, display the caveat rather than hiding it.
- If live forecast source metadata is not trusted, export in research/backtest mode and show warnings.
- The flat `latest/forecast_*` files are best-effort compatibility copies and cannot provide a consistent multi-file snapshot during updates. Live consumers should use the API or resolve `current.json` once; see `docs/FORECAST_OUTPUT_SCHEMA.md`.

## Verification

Run:

```bash
python -m pytest tests/test_api_forecast.py tests/test_forecast_exporter.py tests/test_forecast_pipeline.py tests/test_forecast_snapshot.py tests/test_weather_code.py tests/test_web_mvp.py -q
python -m weather_korea_forecast.service.freeze_guard --freeze-record data/artifacts/g030_production_freeze_final/production_model_freeze_record.json --changed-files .omx/reports/g031-site-only-changed-files.txt --json
python -m pytest -q
```

Use `python -m compileall src tests` when dependency availability is limited.

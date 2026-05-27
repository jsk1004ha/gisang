# Deployment Plan

1. Generate forecast artifacts with `weather_korea_forecast.service.export_forecast`.
2. Serve API with `weather_korea_forecast.api.main`.
3. Serve `web/` as a static frontend and point it at the API base URL.
4. Keep production forecast artifacts under `data/forecasts/` outside git.
5. Block public operational mode until a trusted operational-valid model exists.

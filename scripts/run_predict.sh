#!/usr/bin/env bash
python -m weather_korea_forecast.inference.predict --experiment-dir data/artifacts/experiments/latest --station-id SEOUL --forecast-init-time 2025-01-03T00:00:00Z

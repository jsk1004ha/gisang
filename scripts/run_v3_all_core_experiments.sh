#!/usr/bin/env bash
set -u
bash scripts/run_v3_temp_mos_experiments.sh
bash scripts/run_v3_humidity_experiments.sh
PYTHONPATH=src "${PYTHON_BIN:-.venv312/Scripts/python.exe}" -m weather_korea_forecast.reporting.generate_report --experiments-root data/artifacts --output-dir reports --title "Gisang V1-V3 Experiment Report" || true

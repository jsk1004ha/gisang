#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

python -m weather_korea_forecast.data.download_kma_station_metadata --config configs/data/asos_station_metadata_v2.yaml
python -m weather_korea_forecast.data.download_kma_obs --config configs/data/asos_download_multistation_v2.yaml
python -m weather_korea_forecast.data.download_era5 --config configs/data/era5_download_multistation_v2_2024q4.yaml
python -m weather_korea_forecast.data.download_era5 --config configs/data/era5_download_multistation_v2_2025q1.yaml
python -m weather_korea_forecast.data.extract_era5_at_station --era5-path data/raw/era5/asos_multistation_v2_2024q4.nc --station-metadata-path data/raw/metadata/stations_v2.csv --output-path data/raw/era5/asos_multistation_v2_2024q4_station.csv --mode bilinear
python -m weather_korea_forecast.data.extract_era5_at_station --era5-path data/raw/era5/asos_multistation_v2_2025q1.nc --station-metadata-path data/raw/metadata/stations_v2.csv --output-path data/raw/era5/asos_multistation_v2_2025q1_station.csv --mode bilinear
python -m weather_korea_forecast.data.concat_tables --input-path data/raw/era5/asos_multistation_v2_2024q4_station.csv --input-path data/raw/era5/asos_multistation_v2_2025q1_station.csv --output-path data/raw/era5/asos_multistation_v2_station.csv --sort-column station_id --sort-column datetime

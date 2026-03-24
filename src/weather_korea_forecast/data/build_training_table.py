from __future__ import annotations

import argparse

import pandas as pd

from weather_korea_forecast.data.extract_era5_at_station import extract_era5_at_stations
from weather_korea_forecast.data.load_kma_asos import load_kma_asos
from weather_korea_forecast.data.station_metadata import load_station_metadata
from weather_korea_forecast.data.split_time_series import assign_time_splits
from weather_korea_forecast.features.time_features import add_time_features
from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.utils.io import read_table, write_table
from weather_korea_forecast.utils.logger import get_logger
from weather_korea_forecast.utils.paths import resolve_path

LOGGER = get_logger(__name__)


def build_training_table(config: dict) -> pd.DataFrame:
    obs_path = resolve_path(config["paths"]["observation_csv"])
    era5_path = resolve_path(config["paths"]["era5_csv"])
    metadata_path = resolve_path(config["paths"]["station_metadata_csv"])

    observations = load_kma_asos(
        obs_path,
        column_mapping=config.get("observation_columns"),
        source_tz=config.get("timezone", {}).get("source", "Asia/Seoul"),
    )
    station_metadata = load_station_metadata(metadata_path)
    era5_raw = read_table(era5_path)
    era5_features = extract_era5_at_stations(
        era5_df=era5_raw,
        station_df=station_metadata,
        mode=config.get("era5", {}).get("extraction_mode", "nearest"),
    )

    merged = observations.merge(era5_features, on=["station_id", "datetime"], how="left")
    merged = merged.merge(station_metadata, on="station_id", how="left")
    merged = add_time_features(merged)

    merged["target_temp"] = merged["temp"]
    merged["obs_temp"] = merged["temp"]
    if "humidity" in merged.columns:
        merged["target_humidity"] = merged["humidity"]
        merged["obs_humidity"] = merged["humidity"]
    if "pressure" in merged.columns:
        merged["obs_pressure"] = merged["pressure"]
    if "wind_speed" in merged.columns:
        merged["obs_wind_speed"] = merged["wind_speed"]

    merged = assign_time_splits(merged, config["split"])
    merged = merged.sort_values(["station_id", "datetime"]).reset_index(drop=True)
    merged["quality_flag"] = merged["quality_flag"].fillna("")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build long-form training table for weather forecasting.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    training_table = build_training_table(config)
    output_path = config["paths"]["output_training_table"]
    write_table(training_table, output_path)
    LOGGER.info("Wrote training table with %s rows to %s", len(training_table), output_path)


if __name__ == "__main__":
    main()

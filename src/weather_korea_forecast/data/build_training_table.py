from __future__ import annotations

import argparse

import pandas as pd

from weather_korea_forecast.data.extract_era5_at_station import extract_era5_at_stations
from weather_korea_forecast.data.load_kma_asos import load_kma_asos
from weather_korea_forecast.data.load_observations import load_observation_sources
from weather_korea_forecast.data.station_metadata import load_station_metadata
from weather_korea_forecast.data.split_time_series import assign_time_splits
from weather_korea_forecast.features.time_features import add_time_features
from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.utils.io import read_table, write_table
from weather_korea_forecast.utils.logger import get_logger
from weather_korea_forecast.utils.paths import resolve_path

LOGGER = get_logger(__name__)


def build_training_table(config: dict) -> pd.DataFrame:
    era5_path = resolve_path(config["paths"]["era5_csv"])
    metadata_path = resolve_path(config["paths"]["station_metadata_csv"])

    observations = _load_observations_from_config(config)
    station_metadata = load_station_metadata(metadata_path)
    era5_raw = read_table(era5_path)
    era5_features = extract_era5_at_stations(
        era5_df=era5_raw,
        station_df=station_metadata,
        mode=config.get("era5", {}).get("extraction_mode", "nearest"),
    )
    observations["station_id"] = observations["station_id"].astype(str)
    era5_features["station_id"] = era5_features["station_id"].astype(str)
    station_metadata["station_id"] = station_metadata["station_id"].astype(str)

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

    merged = _fill_continuous_feature_gaps(merged)
    merged = assign_time_splits(merged, config["split"])
    merged = merged.sort_values(["station_id", "datetime"]).reset_index(drop=True)
    merged["quality_flag"] = merged["quality_flag"].fillna("")
    return merged


def _load_observations_from_config(config: dict) -> pd.DataFrame:
    default_source_tz = config.get("timezone", {}).get("source", "Asia/Seoul")
    observations_config = config.get("observations", {})
    sources = observations_config.get("sources") or _legacy_observation_sources(config)
    if len(sources) == 1 and sources[0].get("kind", "asos") == "asos" and not sources[0].get("resample_rule"):
        return load_kma_asos(
            sources[0]["path"],
            column_mapping=sources[0].get("column_mapping"),
            source_tz=sources[0].get("source_tz", default_source_tz),
        )
    return load_observation_sources(
        sources=sources,
        default_source_tz=default_source_tz,
        merge_strategy=observations_config.get("merge_strategy", "priority"),
    )


def _legacy_observation_sources(config: dict) -> list[dict]:
    paths = config["paths"]
    default_source_tz = config.get("timezone", {}).get("source", "Asia/Seoul")
    sources = [
        {
            "name": "asos",
            "kind": "asos",
            "path": paths["observation_csv"],
            "column_mapping": config.get("observation_columns"),
            "source_tz": default_source_tz,
            "priority": 0,
        }
    ]
    aws_path = paths.get("aws_observation_csv")
    if aws_path:
        aws_config = config.get("aws", {})
        sources.append(
            {
                "name": aws_config.get("name", "aws"),
                "kind": "aws",
                "path": aws_path,
                "column_mapping": config.get("aws_observation_columns", config.get("observation_columns")),
                "source_tz": aws_config.get("source_tz", default_source_tz),
                "priority": int(aws_config.get("priority", 1)),
                "resample_rule": aws_config.get("resample_rule"),
                "aggregation": aws_config.get("aggregation"),
                "station_id": aws_config.get("station_id"),
            }
        )
    return sources


def _fill_continuous_feature_gaps(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    candidate_columns = [column for column in frame.columns if column.startswith("era5_") or column.startswith("obs_")]
    for column in candidate_columns:
        frame[column] = (
            frame.groupby("station_id")[column]
            .transform(lambda series: series.astype(float).interpolate(limit=12, limit_direction="both").ffill().bfill())
        )
    return frame


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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from weather_korea_forecast.data.extract_era5_at_station import extract_era5_at_stations
from weather_korea_forecast.data.load_kma_asos import load_kma_asos
from weather_korea_forecast.data.load_observations import load_observation_sources
from weather_korea_forecast.data.split_time_series import assign_time_splits
from weather_korea_forecast.data.station_metadata import load_station_metadata
from weather_korea_forecast.features.time_features import add_time_features
from weather_korea_forecast.utils.io import read_table, write_json, write_table
from weather_korea_forecast.utils.paths import resolve_path


def build_v2_training_table(config: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    paths = config["paths"]
    data_config = config["data"]
    target_name = str(data_config["target_name"])

    observations = _load_observations_from_config(config)
    station_metadata = load_station_metadata(paths["station_metadata_csv"])
    era5_raw = read_table(paths["era5_csv"])
    era5_features = extract_era5_at_stations(
        era5_df=era5_raw,
        station_df=station_metadata,
        mode=data_config.get("era5", {}).get("extraction_mode", "nearest"),
    )

    observations["station_id"] = observations["station_id"].astype(str)
    station_metadata["station_id"] = station_metadata["station_id"].astype(str)
    era5_features["station_id"] = era5_features["station_id"].astype(str)

    merged = observations.merge(era5_features, on=["station_id", "datetime"], how="left")
    merged = merged.merge(station_metadata, on="station_id", how="left")
    merged = add_time_features(merged)
    merged = _ensure_region_columns(merged)
    merged = _add_observation_aliases(merged)
    merged["target_value"] = merged[target_name].astype(float)
    merged["target_name"] = target_name
    merged = _fill_raw_continuous_columns(merged, config)
    merged = _apply_feature_engineering(merged, config)
    merged = assign_time_splits(merged, data_config["split"])
    merged = merged.sort_values(["station_id", "datetime"]).reset_index(drop=True)
    merged["quality_flag"] = merged.get("quality_flag", "").fillna("")

    quality_report = summarize_time_index_quality(merged)
    return merged, quality_report


def prepare_v2_data(config: dict) -> tuple[Path, Path | None]:
    training_table, quality_report = build_v2_training_table(config)
    output_path = write_table(training_table, config["paths"]["output_training_table"])
    quality_path = None
    if "output_data_quality" in config["paths"]:
        quality_path = write_json(quality_report, config["paths"]["output_data_quality"])
    return output_path, quality_path


def summarize_time_index_quality(frame: pd.DataFrame) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for station_id, group in frame.sort_values(["station_id", "datetime"]).groupby("station_id"):
        datetimes = pd.to_datetime(group["datetime"], utc=True).drop_duplicates().sort_values()
        diffs = datetimes.diff().dropna()
        missing_hours = int(sum(max(int(diff / pd.Timedelta(hours=1)) - 1, 0) for diff in diffs))
        duplicate_count = int(group.duplicated(subset=["station_id", "datetime"]).sum())
        rows.append(
            {
                "station_id": str(station_id),
                "start": str(datetimes.min()) if not datetimes.empty else None,
                "end": str(datetimes.max()) if not datetimes.empty else None,
                "row_count": int(len(group)),
                "timestamp_count": int(datetimes.nunique()),
                "duplicate_count": duplicate_count,
                "missing_hour_count": missing_hours,
            }
        )
    return {
        "station_count": int(frame["station_id"].nunique()) if "station_id" in frame.columns else 0,
        "stations": rows,
    }


def _load_observations_from_config(config: dict) -> pd.DataFrame:
    paths = config["paths"]
    data_config = config["data"]
    default_source_tz = data_config.get("timezone", {}).get("source", "Asia/Seoul")
    observations_config = data_config.get("observations", {})
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
    data_config = config["data"]
    default_source_tz = data_config.get("timezone", {}).get("source", "Asia/Seoul")
    sources = [
        {
            "name": "asos",
            "kind": "asos",
            "path": paths["observation_csv"],
            "column_mapping": data_config.get("observation_columns"),
            "source_tz": default_source_tz,
            "priority": 0,
        }
    ]
    aws_path = paths.get("aws_observation_csv")
    if aws_path:
        aws_config = data_config.get("aws", {})
        sources.append(
            {
                "name": aws_config.get("name", "aws"),
                "kind": "aws",
                "path": aws_path,
                "column_mapping": data_config.get("aws_observation_columns", data_config.get("observation_columns")),
                "source_tz": aws_config.get("source_tz", default_source_tz),
                "priority": int(aws_config.get("priority", 1)),
                "resample_rule": aws_config.get("resample_rule"),
                "aggregation": aws_config.get("aggregation"),
                "station_id": aws_config.get("station_id"),
            }
        )
    return sources


def _ensure_region_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "region_class" not in enriched.columns:
        if "region" in enriched.columns:
            enriched["region_class"] = enriched["region"]
        else:
            enriched["region_class"] = "unknown"
    if "region" not in enriched.columns:
        enriched["region"] = enriched["region_class"]
    return enriched


def _add_observation_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    for source_column in ("temp", "humidity", "pressure", "wind_speed", "precipitation"):
        if source_column in enriched.columns:
            enriched[f"obs_{source_column}"] = enriched[source_column]
    return enriched


def _fill_raw_continuous_columns(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    enriched = frame.copy()
    data_config = config["data"]
    interpolate_limit = int(data_config.get("cleaning", {}).get("interpolate_limit_hours", 12))
    numeric_columns = [
        column
        for column in enriched.columns
        if column.startswith("obs_") or column.startswith("era5_") or column in {"target_value", "coastal_distance_km"}
    ]
    static_numeric_columns = data_config.get("features", {}).get("static_real", [])
    numeric_columns.extend([column for column in static_numeric_columns if column in enriched.columns])
    numeric_columns = sorted(set(numeric_columns))

    for column in numeric_columns:
        enriched[column] = (
            enriched.groupby("station_id")[column]
            .transform(
                lambda series: series.astype(float)
                .interpolate(limit=interpolate_limit, limit_direction="both")
                .ffill()
                .bfill()
            )
        )
    return enriched


def _apply_feature_engineering(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    enriched = frame.sort_values(["station_id", "datetime"]).copy()
    feature_config = config["data"].get("feature_engineering", {})
    lag_features = feature_config.get(
        "lag_features",
        {
            "target_value": [1, 3, 6, 12, 24, 48, 72],
            "obs_pressure": [1, 6, 24],
            "obs_wind_speed": [1, 6, 24],
            "era5_t2m": [1, 3, 6, 12, 24],
            "era5_sp": [1, 6, 24],
        },
    )
    rolling_features = feature_config.get("rolling_features", {"target_value": [3, 6, 12, 24]})
    delta_features = feature_config.get("delta_features", {"target_value": [1, 6, 24]})

    grouped = enriched.groupby("station_id", group_keys=False)
    for column, lags in lag_features.items():
        if column not in enriched.columns:
            continue
        for lag in sorted({int(value) for value in lags}):
            enriched[f"{column}_lag_{lag}"] = grouped[column].shift(lag)

    for column, windows in rolling_features.items():
        if column not in enriched.columns:
            continue
        shifted = grouped[column].shift(1)
        for window in sorted({int(value) for value in windows}):
            enriched[f"{column}_roll_mean_{window}"] = shifted.groupby(enriched["station_id"]).transform(lambda series: series.rolling(window).mean())
            enriched[f"{column}_roll_std_{window}"] = shifted.groupby(enriched["station_id"]).transform(
                lambda series: series.rolling(window).std(ddof=0)
            )

    for column, periods in delta_features.items():
        if column not in enriched.columns:
            continue
        for period in sorted({int(value) for value in periods}):
            enriched[f"{column}_delta_{period}"] = grouped[column].transform(lambda series: series - series.shift(period))

    if "target_value_lag_24" in enriched.columns:
        enriched["target_value_same_hour_prev_day"] = enriched["target_value_lag_24"]
    return enriched


def load_or_prepare_v2_training_table(config: dict) -> pd.DataFrame:
    output_path = resolve_path(config["paths"]["output_training_table"])
    if output_path.exists():
        return read_table(output_path)
    training_table, quality_report = build_v2_training_table(config)
    write_table(training_table, output_path)
    if "output_data_quality" in config["paths"]:
        write_json(quality_report, config["paths"]["output_data_quality"])
    return training_table

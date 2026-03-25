from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from weather_korea_forecast.data.align_time_index import align_dataframe_timezone
from weather_korea_forecast.utils.paths import resolve_path

STANDARD_COLUMNS = {
    "station_id": "station_id",
    "datetime": "datetime",
    "temp": "temp",
    "humidity": "humidity",
    "pressure": "pressure",
    "wind_speed": "wind_speed",
    "precipitation": "precipitation",
    "quality_flag": "quality_flag",
}

DEFAULT_RESAMPLE_AGGREGATION = {
    "temp": "mean",
    "humidity": "mean",
    "pressure": "mean",
    "wind_speed": "mean",
    "precipitation": "sum",
    "quality_flag": "last",
}


def load_observation_table(
    path: str | Path,
    column_mapping: dict[str, str] | None = None,
    source_tz: str = "Asia/Seoul",
    station_id: str | None = None,
    resample_rule: str | None = None,
    aggregation: dict[str, str] | None = None,
    source_name: str | None = None,
) -> pd.DataFrame:
    mapping = column_mapping or STANDARD_COLUMNS
    raw = pd.read_csv(resolve_path(path))
    renamed = raw.rename(columns={value: key for key, value in mapping.items()})
    if station_id is not None:
        renamed["station_id"] = station_id
    missing = {"station_id", "datetime", "temp"} - set(renamed.columns)
    if missing:
        raise ValueError(f"Missing required observation columns: {sorted(missing)}")
    for column in STANDARD_COLUMNS:
        if column not in renamed.columns:
            renamed[column] = pd.NA

    standardized = renamed[list(STANDARD_COLUMNS)]
    standardized = align_dataframe_timezone(standardized, source_tz=source_tz)
    standardized = standardized.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    if resample_rule:
        standardized = _resample_observations(standardized, rule=resample_rule, aggregation=aggregation or DEFAULT_RESAMPLE_AGGREGATION)

    standardized["station_id"] = standardized["station_id"].astype(str)
    standardized["quality_flag"] = standardized["quality_flag"].fillna("")
    if source_name:
        standardized["observation_source"] = source_name
    return standardized


def load_observation_sources(
    sources: list[dict[str, Any]],
    default_source_tz: str = "Asia/Seoul",
    merge_strategy: str = "priority",
) -> pd.DataFrame:
    if not sources:
        raise ValueError("At least one observation source must be configured.")
    if merge_strategy != "priority":
        raise ValueError(f"Unsupported observation merge strategy: {merge_strategy}")

    frames: list[pd.DataFrame] = []
    for index, source in enumerate(sources):
        path = source.get("path") or source.get("csv")
        if not path:
            raise ValueError("Observation source is missing 'path'.")
        name = str(source.get("name") or source.get("kind") or Path(path).stem)
        frame = load_observation_table(
            path=path,
            column_mapping=source.get("column_mapping"),
            source_tz=source.get("source_tz", default_source_tz),
            station_id=source.get("station_id"),
            resample_rule=source.get("resample_rule"),
            aggregation=source.get("aggregation"),
            source_name=name,
        ).copy()
        frame["_priority"] = int(source.get("priority", index))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return _merge_observation_frames(combined)


def _merge_observation_frames(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["station_id", "datetime"]
    ordered = frame.sort_values(keys + ["_priority"]).reset_index(drop=True)
    value_columns = [column for column in ordered.columns if column not in keys + ["_priority"]]

    merged_rows: list[dict[str, Any]] = []
    for (station_id, timestamp), group in ordered.groupby(keys, sort=True):
        row: dict[str, Any] = {"station_id": station_id, "datetime": timestamp}
        for column in value_columns:
            if column == "observation_source":
                row[column] = _first_non_empty_string(group[column])
            else:
                row[column] = _first_non_null(group[column])
        if "observation_source" in group.columns:
            unique_sources = [str(value) for value in pd.unique(group["observation_source"]) if pd.notna(value)]
            row["observation_sources"] = ",".join(unique_sources)
        merged_rows.append(row)

    merged = pd.DataFrame(merged_rows)
    merged = merged.sort_values(keys).reset_index(drop=True)
    if "quality_flag" in merged.columns:
        merged["quality_flag"] = merged["quality_flag"].fillna("")
    return merged


def _resample_observations(
    frame: pd.DataFrame,
    rule: str,
    aggregation: dict[str, str],
) -> pd.DataFrame:
    agg_map = {column: aggregation.get(column, DEFAULT_RESAMPLE_AGGREGATION.get(column, "last")) for column in STANDARD_COLUMNS if column not in {"station_id", "datetime"}}
    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    for station_id, group in frame.groupby("station_id", sort=False):
        resampled = (
            group.set_index("datetime")[list(agg_map)]
            .resample(rule)
            .agg(agg_map)
            .dropna(subset=["temp"], how="all")
            .reset_index()
        )
        resampled["station_id"] = station_id
        parts.append(resampled)

    if not parts:
        return frame.iloc[0:0].copy()

    result = pd.concat(parts, ignore_index=True)
    return result[["station_id", "datetime", *[column for column in STANDARD_COLUMNS if column not in {"station_id", "datetime"}]]]


def _first_non_null(series: pd.Series) -> Any:
    non_null = series[series.notna()]
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def _first_non_empty_string(series: pd.Series) -> str:
    non_null = [str(value) for value in series if pd.notna(value) and str(value).strip()]
    if non_null:
        return non_null[0]
    return ""

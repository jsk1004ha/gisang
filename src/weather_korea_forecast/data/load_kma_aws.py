from __future__ import annotations

from pathlib import Path

import pandas as pd

from weather_korea_forecast.data.align_time_index import align_dataframe_timezone

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


def load_kma_aws(
    path: str | Path,
    column_mapping: dict[str, str] | None = None,
    source_tz: str = "Asia/Seoul",
) -> pd.DataFrame:
    mapping = column_mapping or STANDARD_COLUMNS
    raw = pd.read_csv(path)
    renamed = raw.rename(columns={value: key for key, value in mapping.items()})
    missing = {"station_id", "datetime", "temp"} - set(renamed.columns)
    if missing:
        raise ValueError(f"Missing required observation columns: {sorted(missing)}")
    for column in STANDARD_COLUMNS:
        if column not in renamed.columns:
            renamed[column] = pd.NA
    renamed = align_dataframe_timezone(renamed, source_tz=source_tz)
    renamed = renamed.sort_values(["station_id", "datetime"]).reset_index(drop=True)
    return renamed[list(STANDARD_COLUMNS)]

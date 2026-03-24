from __future__ import annotations

import pandas as pd


def align_dataframe_timezone(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    source_tz: str | None = "Asia/Seoul",
) -> pd.DataFrame:
    frame = df.copy()
    timestamps = pd.to_datetime(frame[datetime_col])
    if timestamps.dt.tz is None:
        if source_tz is None:
            timestamps = timestamps.dt.tz_localize("UTC")
        else:
            timestamps = timestamps.dt.tz_localize(source_tz)
    frame[datetime_col] = timestamps.dt.tz_convert("UTC")
    return frame


def convert_utc_to_kst(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, utc=True)
    return timestamps.dt.tz_convert("Asia/Seoul")


def assert_regular_hourly_index(
    df: pd.DataFrame,
    group_col: str = "station_id",
    datetime_col: str = "datetime",
) -> None:
    for _, group in df.sort_values([group_col, datetime_col]).groupby(group_col):
        diffs = pd.to_datetime(group[datetime_col], utc=True).diff().dropna()
        if not diffs.empty and not (diffs == pd.Timedelta(hours=1)).all():
            raise ValueError("Detected non-hourly gaps in aligned data.")

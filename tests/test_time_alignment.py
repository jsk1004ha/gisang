from __future__ import annotations

import pandas as pd

from weather_korea_forecast.data.align_time_index import align_dataframe_timezone, assert_regular_hourly_index


def test_align_dataframe_timezone_localizes_to_utc() -> None:
    frame = pd.DataFrame({"station_id": ["A", "A"], "datetime": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"]})
    aligned = align_dataframe_timezone(frame, source_tz="Asia/Seoul")
    assert str(aligned.loc[0, "datetime"]) == "2023-12-31 15:00:00+00:00"


def test_assert_regular_hourly_index_passes_for_hourly_series() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["A", "A", "A"],
            "datetime": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"]),
        }
    )
    assert_regular_hourly_index(frame)

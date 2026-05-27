from __future__ import annotations

import pandas as pd

from weather_korea_forecast.data.humidity_sanity import (
    clip_relative_humidity,
    normalize_humidity_features,
    restore_relative_humidity_from_depression,
    restore_relative_humidity_from_dew_point,
)


def test_humidity_sanity_converts_kelvin_dewpoint_and_adds_time_features() -> None:
    frame = pd.DataFrame(
        {
            "valid_time": ["2026-01-01T03:00:00Z"],
            "era5_t2m": [283.15],
            "era5_dew_point_c": [278.15],
            "humidity_percent": [120.0],
        }
    )

    output = normalize_humidity_features(frame)

    assert output.loc[0, "era5_t2m_c"] == 10.0
    assert output.loc[0, "era5_dew_point_c"] == 5.0
    assert output.loc[0, "era5_dew_point_depression"] == 5.0
    assert output.loc[0, "humidity_percent"] == 100.0
    assert {"hour_sin", "hour_cos", "doy_sin", "doy_cos"}.issubset(output.columns)


def test_restore_relative_humidity_and_clip() -> None:
    rh = restore_relative_humidity_from_dew_point(pd.Series([20.0]), pd.Series([20.0]))
    assert round(float(rh.iloc[0]), 6) == 100.0
    depressed = restore_relative_humidity_from_depression(pd.Series([20.0]), pd.Series([10.0]))
    assert 0.0 < float(depressed.iloc[0]) < 100.0
    assert list(clip_relative_humidity(pd.Series([-5.0, 50.0, 120.0]))) == [0.0, 50.0, 100.0]

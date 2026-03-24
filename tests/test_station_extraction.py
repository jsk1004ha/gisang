from __future__ import annotations

import pandas as pd

from weather_korea_forecast.data.extract_era5_at_station import extract_era5_at_stations


def test_extract_era5_nearest_and_bilinear() -> None:
    era5 = pd.DataFrame(
        {
            "datetime": ["2024-01-01T00:00:00Z"] * 4,
            "lat": [0.0, 0.0, 1.0, 1.0],
            "lon": [0.0, 1.0, 0.0, 1.0],
            "era5_t2m": [10.0, 12.0, 14.0, 16.0],
        }
    )
    stations = pd.DataFrame({"station_id": ["S1"], "lat": [0.5], "lon": [0.5], "elevation": [1.0]})

    nearest = extract_era5_at_stations(era5, stations, mode="nearest")
    bilinear = extract_era5_at_stations(era5, stations, mode="bilinear")

    assert nearest.loc[0, "station_id"] == "S1"
    assert bilinear.loc[0, "era5_t2m"] == 13.0

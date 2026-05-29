from __future__ import annotations

import pandas as pd

from weather_korea_forecast.data.gfs_forecast_archive import build_gfs_prepared_forecast, gfs_nomads_filter_url
from weather_korea_forecast.data.gfs_surface_forecast import MESSAGE_SPECS, _append_nwp_aliases


def test_gfs_local_table_maps_to_prepared_schema_and_nearest_station_grid() -> None:
    raw = pd.DataFrame(
        {
            "forecast_init_time": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "horizon_step": [1, 1],
            "lat": [37.0, 37.6],
            "lon": [126.0, 127.0],
            "gfs_temp_2m_c": [1.0, 2.0],
            "gfs_surface_pressure": [101325.0, 101000.0],
            "gfs_u10": [1.0, 3.0],
            "gfs_v10": [2.0, 4.0],
            "gfs_total_precipitation": [0.1, 0.2],
            "gfs_dew_point_2m_c": [-1.0, 0.0],
            "gfs_cloud_cover": [20.0, 40.0],
        }
    )
    stations = pd.DataFrame({"station_id": ["108"], "lat": [37.55], "lon": [126.98]})

    prepared = build_gfs_prepared_forecast(raw, stations)

    assert prepared.shape[0] == 1
    row = prepared.iloc[0]
    assert row["station_id"] == "108"
    assert row["valid_time"].isoformat() == "2026-01-01T01:00:00+00:00"
    assert row["nwp_t2m"] == 2.0
    assert row["nwp_sp"] == 1010.0
    assert row["nwp_u10"] == 3.0
    assert row["nwp_v10"] == 4.0
    assert row["nwp_tp"] == 0.2
    assert row["nwp_dew_point"] == 0.0
    assert row["nwp_cloud_cover"] == 40.0
    assert row["source"] == "gfs_forecast"


def test_gfs_nomads_filter_url_documents_http_download_skeleton() -> None:
    url = gfs_nomads_filter_url("20260101", "00", 24, variables=["TMP", "UGRD"], leftlon=124, rightlon=132, toplat=39, bottomlat=33)

    assert "filter_gfs_0p25.pl" in url
    assert "file=gfs.t00z.pgrb2.0p25.f024" in url
    assert "var_TMP=on" in url
    assert "var_UGRD=on" in url
    assert "leftlon=124" in url


def test_gfs_surface_forecast_supports_full_variable_aliases() -> None:
    assert MESSAGE_SPECS["gust"][:2] == ("GUST", "surface")
    assert MESSAGE_SPECS["spfh2m"][:2] == ("SPFH", "2 m above ground")
    assert MESSAGE_SPECS["pwat"][:2] == ("PWAT", "entire atmosphere (considered as a single layer)")
    row: dict[str, object] = {
        "gfs_gust": 12.0,
        "gfs_total_cloud_cover": 80.0,
        "gfs_low_cloud_cover": 20.0,
        "gfs_specific_humidity_2m": 0.006,
        "gfs_precipitable_water": 18.0,
        "gfs_mean_sea_level_pressure": 1013.2,
        "gfs_u10": 3.0,
        "gfs_v10": 4.0,
    }

    _append_nwp_aliases(row)

    assert row["nwp_gust"] == 12.0
    assert row["nwp_cloud_cover"] == 80.0
    assert row["nwp_low_cloud_cover"] == 20.0
    assert row["nwp_specific_humidity"] == 0.006
    assert row["nwp_pwat"] == 18.0
    assert row["nwp_mslp"] == 1013.2
    assert row["nwp_wind_speed"] == 5.0

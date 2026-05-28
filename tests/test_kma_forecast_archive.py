from __future__ import annotations

import pandas as pd

from weather_korea_forecast.data.kma_forecast_archive import build_kma_prepared_forecast, build_station_grid_mapping


def test_kma_category_mapping_station_grid_and_forecast_times(tmp_path) -> None:
    raw = pd.DataFrame(
        [
            {"baseDate": "20260101", "baseTime": "0500", "fcstDate": "20260101", "fcstTime": "0600", "nx": 60, "ny": 127, "category": "TMP", "fcstValue": "3"},
            {"baseDate": "20260101", "baseTime": "0500", "fcstDate": "20260101", "fcstTime": "0600", "nx": 60, "ny": 127, "category": "REH", "fcstValue": "70"},
            {"baseDate": "20260101", "baseTime": "0500", "fcstDate": "20260101", "fcstTime": "0600", "nx": 60, "ny": 127, "category": "WSD", "fcstValue": "2.5"},
            {"baseDate": "20260101", "baseTime": "0500", "fcstDate": "20260101", "fcstTime": "0600", "nx": 60, "ny": 127, "category": "VEC", "fcstValue": "270"},
            {"baseDate": "20260101", "baseTime": "0500", "fcstDate": "20260101", "fcstTime": "0600", "nx": 60, "ny": 127, "category": "SKY", "fcstValue": "3"},
            {"baseDate": "20260101", "baseTime": "0500", "fcstDate": "20260101", "fcstTime": "0600", "nx": 60, "ny": 127, "category": "PTY", "fcstValue": "1"},
            {"baseDate": "20260101", "baseTime": "0500", "fcstDate": "20260101", "fcstTime": "0600", "nx": 60, "ny": 127, "category": "POP", "fcstValue": "30"},
            {"baseDate": "20260101", "baseTime": "0500", "fcstDate": "20260101", "fcstTime": "0600", "nx": 60, "ny": 127, "category": "PCP", "fcstValue": "강수없음"},
            {"baseDate": "20260101", "baseTime": "0500", "fcstDate": "20260101", "fcstTime": "0600", "nx": 60, "ny": 127, "category": "SNO", "fcstValue": "적설없음"},
        ]
    )
    stations = pd.DataFrame({"station_id": ["108"], "lat": [37.57], "lon": [126.98], "grid_x": [60], "grid_y": [127]})
    mapping_path = tmp_path / "station_forecast_grid_mapping.csv"

    prepared = build_kma_prepared_forecast(raw, stations, mapping_output=mapping_path)

    assert prepared.shape[0] == 1
    row = prepared.iloc[0]
    assert row["station_id"] == "108"
    assert row["forecast_init_time"].isoformat() == "2025-12-31T20:00:00+00:00"
    assert row["valid_time"].isoformat() == "2025-12-31T21:00:00+00:00"
    assert int(row["horizon_step"]) == 1
    assert row["nwp_t2m"] == 3.0
    assert row["nwp_humidity"] == 70.0
    assert row["nwp_wind_speed"] == 2.5
    assert row["nwp_wind_direction"] == 270.0
    assert row["nwp_sky_code"] == 3
    assert row["nwp_cloud_cover"] == 75.0
    assert row["nwp_precip_type"] == 1
    assert row["nwp_precip_probability"] == 30.0
    assert row["nwp_precip_amount"] == 0.0
    assert row["nwp_snow_amount"] == 0.0
    assert row["source"] == "kma_forecast"
    mapping = pd.read_csv(mapping_path)
    assert mapping[["station_id", "forecast_grid_x", "forecast_grid_y"]].to_dict("records") == [
        {"station_id": 108, "forecast_grid_x": 60, "forecast_grid_y": 127}
    ]


def test_kma_station_grid_mapping_uses_nearest_available_grid_when_station_has_no_grid() -> None:
    stations = pd.DataFrame({"station_id": ["108"], "lat": [37.55], "lon": [127.02]})
    forecast_grid = pd.DataFrame({"nx": [55, 60], "ny": [125, 127], "lat": [36.0, 37.56], "lon": [126.0, 127.0]})

    mapping = build_station_grid_mapping(stations, forecast_grid=forecast_grid)

    assert mapping.iloc[0]["forecast_grid_x"] == 60
    assert mapping.iloc[0]["forecast_grid_y"] == 127


def test_kma_precipitation_and_snow_ranges_parse_to_midpoint() -> None:
    assert build_kma_prepared_forecast(
        pd.DataFrame(
            [
                {"baseDate": "20260101", "baseTime": "0500", "fcstDate": "20260101", "fcstTime": "0600", "nx": 60, "ny": 127, "category": "PCP", "fcstValue": "30.0~50.0mm"},
                {"baseDate": "20260101", "baseTime": "0500", "fcstDate": "20260101", "fcstTime": "0600", "nx": 60, "ny": 127, "category": "SNO", "fcstValue": "5.0~10.0cm"},
            ]
        ),
        pd.DataFrame({"station_id": ["108"], "lat": [37.57], "lon": [126.98], "grid_x": [60], "grid_y": [127]}),
    ).loc[0, ["nwp_precip_amount", "nwp_snow_amount"]].to_dict() == {"nwp_precip_amount": 40.0, "nwp_snow_amount": 7.5}


def test_kma_native_times_are_interpreted_as_korea_time_and_converted_to_utc() -> None:
    raw = pd.DataFrame(
        [
            {"baseDate": "20260101", "baseTime": "0500", "fcstDate": "20260101", "fcstTime": "0600", "nx": 60, "ny": 127, "category": "TMP", "fcstValue": "3"},
        ]
    )
    stations = pd.DataFrame({"station_id": ["108"], "lat": [37.57], "lon": [126.98], "grid_x": [60], "grid_y": [127]})

    prepared = build_kma_prepared_forecast(raw, stations)

    assert prepared.loc[0, "forecast_init_time"].isoformat() == "2025-12-31T20:00:00+00:00"
    assert prepared.loc[0, "valid_time"].isoformat() == "2025-12-31T21:00:00+00:00"
    assert int(prepared.loc[0, "horizon_step"]) == 1

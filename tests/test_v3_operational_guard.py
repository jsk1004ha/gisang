from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from weather_korea_forecast.v2.future_features import build_future_feature_metadata, load_future_weather_features
from weather_korea_forecast.v2.predict import _validate_future_feature_inference_mode


class _Bundle:
    metadata = {"future_features": {"uses_future_weather_features": True, "future_feature_source": "era5_reanalysis", "backtest_only": True}}


def test_operational_inference_rejects_backtest_only_model() -> None:
    with pytest.raises(ValueError, match="backtest-only"):
        _validate_future_feature_inference_mode(_Bundle(), operational_mode=True)


def test_prepared_forecast_csv_adapter_validates_and_converts_units(tmp_path: Path) -> None:
    path = tmp_path / "forecast.csv"
    pd.DataFrame(
        [
            {
                "station_id": "108",
                "forecast_init_time": "2024-01-01T00:00:00Z",
                "valid_time": "2024-01-01T01:00:00Z",
                "horizon_step": 1,
                "nwp_t2m": 281.15,
                "nwp_sp": 101325.0,
                "nwp_u10": 1.0,
                "nwp_v10": 2.0,
                "nwp_tp": 0.0,
            }
        ]
    ).to_csv(path, index=False)

    features = load_future_weather_features("prepared_forecast_csv", "2024-01-01T00:00:00Z", 1, ["108"], path=path)

    assert features.loc[0, "nwp_t2m"] == pytest.approx(8.0)
    assert features.loc[0, "nwp_sp"] == pytest.approx(1013.25)
    assert features.loc[0, "source"] == "prepared_forecast_csv"
    assert pd.Timestamp(features.loc[0, "forecast_init_time"]) == pd.Timestamp("2024-01-01T00:00:00Z")


def test_prepared_forecast_csv_adapter_rejects_missing_horizon(tmp_path: Path) -> None:
    path = tmp_path / "forecast.csv"
    pd.DataFrame(
        [{"station_id": "108", "valid_time": "2024-01-01T01:00:00Z", "horizon_step": 1, "nwp_t2m": 1.0}]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing forecast horizons"):
        load_future_weather_features("prepared_forecast_csv", "2024-01-01T00:00:00Z", 2, ["108"], path=path)




def test_prepared_forecast_csv_adapter_enforces_v4_issue_and_horizon_alignment(tmp_path: Path) -> None:
    path = tmp_path / "forecast.csv"
    pd.DataFrame(
        [
            {
                "station_id": "108",
                "forecast_init_time": "2024-01-01T00:00:00Z",
                "issue_time": "2024-01-01T01:00:00Z",
                "valid_time": "2024-01-01T01:00:00Z",
                "horizon_step": 1,
                "nwp_t2m": 281.15,
            }
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="none are at or before"):
        load_future_weather_features("prepared_forecast_csv", "2024-01-01T00:00:00Z", 1, ["108"], path=path)

    pd.DataFrame(
        [
            {
                "station_id": "108",
                "forecast_init_time": "2024-01-01T00:00:00Z",
                "issue_time": "2024-01-01T00:00:00Z",
                "valid_time": "2024-01-01T01:00:00Z",
                "horizon_step": 2,
                "nwp_t2m": 281.15,
            }
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="valid_time rows"):
        load_future_weather_features("prepared_forecast_csv", "2024-01-01T00:00:00Z", 1, ["108"], path=path)


def test_prepared_forecast_csv_adapter_returns_v4_schema_fields_from_aliases(tmp_path: Path) -> None:
    path = tmp_path / "forecast.csv"
    pd.DataFrame(
        [
            {
                "station_id": "108",
                "forecast_init_time": "2024-01-01T00:00:00Z",
                "issue_time": "2023-12-31T18:00:00Z",
                "valid_time": "2024-01-01T01:00:00Z",
                "horizon_step": 1,
                "nwp_temp_2m": 281.15,
                "nwp_surface_pressure": 101325.0,
                "nwp_u10": 1.0,
                "nwp_v10": 2.0,
                "nwp_total_precipitation": 0.0,
                "nwp_rh": 65.0,
                "gfs_tcc": 0.75,
            }
        ]
    ).to_csv(path, index=False)

    features = load_future_weather_features("prepared", "2024-01-01T00:00:00Z", 1, ["108"], path=path)

    assert features.loc[0, "source"] == "prepared_forecast_csv"
    assert features.loc[0, "nwp_t2m"] == pytest.approx(8.0)
    assert features.loc[0, "nwp_sp"] == pytest.approx(1013.25)
    assert features.loc[0, "nwp_relative_humidity"] == pytest.approx(65.0)
    assert features.loc[0, "nwp_cloud_cover"] == pytest.approx(0.75)
    assert pd.Timestamp(features.loc[0, "issue_time"]) == pd.Timestamp("2023-12-31T18:00:00Z")


def test_prepared_forecast_csv_metadata_can_be_operational_valid() -> None:
    metadata = build_future_feature_metadata(
        {
            "data": {
                "features": {"decoder_known": ["era5_t2m", "era5_sp"]},
                "future_features": {"source": "prepared_forecast_csv", "track": "nwp_assisted_mos"},
            }
        }
    )

    assert metadata["uses_future_weather_features"] is True
    assert metadata["future_feature_source"] == "prepared_forecast_csv"
    assert metadata["operational_valid"] is True
    assert metadata["backtest_only"] is False

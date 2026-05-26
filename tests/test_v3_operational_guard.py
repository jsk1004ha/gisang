from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from weather_korea_forecast.v2.future_features import load_future_weather_features
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


def test_prepared_forecast_csv_adapter_rejects_missing_horizon(tmp_path: Path) -> None:
    path = tmp_path / "forecast.csv"
    pd.DataFrame(
        [{"station_id": "108", "valid_time": "2024-01-01T01:00:00Z", "horizon_step": 1, "nwp_t2m": 1.0}]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing forecast horizons"):
        load_future_weather_features("prepared_forecast_csv", "2024-01-01T00:00:00Z", 2, ["108"], path=path)

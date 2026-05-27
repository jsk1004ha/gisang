from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from weather_korea_forecast.utils.io import read_table
from weather_korea_forecast.v2.future_features import load_future_weather_archive, load_future_weather_features
from weather_korea_forecast.v3 import nwp_mos


def _synthetic_observation_frame() -> pd.DataFrame:
    datetimes = pd.date_range("2024-01-01T00:00:00Z", periods=14 * 24, freq="1h")
    rows = []
    for timestamp in datetimes:
        hour = timestamp.hour
        humidity = 55.0 + 8.0 * np.sin(2 * np.pi * hour / 24.0) + 0.04 * ((timestamp.dayofyear - 1) * 24 + hour)
        if timestamp < pd.Timestamp("2024-01-08T00:00:00Z"):
            split = "train"
        elif timestamp < pd.Timestamp("2024-01-11T00:00:00Z"):
            split = "val"
        else:
            split = "test"
        rows.append(
            {
                "station_id": "108",
                "datetime": timestamp,
                "humidity": humidity,
                "target_value": humidity,
                "obs_humidity": humidity,
                "obs_temp": 10.0 + 0.1 * hour,
                "obs_dew_point_c": 4.0,
                "obs_dew_point_depression": 6.0,
                "lat": 37.5,
                "lon": 127.0,
                "elevation": 80.0,
                "coastal_distance_km": 35.0,
                "split": split,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_nwp_archive(observations: pd.DataFrame, path: Path) -> Path:
    lookup = observations.set_index("datetime")
    rows = []
    for issue_time in pd.date_range("2024-01-04T00:00:00Z", "2024-01-13T20:00:00Z", freq="4h"):
        for lead_hour in range(1, 4):
            valid_time = issue_time + pd.Timedelta(hours=lead_hour)
            if valid_time not in lookup.index:
                continue
            actual = float(lookup.loc[valid_time, "humidity"])
            # Honest forecast baseline has a deterministic lead bias; MOS should learn it.
            nwp_rh = actual - (1.5 + 0.25 * lead_hour)
            rows.append(
                {
                    "station_id": "108",
                    "issue_time": issue_time.isoformat(),
                    "valid_time": valid_time.isoformat(),
                    "lead_hour": lead_hour,
                    "gfs_relative_humidity_2m": nwp_rh,
                    "gfs_temp_2m_c": 10.0,
                    "gfs_dew_point_2m_c": 5.0,
                    "gfs_u10": 1.0,
                    "gfs_v10": 2.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_future_weather_archive_maps_nwp_humidity_aliases(tmp_path: Path) -> None:
    forecast_path = tmp_path / "gfs.csv"
    pd.DataFrame(
        [
            {
                "station_id": "108",
                "issue_time": "2024-01-01T00:00:00Z",
                "valid_time": "2024-01-01T03:00:00Z",
                "gfs_relative_humidity_2m": 73.5,
                "gfs_temp_2m_c": 10.0,
                "gfs_dew_point_2m_c": 5.0,
                "gfs_u10": 3.0,
                "gfs_v10": 4.0,
            }
        ]
    ).to_csv(forecast_path, index=False)

    archive = load_future_weather_archive(
        forecast_path,
        {"data": {"future_features": {"source": "gfs_forecast", "operational_valid": True}}},
    )

    assert archive.loc[0, "lead_hour"] == 3
    assert archive.loc[0, "nwp_relative_humidity_2m"] == pytest.approx(73.5)
    assert archive.loc[0, "nwp_wind_speed"] == pytest.approx(5.0)
    assert "datetime" in archive.columns


def test_load_future_weather_archive_rejects_bad_lead_alignment(tmp_path: Path) -> None:
    forecast_path = tmp_path / "bad_gfs.csv"
    pd.DataFrame(
        [
            {
                "station_id": "108",
                "issue_time": "2024-01-01T00:00:00Z",
                "valid_time": "2024-01-01T03:00:00Z",
                "lead_hour": 2,
                "gfs_temp_2m_c": 10.0,
            }
        ]
    ).to_csv(forecast_path, index=False)

    with pytest.raises(ValueError, match="valid_time must equal issue_time"):
        load_future_weather_archive(
            forecast_path,
            {"data": {"future_features": {"source": "gfs_forecast", "operational_valid": True}}},
        )


def test_nwp_mos_trains_issue_time_aligned_residual_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observations = _synthetic_observation_frame()
    forecast_path = _synthetic_nwp_archive(observations, tmp_path / "nwp.csv")
    monkeypatch.setattr(nwp_mos, "load_or_prepare_v2_training_table", lambda config: observations.copy())
    config = {
        "experiment": {"name": "synthetic_humidity_nwp_mos", "version": "v3"},
        "paths": {"nwp_forecast_csv": str(forecast_path)},
        "data": {
            "target_name": "humidity",
            "window": {"prediction_length": 3},
            "features": {
                "nwp_features": ["nwp_relative_humidity_2m", "nwp_temp_2m_c", "nwp_dew_point_2m_c", "nwp_wind_speed"],
                "init_features": ["obs_humidity", "obs_temp"],
                "static_features": ["lat", "lon", "elevation"],
            },
            "postprocess": {"clip_prediction": [0, 100]},
            "future_features": {
                "track": "nwp_assisted_mos",
                "source": "gfs_forecast",
                "operational_valid": True,
                "weather_columns": ["nwp_relative_humidity_2m"],
            },
        },
        "model": {
            "type": "nwp_mos_lightgbm",
            "residual": True,
            "baseline_column": "nwp_relative_humidity_2m",
            "params": {"n_estimators": 80, "learning_rate": 0.08, "num_leaves": 15, "n_jobs": 1, "verbosity": -1},
        },
        "artifacts": {"root_dir": str(tmp_path / "artifacts")},
    }

    experiment_dir = nwp_mos.run_nwp_mos_experiment(config)
    metrics = json.loads((experiment_dir / "metrics_test.json").read_text(encoding="utf-8"))
    predictions = read_table(experiment_dir / "predictions_test.csv")
    metadata = json.loads((experiment_dir / "future_feature_metadata.json").read_text(encoding="utf-8"))

    assert metrics["rmse"] < 0.5
    assert predictions["horizon_step"].isin([1, 2, 3]).all()
    assert metadata["issue_time_aligned"] is True
    assert metadata["operational_valid"] is True


def test_load_future_weather_features_adapter_returns_canonical_schema(tmp_path: Path) -> None:
    forecast_path = tmp_path / "gfs_features.csv"
    pd.DataFrame(
        [
            {
                "station_id": "108",
                "issue_time": "2024-01-01T00:00:00Z",
                "valid_time": "2024-01-01T01:00:00Z",
                "gfs_t2m": 281.15,
                "gfs_sp": 1005.0,
                "gfs_u10": 1.0,
                "gfs_v10": 2.0,
                "gfs_tp": 0.0,
                "gfs_dew_point_2m_c": 3.0,
            }
        ]
    ).to_csv(forecast_path, index=False)

    features = load_future_weather_features(
        "gfs_forecast",
        "2024-01-01T00:00:00Z",
        1,
        ["108"],
        path=forecast_path,
        config={"data": {"future_features": {"source": "gfs_forecast"}}},
    )

    assert {"station_id", "valid_time", "horizon_step", "nwp_t2m", "nwp_sp", "nwp_u10", "nwp_v10", "nwp_tp", "nwp_dew_point"}.issubset(features.columns)
    assert features.loc[0, "horizon_step"] == 1
    assert features.loc[0, "nwp_t2m"] == pytest.approx(8.0)

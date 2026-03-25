from __future__ import annotations

import pandas as pd

from weather_korea_forecast.data.build_training_table import build_training_table


def test_build_training_table_merges_asos_with_resampled_aws(synthetic_project: dict) -> None:
    root = synthetic_project["root"]
    raw_dir = root / "data" / "raw"
    aws_path = raw_dir / "aws" / "seoul_minutely.csv"
    aws_path.parent.mkdir(parents=True, exist_ok=True)

    minute_times = pd.date_range("2024-01-01T00:00:00Z", periods=180, freq="20min")
    aws = pd.DataFrame(
        {
            "station_id": "SEOUL",
            "datetime": minute_times.astype(str),
            "temp": 999.0,
            "humidity": [70.0, 74.0, 80.0] * 60,
            "pressure": 1008.0,
            "wind_speed": [1.0, 3.0, 5.0] * 60,
            "precipitation": 0.1,
            "quality_flag": "aws",
        }
    )
    aws.to_csv(aws_path, index=False)

    data_config = synthetic_project["data_config"] | {
        "paths": {
            **synthetic_project["data_config"]["paths"],
            "aws_observation_csv": str(aws_path),
        },
        "aws": {
            "source_tz": "UTC",
            "priority": 1,
            "resample_rule": "1h",
            "aggregation": {
                "temp": "mean",
                "humidity": "mean",
                "pressure": "mean",
                "wind_speed": "mean",
                "precipitation": "sum",
                "quality_flag": "last",
            },
        },
    }

    training_table = build_training_table(data_config)
    first_row = training_table.loc[
        (training_table["station_id"] == "SEOUL") & (pd.to_datetime(training_table["datetime"], utc=True) == pd.Timestamp("2024-01-01T00:00:00Z"))
    ].iloc[0]

    assert first_row["temp"] != 999.0
    assert round(float(first_row["humidity"]), 6) == round(74.66666666666667, 6)
    assert first_row["wind_speed"] == 3.0
    assert "aws" in first_row["observation_sources"]

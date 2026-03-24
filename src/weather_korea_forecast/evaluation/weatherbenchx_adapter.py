from __future__ import annotations

import pandas as pd


def to_weatherbenchx_sparse_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["init_time"] = pd.to_datetime(frame["prediction_start"], utc=True)
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
    columns = ["station_id", "init_time", "valid_time", "prediction", "actual"]
    optional = [column for column in ("region", "season", "horizon_step") if column in frame.columns]
    return frame[columns + optional]

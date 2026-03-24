from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    rmse = float(math.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(error) / denom) * 100.0)
    return {"rmse": rmse, "mae": mae, "bias": bias, "mape": mape}


def compute_group_metrics(
    prediction_frame: pd.DataFrame,
    group_columns: list[str],
    actual_column: str = "actual",
    predicted_column: str = "prediction",
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for keys, group in prediction_frame.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = compute_point_metrics(group[actual_column].to_numpy(), group[predicted_column].to_numpy())
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(metrics)
        row["count"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)

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


def compute_prediction_metrics(
    prediction_frame: pd.DataFrame,
    actual_column: str = "actual",
    predicted_column: str = "prediction",
    target_group_column: str = "target_name",
) -> dict[str, object]:
    global_metrics = compute_point_metrics(prediction_frame[actual_column].to_numpy(), prediction_frame[predicted_column].to_numpy())
    if target_group_column not in prediction_frame.columns or prediction_frame[target_group_column].nunique() <= 1:
        summary = dict(global_metrics)
        summary["aggregation"] = "global"
        if target_group_column in prediction_frame.columns and not prediction_frame.empty:
            target_name = str(prediction_frame[target_group_column].iloc[0])
            summary["by_target"] = {target_name: global_metrics}
        return summary

    by_target: dict[str, dict[str, float]] = {}
    for target_name, group in prediction_frame.groupby(target_group_column):
        by_target[str(target_name)] = compute_point_metrics(group[actual_column].to_numpy(), group[predicted_column].to_numpy())

    metric_names = next(iter(by_target.values())).keys()
    macro_metrics = {
        metric_name: float(np.mean([metrics[metric_name] for metrics in by_target.values()]))
        for metric_name in metric_names
    }
    macro_metrics["aggregation"] = "macro_by_target"
    macro_metrics["global"] = global_metrics
    macro_metrics["by_target"] = by_target
    return macro_metrics

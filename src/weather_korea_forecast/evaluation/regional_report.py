from __future__ import annotations

import pandas as pd

from weather_korea_forecast.training.metrics import compute_group_metrics


def build_breakdown_reports(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    reports: dict[str, pd.DataFrame] = {}
    group_sets: list[list[str]] = [["station_id"], ["region"], ["season"], ["horizon_step"]]
    if "target_name" in predictions.columns and predictions["target_name"].nunique() > 1:
        group_sets = [["target_name"], ["target_name", "station_id"], ["target_name", "region"], ["target_name", "season"], ["target_name", "horizon_step"]]
    for group_columns in group_sets:
        existing = [column for column in group_columns if column in predictions.columns]
        if existing:
            reports["_".join(existing)] = compute_group_metrics(predictions, existing)
    return reports

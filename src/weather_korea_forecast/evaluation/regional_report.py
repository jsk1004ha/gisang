from __future__ import annotations

import pandas as pd

from weather_korea_forecast.training.metrics import compute_group_metrics


def build_breakdown_reports(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    reports: dict[str, pd.DataFrame] = {}
    for group_columns in (["station_id"], ["region"], ["season"], ["horizon_step"]):
        existing = [column for column in group_columns if column in predictions.columns]
        if existing:
            reports["_".join(existing)] = compute_group_metrics(predictions, existing)
    return reports

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    frame = df.copy()
    dt = pd.to_datetime(frame[datetime_col], utc=True)
    hour = dt.dt.hour.astype(int)
    dayofyear = dt.dt.dayofyear.astype(int)
    month = dt.dt.month.astype(int)

    frame["hour"] = hour
    frame["month"] = month
    frame["dayofyear"] = dayofyear
    frame["season"] = month.map(_month_to_season)
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["doy_sin"] = np.sin(2 * np.pi * dayofyear / 366.0)
    frame["doy_cos"] = np.cos(2 * np.pi * dayofyear / 366.0)
    frame["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return frame


def _month_to_season(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"

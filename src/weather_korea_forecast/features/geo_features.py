from __future__ import annotations

import pandas as pd


def enrich_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "region" not in frame.columns:
        frame["region"] = "unknown"
    if "coastal_distance_km" not in frame.columns:
        frame["coastal_distance_km"] = float("nan")
    if "terrain_class" not in frame.columns:
        frame["terrain_class"] = frame.get("elevation", pd.Series(dtype=float)).fillna(0).map(_terrain_from_elevation)
    return frame


def _terrain_from_elevation(elevation: float) -> str:
    if elevation >= 400:
        return "mountain"
    if elevation >= 100:
        return "upland"
    return "lowland"

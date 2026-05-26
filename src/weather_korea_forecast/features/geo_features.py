from __future__ import annotations

import pandas as pd


def enrich_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "region" not in frame.columns:
        frame["region"] = "unknown"
    if "region_class" not in frame.columns:
        frame["region_class"] = frame["region"]
    frame["region_class"] = frame.apply(_region_class_for_row, axis=1)
    frame["region"] = frame["region"].where(~frame["region"].isna(), frame["region_class"])
    frame.loc[frame["region"].astype(str).str.lower().isin({"", "nan", "none", "unknown"}), "region"] = frame["region_class"]
    if "coastal_distance_km" not in frame.columns:
        frame["coastal_distance_km"] = float("nan")
    if "coastal_class" not in frame.columns:
        frame["coastal_class"] = frame["coastal_distance_km"].map(_coastal_class_from_distance)
    if "terrain_class" not in frame.columns:
        frame["terrain_class"] = frame.get("elevation", pd.Series(dtype=float)).fillna(0).map(_terrain_from_elevation)
    return frame


def _region_class_for_row(row: pd.Series) -> str:
    configured = str(row.get("region_class", row.get("region", "unknown")) or "unknown").strip()
    station_id = str(row.get("station_id", "")).strip()
    known = {
        "90": "east_coast",
        "93": "mountain_adjacent",
        "108": "capital",
        "112": "west_coast",
        "133": "inland",
        "138": "east_coast",
        "143": "basin",
        "152": "east_coast",
        "156": "inland",
        "159": "south_coast",
        "184": "island",
        "192": "basin",
    }
    if station_id in known and (configured.lower() in {"", "nan", "none", "unknown", "inland"} or station_id in {"108", "143", "192"}):
        return known[station_id]
    if configured.lower() not in {"", "nan", "none", "unknown"}:
        return configured
    lat = _float_or_none(row.get("lat"))
    lon = _float_or_none(row.get("lon"))
    coastal_distance = _float_or_none(row.get("coastal_distance_km"))
    elevation = _float_or_none(row.get("elevation"))
    if lat is not None and lat < 34.5:
        return "island"
    if coastal_distance is not None and coastal_distance <= 5.0:
        if lon is not None and lon >= 128.0:
            return "east_coast" if lat is not None and lat >= 35.5 else "south_coast"
        if lon is not None and lon <= 127.0:
            return "west_coast"
        return "coast"
    if elevation is not None and elevation >= 400.0:
        return "mountain"
    if lat is not None and lon is not None and 37.0 <= lat <= 37.8 and 126.5 <= lon <= 127.5:
        return "capital"
    return "inland"


def _coastal_class_from_distance(distance: float) -> str:
    value = _float_or_none(distance)
    if value is None:
        return "unknown"
    if value <= 5.0:
        return "coastal"
    if value <= 30.0:
        return "near_coast"
    return "inland"


def _terrain_from_elevation(elevation: float) -> str:
    if elevation >= 400:
        return "mountain"
    if elevation >= 100:
        return "upland"
    return "lowland"


def _float_or_none(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric

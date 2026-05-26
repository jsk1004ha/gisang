from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from weather_korea_forecast.features.geo_features import enrich_station_metadata


def load_station_metadata(path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    required = {"station_id", "lat", "lon", "elevation"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing required station metadata columns: {sorted(missing)}")
    metadata["station_id"] = metadata["station_id"].astype(str)
    enriched = enrich_station_metadata(metadata)
    if "region_class" not in enriched.columns:
        enriched["region_class"] = enriched.get("region", "inland")
    enriched = _ensure_v3_metadata_completeness(enriched)
    missing_required = [column for column in _V3_METADATA_COLUMNS if column not in enriched.columns]
    if missing_required:
        warnings.warn(f"Station metadata missing V3 columns after enrichment: {missing_required}", RuntimeWarning, stacklevel=2)
    return enriched


_V3_METADATA_COLUMNS = [
    "station_id",
    "station_name",
    "lat",
    "lon",
    "elevation",
    "coastal_distance_km",
    "region_class",
    "terrain_class",
    "coastal_class",
    "urban_class",
]


def _ensure_v3_metadata_completeness(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "station_name" not in enriched.columns:
        enriched["station_name"] = enriched["station_id"].astype(str)
    if "coastal_distance_km" not in enriched.columns:
        enriched["coastal_distance_km"] = pd.NA
    enriched["region_class"] = enriched["region_class"].fillna("inland").astype(str)
    enriched.loc[enriched["region_class"].str.lower().isin(["", "nan", "none", "unknown"]), "region_class"] = "inland"
    if "region" not in enriched.columns:
        enriched["region"] = enriched["region_class"]
    enriched["region"] = enriched["region"].fillna(enriched["region_class"]).astype(str)
    enriched.loc[enriched["region"].str.lower().isin(["", "nan", "none", "unknown"]), "region"] = enriched["region_class"]
    if "terrain_class" not in enriched.columns:
        enriched["terrain_class"] = "plain"
    enriched["terrain_class"] = enriched["terrain_class"].fillna("plain").astype(str).replace({"lowland": "plain", "upland": "mountain_adjacent"})
    if "coastal_class" not in enriched.columns:
        enriched["coastal_class"] = "inland"
    enriched["coastal_class"] = enriched["coastal_class"].fillna("inland").astype(str).replace({"near_coast": "inland"})
    directional = enriched["region_class"].where(
        enriched["region_class"].isin(["west_coast", "east_coast", "south_coast", "island"]),
        "inland",
    )
    enriched.loc[enriched["coastal_class"].str.lower().isin(["", "nan", "none", "unknown", "coastal"]), "coastal_class"] = directional
    if "urban_class" not in enriched.columns:
        enriched["urban_class"] = "non_urban"
    capital_mask = enriched["region_class"].astype(str).eq("capital")
    enriched.loc[capital_mask, "urban_class"] = enriched.loc[capital_mask, "urban_class"].replace({"non_urban": "urban"})
    return enriched

from __future__ import annotations

from pathlib import Path

import pandas as pd

from weather_korea_forecast.features.geo_features import enrich_station_metadata


def load_station_metadata(path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    required = {"station_id", "lat", "lon", "elevation"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing required station metadata columns: {sorted(missing)}")
    metadata["station_id"] = metadata["station_id"].astype(str)
    return enrich_station_metadata(metadata)

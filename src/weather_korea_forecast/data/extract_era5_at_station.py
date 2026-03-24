from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from weather_korea_forecast.data.align_time_index import align_dataframe_timezone
from weather_korea_forecast.data.station_metadata import load_station_metadata
from weather_korea_forecast.utils.io import read_table, write_table


def extract_era5_at_stations(
    era5_df: pd.DataFrame,
    station_df: pd.DataFrame,
    mode: str = "nearest",
) -> pd.DataFrame:
    if {"station_id", "datetime"}.issubset(era5_df.columns):
        aligned = align_dataframe_timezone(era5_df, source_tz="UTC")
        return aligned.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    required = {"datetime", "lat", "lon"}
    missing = required - set(era5_df.columns)
    if missing:
        raise ValueError(f"Missing required ERA5 grid columns: {sorted(missing)}")

    frame = align_dataframe_timezone(era5_df, source_tz="UTC")
    value_columns = [column for column in frame.columns if column not in {"datetime", "lat", "lon"}]
    outputs: list[dict[str, float | str | pd.Timestamp]] = []

    for timestamp, grid in frame.groupby("datetime"):
        for station in station_df.itertuples(index=False):
            if mode == "bilinear":
                values = _bilinear_extract(grid, station.lat, station.lon, value_columns)
            else:
                values = _nearest_extract(grid, station.lat, station.lon, value_columns)
            row = {"station_id": station.station_id, "datetime": timestamp}
            row.update(values)
            outputs.append(row)

    result = pd.DataFrame(outputs)
    return result.sort_values(["station_id", "datetime"]).reset_index(drop=True)


def _nearest_extract(grid: pd.DataFrame, lat: float, lon: float, value_columns: list[str]) -> dict[str, float]:
    distances = (grid["lat"] - lat) ** 2 + (grid["lon"] - lon) ** 2
    row = grid.loc[distances.idxmin()]
    return {column: float(row[column]) for column in value_columns}


def _bilinear_extract(grid: pd.DataFrame, lat: float, lon: float, value_columns: list[str]) -> dict[str, float]:
    lats = np.sort(grid["lat"].unique())
    lons = np.sort(grid["lon"].unique())
    if lat < lats.min() or lat > lats.max() or lon < lons.min() or lon > lons.max():
        return _nearest_extract(grid, lat, lon, value_columns)

    lat_hi_index = np.searchsorted(lats, lat, side="left")
    lon_hi_index = np.searchsorted(lons, lon, side="left")
    lat_lo = lats[max(lat_hi_index - 1, 0)]
    lat_hi = lats[min(lat_hi_index, len(lats) - 1)]
    lon_lo = lons[max(lon_hi_index - 1, 0)]
    lon_hi = lons[min(lon_hi_index, len(lons) - 1)]

    corners = grid.set_index(["lat", "lon"])
    points = [(lat_lo, lon_lo), (lat_lo, lon_hi), (lat_hi, lon_lo), (lat_hi, lon_hi)]
    if any(point not in corners.index for point in points):
        return _nearest_extract(grid, lat, lon, value_columns)

    if lat_hi == lat_lo or lon_hi == lon_lo:
        return _nearest_extract(grid, lat, lon, value_columns)

    lat_weight = (lat - lat_lo) / (lat_hi - lat_lo)
    lon_weight = (lon - lon_lo) / (lon_hi - lon_lo)
    result: dict[str, float] = {}
    for column in value_columns:
        q11 = corners.loc[(lat_lo, lon_lo), column]
        q12 = corners.loc[(lat_lo, lon_hi), column]
        q21 = corners.loc[(lat_hi, lon_lo), column]
        q22 = corners.loc[(lat_hi, lon_hi), column]
        result[column] = float(
            q11 * (1 - lat_weight) * (1 - lon_weight)
            + q21 * lat_weight * (1 - lon_weight)
            + q12 * (1 - lat_weight) * lon_weight
            + q22 * lat_weight * lon_weight
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ERA5 grid values at station locations.")
    parser.add_argument("--era5-path", required=True)
    parser.add_argument("--station-metadata-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--mode", choices=["nearest", "bilinear"], default="nearest")
    args = parser.parse_args()

    era5_df = read_table(args.era5_path)
    station_df = load_station_metadata(args.station_metadata_path)
    extracted = extract_era5_at_stations(era5_df=era5_df, station_df=station_df, mode=args.mode)
    write_table(extracted, args.output_path)


if __name__ == "__main__":
    main()

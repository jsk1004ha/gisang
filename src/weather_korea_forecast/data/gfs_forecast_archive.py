from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

GFS_OUTPUT_COLUMNS = [
    "station_id",
    "forecast_init_time",
    "valid_time",
    "horizon_step",
    "nwp_t2m",
    "nwp_sp",
    "nwp_u10",
    "nwp_v10",
    "nwp_tp",
    "nwp_dew_point",
    "nwp_cloud_cover",
    "source",
    "issue_time",
]

GFS_ALIASES = {
    "gfs_temp_2m_c": "nwp_t2m",
    "gfs_t2m": "nwp_t2m",
    "t2m": "nwp_t2m",
    "tmp": "nwp_t2m",
    "gfs_surface_pressure": "nwp_sp",
    "surface_pressure": "nwp_sp",
    "sp": "nwp_sp",
    "pres": "nwp_sp",
    "gfs_u10": "nwp_u10",
    "u10": "nwp_u10",
    "ugrd": "nwp_u10",
    "gfs_v10": "nwp_v10",
    "v10": "nwp_v10",
    "vgrd": "nwp_v10",
    "gfs_total_precipitation": "nwp_tp",
    "total_precipitation": "nwp_tp",
    "tp": "nwp_tp",
    "apcp": "nwp_tp",
    "gfs_dew_point_2m_c": "nwp_dew_point",
    "dew_point_2m": "nwp_dew_point",
    "d2m": "nwp_dew_point",
    "gfs_cloud_cover": "nwp_cloud_cover",
    "total_cloud_cover": "nwp_cloud_cover",
    "tcc": "nwp_cloud_cover",
}


def build_gfs_prepared_forecast(raw_forecast: pd.DataFrame, station_metadata: pd.DataFrame) -> pd.DataFrame:
    raw = _normalize_gfs_table(raw_forecast)
    stations = station_metadata.copy()
    if "station_id" not in stations.columns:
        raise ValueError("station metadata requires station_id")
    stations["station_id"] = stations["station_id"].astype(str)
    if "station_id" in raw.columns:
        prepared = raw.copy()
        prepared["station_id"] = prepared["station_id"].astype(str)
    else:
        prepared = _nearest_grid_to_stations(raw, stations)
    prepared["source"] = "gfs_forecast"
    prepared["issue_time"] = prepared["forecast_init_time"]
    for column in GFS_OUTPUT_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    return prepared[GFS_OUTPUT_COLUMNS].sort_values(["station_id", "forecast_init_time", "horizon_step"]).reset_index(drop=True)


def gfs_nomads_filter_url(
    date: str,
    cycle: str,
    forecast_hour: int,
    *,
    variables: list[str],
    leftlon: float,
    rightlon: float,
    toplat: float,
    bottomlat: float,
    directory: str | None = None,
) -> str:
    cycle = f"{int(cycle):02d}"
    forecast_hour = int(forecast_hour)
    file_name = f"gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour:03d}"
    params: dict[str, str | int | float] = {
        "dir": directory or f"/gfs.{date}/{cycle}/atmos",
        "file": file_name,
        "lev_2_m_above_ground": "on",
        "lev_10_m_above_ground": "on",
        "lev_surface": "on",
        "subregion": "",
        "leftlon": leftlon,
        "rightlon": rightlon,
        "toplat": toplat,
        "bottomlat": bottomlat,
    }
    for variable in variables:
        params[f"var_{str(variable).upper()}"] = "on"
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?" + urlencode(params)


def _normalize_gfs_table(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "issue_time" in normalized.columns and "forecast_init_time" not in normalized.columns:
        normalized["forecast_init_time"] = normalized["issue_time"]
    if "lead_hour" in normalized.columns and "horizon_step" not in normalized.columns:
        normalized["horizon_step"] = normalized["lead_hour"]
    required = {"forecast_init_time", "horizon_step"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"GFS forecast table missing required columns: {missing}")
    normalized["forecast_init_time"] = pd.to_datetime(normalized["forecast_init_time"], utc=True)
    normalized["horizon_step"] = pd.to_numeric(normalized["horizon_step"], errors="raise").astype(int)
    if "valid_time" in normalized.columns:
        normalized["valid_time"] = pd.to_datetime(normalized["valid_time"], utc=True)
    else:
        normalized["valid_time"] = normalized["forecast_init_time"] + pd.to_timedelta(normalized["horizon_step"], unit="h")
    for source, target in GFS_ALIASES.items():
        if source in normalized.columns and target not in normalized.columns:
            normalized[target] = normalized[source]
    for temp_column in ["nwp_t2m", "nwp_dew_point"]:
        if temp_column in normalized.columns:
            values = pd.to_numeric(normalized[temp_column], errors="coerce")
            normalized[temp_column] = values - 273.15 if values.median(skipna=True) > 150 else values
    if "nwp_sp" in normalized.columns:
        pressure = pd.to_numeric(normalized["nwp_sp"], errors="coerce")
        normalized["nwp_sp"] = pressure / 100.0 if pressure.median(skipna=True) > 2000 else pressure
    return normalized


def _nearest_grid_to_stations(raw: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    if not {"lat", "lon"}.issubset(raw.columns) or not {"lat", "lon"}.issubset(stations.columns):
        raise ValueError("GFS station-less grid table requires raw/station lat/lon for nearest extraction")
    rows: list[dict[str, object]] = []
    group_cols = ["forecast_init_time", "horizon_step", "valid_time"]
    for _, group in raw.groupby(group_cols, dropna=False):
        for station in stations.itertuples(index=False):
            station_data = station._asdict()
            distances = np.square(group["lat"].astype(float) - float(station_data["lat"])) + np.square(group["lon"].astype(float) - float(station_data["lon"]))
            nearest = group.loc[distances.idxmin()].to_dict()
            nearest["station_id"] = str(station_data["station_id"])
            rows.append(nearest)
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert local GFS/NOMADS forecast table to prepared forecast archive schema.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--station-metadata", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    raw = pd.read_csv(args.input)
    stations = pd.read_csv(args.station_metadata)
    prepared = build_gfs_prepared_forecast(raw, stations)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from weather_korea_forecast.data.station_metadata import load_station_metadata
from weather_korea_forecast.utils.io import write_table
from weather_korea_forecast.utils.paths import resolve_path

GFS_S3_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

MESSAGE_SPECS = {
    "rh2m": ("RH", "2 m above ground", "gfs_relative_humidity_2m"),
    "t2m": ("TMP", "2 m above ground", "gfs_temp_2m_c"),
    "d2m": ("DPT", "2 m above ground", "gfs_dew_point_2m_c"),
    "sp": ("PRES", "surface", "gfs_surface_pressure"),
    "u10": ("UGRD", "10 m above ground", "gfs_u10"),
    "v10": ("VGRD", "10 m above ground", "gfs_v10"),
    "tp": ("APCP", "surface", "gfs_total_precipitation"),
    "gust": ("GUST", "surface", "gfs_gust"),
    "tcc": ("TCDC", "entire atmosphere", "gfs_total_cloud_cover"),
    "lcc": ("LCDC", "low cloud layer", "gfs_low_cloud_cover"),
    "dswrf": ("DSWRF", "surface", "gfs_shortwave_radiation"),
    "dlwrf": ("DLWRF", "surface", "gfs_longwave_radiation"),
    "tssoil": ("TSOIL", "0-0.1 m below ground", "gfs_soil_temperature_c"),
    "land": ("LAND", "surface", "gfs_land_sea_mask"),
    "spfh2m": ("SPFH", "2 m above ground", "gfs_specific_humidity_2m"),
    "pwat": ("PWAT", "entire atmosphere (considered as a single layer)", "gfs_precipitable_water"),
    "prate": ("PRATE", "surface", "gfs_precipitation_rate"),
    "prmsl": ("PRMSL", "mean sea level", "gfs_mean_sea_level_pressure"),
}

CFGRIB_NAMES = {
    "gfs_relative_humidity_2m": "r2",
    "gfs_temp_2m_c": "t2m",
    "gfs_dew_point_2m_c": "d2m",
    "gfs_surface_pressure": "sp",
    "gfs_u10": "u10",
    "gfs_v10": "v10",
    "gfs_total_precipitation": "tp",
    "gfs_gust": "gust",
    "gfs_total_cloud_cover": "tcc",
    "gfs_low_cloud_cover": "lcc",
    "gfs_shortwave_radiation": "sdswrf",
    "gfs_longwave_radiation": "sdlwrf",
    "gfs_soil_temperature_c": "st",
    "gfs_land_sea_mask": "lsm",
    "gfs_specific_humidity_2m": "sh2",
    "gfs_precipitable_water": "pwat",
    "gfs_precipitation_rate": "prate",
    "gfs_mean_sea_level_pressure": "prmsl",
}


@dataclass(frozen=True)
class IndexRow:
    number: int
    start: int
    variable: str
    level: str
    text: str


def export_gfs_surface_forecasts(
    station_metadata_csv: str | Path,
    output_csv: str | Path,
    start_date: str,
    end_date: str,
    station_ids: Iterable[str] | None = None,
    cycles: Iterable[str] = ("00", "06", "12", "18"),
    lead_hours: Iterable[int] = range(1, 25),
    variables: Iterable[str] = ("rh2m", "t2m", "d2m", "sp", "u10", "v10", "tp"),
    cache_dir: str | Path = "data/raw/nwp/gfs_cache",
) -> Path:
    """Download selected GFS surface messages and export station-nearest forecast CSV.

    Output schema includes ``station_id, issue_time, valid_time, lead_hour`` plus
    canonical GFS/NWP columns suitable for the V3 NWP-MOS trainer.
    """

    stations = load_station_metadata(station_metadata_csv).copy()
    stations["station_id"] = stations["station_id"].astype(str)
    if station_ids:
        wanted = {str(value) for value in station_ids}
        stations = stations.loc[stations["station_id"].isin(wanted)].copy()
    if stations.empty:
        raise ValueError("No stations selected for GFS extraction.")

    dates = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")
    cache_path = resolve_path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    selected_variables = [str(v) for v in variables]

    for date in dates:
        date_str = date.strftime("%Y%m%d")
        for cycle in cycles:
            cycle = f"{int(cycle):02d}"
            issue_time = pd.to_datetime(f"{date_str}{cycle}", format="%Y%m%d%H", utc=True)
            for lead in lead_hours:
                lead = int(lead)
                try:
                    grib_path = _download_gfs_subset(date_str, cycle, lead, selected_variables, cache_path)
                    extracted = _extract_station_values(grib_path, stations)
                except Exception as exc:
                    print(f"[gfs] skip {date_str} {cycle} f{lead:03d}: {exc}")
                    continue
                valid_time = issue_time + pd.Timedelta(hours=lead)
                for station_id, values in extracted.items():
                    row = {
                        "station_id": station_id,
                        "issue_time": issue_time.isoformat(),
                        "valid_time": valid_time.isoformat(),
                        "lead_hour": lead,
                    }
                    row.update(values)
                    _append_nwp_aliases(row)
                    rows.append(row)

    if not rows:
        raise RuntimeError("No GFS forecast rows were extracted.")
    frame = pd.DataFrame(rows).sort_values(["station_id", "issue_time", "lead_hour"]).reset_index(drop=True)
    return write_table(frame, output_csv)


def _download_gfs_subset(date_str: str, cycle: str, lead: int, variables: list[str], cache_dir: Path) -> Path:
    file_name = f"gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}"
    out = cache_dir / f"gfs_{date_str}_{cycle}_f{lead:03d}_{'-'.join(variables)}.grib2"
    if out.exists() and out.stat().st_size > 0:
        return out
    base = f"{GFS_S3_BASE}/gfs.{date_str}/{cycle}/atmos/{file_name}"
    idx_text = _get_text_with_retries(base + ".idx")
    index_rows = _parse_index(idx_text)
    starts = [row.start for row in index_rows]
    selected: list[tuple[IndexRow, int | None, str]] = []
    for variable in variables:
        if variable not in MESSAGE_SPECS:
            raise ValueError(f"Unsupported GFS variable {variable!r}; choose from {sorted(MESSAGE_SPECS)}")
        grib_var, level, _ = MESSAGE_SPECS[variable]
        matches = [row for row in index_rows if row.variable == grib_var and row.level == level]
        if not matches:
            raise RuntimeError(f"No GFS index message for variable={grib_var} level={level} in {base}.idx")
        # Prefer the first APCP surface message; duplicate APCP entries can share the same accumulation interval.
        row = matches[0]
        next_start = next((start for start in starts if start > row.start), None)
        selected.append((row, next_start, variable))

    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        for row, next_start, variable in selected:
            end = next_start - 1 if next_start is not None else ""
            content = _get_bytes_with_retries(base, headers={"Range": f"bytes={row.start}-{end}"})
            handle.write(content)
    tmp.replace(out)
    return out


def _parse_index(text: str) -> list[IndexRow]:
    rows: list[IndexRow] = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) < 5:
            continue
        try:
            rows.append(IndexRow(number=int(parts[0]), start=int(parts[1]), variable=parts[3], level=parts[4], text=line))
        except ValueError:
            continue
    return rows


def _extract_station_values(grib_path: Path, stations: pd.DataFrame) -> dict[str, dict[str, float]]:
    try:
        import cfgrib  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("GFS extraction requires optional package cfgrib. Install cfgrib/eccodes in the runtime environment.") from exc

    datasets = cfgrib.open_datasets(str(grib_path), backend_kwargs={"indexpath": ""})
    values_by_station: dict[str, dict[str, float]] = {str(row.station_id): {} for row in stations.itertuples()}
    for dataset in datasets:
        for canonical, cf_name in CFGRIB_NAMES.items():
            if cf_name not in dataset:
                continue
            data_array = dataset[cf_name]
            for row in stations.itertuples():
                lon = float(row.lon) % 360.0
                lat = float(row.lat)
                value = float(data_array.sel(latitude=lat, longitude=lon, method="nearest").item())
                if canonical.endswith("_c"):
                    value = _kelvin_to_celsius_if_needed(value)
                if canonical in {"gfs_surface_pressure", "gfs_mean_sea_level_pressure"} and value > 2000.0:
                    value = value / 100.0
                values_by_station[str(row.station_id)][canonical] = value
    for values in values_by_station.values():
        _append_nwp_aliases(values)
    return values_by_station


def _append_nwp_aliases(row: dict[str, object]) -> None:
    if "gfs_relative_humidity_2m" in row:
        row.setdefault("nwp_relative_humidity_2m", row["gfs_relative_humidity_2m"])
    if "gfs_temp_2m_c" in row:
        row.setdefault("nwp_temp_2m_c", row["gfs_temp_2m_c"])
    if "gfs_dew_point_2m_c" in row:
        row.setdefault("nwp_dew_point_2m_c", row["gfs_dew_point_2m_c"])
    if "gfs_surface_pressure" in row:
        row.setdefault("nwp_surface_pressure", row["gfs_surface_pressure"])
    if "gfs_u10" in row:
        row.setdefault("nwp_u10", row["gfs_u10"])
    if "gfs_v10" in row:
        row.setdefault("nwp_v10", row["gfs_v10"])
    if "gfs_total_precipitation" in row:
        row.setdefault("nwp_total_precipitation", row["gfs_total_precipitation"])
    if "gfs_gust" in row:
        row.setdefault("nwp_gust", row["gfs_gust"])
    if "gfs_total_cloud_cover" in row:
        row.setdefault("nwp_cloud_cover", row["gfs_total_cloud_cover"])
    if "gfs_low_cloud_cover" in row:
        row.setdefault("nwp_low_cloud_cover", row["gfs_low_cloud_cover"])
    if "gfs_shortwave_radiation" in row:
        row.setdefault("nwp_shortwave_radiation", row["gfs_shortwave_radiation"])
    if "gfs_longwave_radiation" in row:
        row.setdefault("nwp_longwave_radiation", row["gfs_longwave_radiation"])
    if "gfs_soil_temperature_c" in row:
        row.setdefault("nwp_soil_temperature", row["gfs_soil_temperature_c"])
    if "gfs_land_sea_mask" in row:
        row.setdefault("nwp_land_sea_mask", row["gfs_land_sea_mask"])
    if "gfs_specific_humidity_2m" in row:
        row.setdefault("nwp_specific_humidity", row["gfs_specific_humidity_2m"])
    if "gfs_precipitable_water" in row:
        row.setdefault("nwp_pwat", row["gfs_precipitable_water"])
    if "gfs_precipitation_rate" in row:
        row.setdefault("nwp_precip_rate", row["gfs_precipitation_rate"])
    if "gfs_mean_sea_level_pressure" in row:
        row.setdefault("nwp_mslp", row["gfs_mean_sea_level_pressure"])
    if "nwp_u10" in row and "nwp_v10" in row:
        u = float(row["nwp_u10"])
        v = float(row["nwp_v10"])
        row.setdefault("nwp_wind_speed", float(np.sqrt(u * u + v * v)))
        row.setdefault("nwp_wind_direction", float((270.0 - np.degrees(np.arctan2(v, u))) % 360.0))


def _kelvin_to_celsius_if_needed(value: float) -> float:
    return value - 273.15 if value > 150.0 else value


def _get_text_with_retries(url: str, tries: int = 3) -> str:
    return _request_with_retries(url, tries=tries).text


def _get_bytes_with_retries(url: str, headers: dict[str, str], tries: int = 3) -> bytes:
    return _request_with_retries(url, headers=headers, tries=tries).content


def _request_with_retries(url: str, headers: dict[str, str] | None = None, tries: int = 3) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(tries):
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            return response
        except Exception as exc:  # pragma: no cover - network retry path.
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed after {tries} attempts: {url}: {last_exc}")


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(v) for v in part.split("-", 1)]
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Export station-nearest NOAA GFS surface forecast covariates.")
    parser.add_argument("--station-metadata-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--station-ids", default="", help="Comma-separated station ids; default all stations in metadata")
    parser.add_argument("--cycles", default="00,06,12,18")
    parser.add_argument("--lead-hours", default="1-24")
    parser.add_argument("--variables", default="rh2m,t2m,d2m,sp,u10,v10,tp")
    parser.add_argument("--cache-dir", default="data/raw/nwp/gfs_cache")
    args = parser.parse_args()

    station_ids = [value.strip() for value in args.station_ids.split(",") if value.strip()] or None
    cycles = [value.strip() for value in args.cycles.split(",") if value.strip()]
    lead_hours = _parse_int_list(args.lead_hours)
    variables = [value.strip() for value in args.variables.split(",") if value.strip()]
    output = export_gfs_surface_forecasts(
        station_metadata_csv=args.station_metadata_csv,
        output_csv=args.output_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        station_ids=station_ids,
        cycles=cycles,
        lead_hours=lead_hours,
        variables=variables,
        cache_dir=args.cache_dir,
    )
    print(output)


if __name__ == "__main__":
    main()

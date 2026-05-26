from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from weather_korea_forecast.data.station_metadata import load_station_metadata
from weather_korea_forecast.utils.io import write_table
from weather_korea_forecast.utils.paths import resolve_path

OPEN_METEO_SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
DEFAULT_HOURLY_VARIABLES = [
    "relative_humidity_2m",
    "temperature_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]

VARIABLE_TO_CANONICAL = {
    "relative_humidity_2m": "nwp_relative_humidity_2m",
    "temperature_2m": "nwp_temp_2m_c",
    "dew_point_2m": "nwp_dew_point_2m_c",
    "surface_pressure": "nwp_surface_pressure",
    "wind_speed_10m": "nwp_wind_speed_10m_kmh",
    "wind_direction_10m": "nwp_wind_direction_10m_deg",
    "precipitation": "nwp_total_precipitation",
}


def export_open_meteo_single_runs(
    station_metadata_csv: str | Path,
    output_csv: str | Path,
    start_date: str,
    end_date: str,
    model: str = "ecmwf_ifs",
    station_ids: Iterable[str] | None = None,
    cycles: Iterable[str] = ("00", "06", "12", "18"),
    lead_hours: Iterable[int] = range(1, 25),
    hourly_variables: Iterable[str] = DEFAULT_HOURLY_VARIABLES,
    batch_size: int = 12,
    sleep_seconds: float = 0.0,
) -> Path:
    """Export station forecast covariates from Open-Meteo Single Runs.

    Open-Meteo Single Runs preserves individual model initialisation times via
    the required ``run`` parameter.  The exported CSV is suitable for the V3
    issue-time-aligned NWP-MOS trainer.
    """

    stations = load_station_metadata(station_metadata_csv).copy()
    stations["station_id"] = stations["station_id"].astype(str)
    if station_ids:
        wanted = {str(value) for value in station_ids}
        stations = stations.loc[stations["station_id"].isin(wanted)].copy()
    if stations.empty:
        raise ValueError("No stations selected for Open-Meteo extraction.")

    leads = sorted({int(value) for value in lead_hours})
    if not leads or min(leads) < 0:
        raise ValueError("lead_hours must contain non-negative integers.")
    forecast_hours = max(leads) + 1  # Open-Meteo includes the run hour as lead 0.
    variables = [str(value) for value in hourly_variables]
    dates = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")
    rows: list[dict[str, object]] = []

    station_batches = [stations.iloc[index : index + batch_size].copy() for index in range(0, len(stations), batch_size)]
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        compact_date = date.strftime("%Y%m%d")
        for cycle in cycles:
            cycle = f"{int(cycle):02d}"
            issue_time = pd.to_datetime(f"{compact_date}{cycle}", format="%Y%m%d%H", utc=True)
            run = f"{date_str}T{cycle}:00"
            for batch in station_batches:
                try:
                    payload = _request_single_run(
                        batch=batch,
                        run=run,
                        forecast_hours=forecast_hours,
                        variables=variables,
                        model=model,
                    )
                except Exception as exc:
                    print(f"[open-meteo] skip run={run} stations={','.join(batch['station_id'])}: {exc}")
                    continue
                for station_row, station_payload in zip(batch.itertuples(index=False), _payloads_as_list(payload)):
                    rows.extend(_rows_from_payload(str(station_row.station_id), issue_time, station_payload, leads, variables, model))
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

    if not rows:
        raise RuntimeError("No Open-Meteo forecast rows were extracted.")
    frame = pd.DataFrame(rows).sort_values(["station_id", "issue_time", "lead_hour"]).reset_index(drop=True)
    return write_table(frame, output_csv)


def _request_single_run(batch: pd.DataFrame, run: str, forecast_hours: int, variables: list[str], model: str):
    params = {
        "latitude": ",".join(str(float(value)) for value in batch["lat"]),
        "longitude": ",".join(str(float(value)) for value in batch["lon"]),
        "run": run,
        "forecast_hours": int(forecast_hours),
        "hourly": ",".join(variables),
        "models": model,
        "timezone": "GMT",
    }
    response = requests.get(OPEN_METEO_SINGLE_RUNS_URL, params=params, timeout=90)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(str(data.get("reason", data)))
    return data


def _payloads_as_list(payload) -> list[dict]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    return [dict(payload)]


def _rows_from_payload(
    station_id: str,
    issue_time: pd.Timestamp,
    payload: dict,
    lead_hours: list[int],
    variables: list[str],
    model: str,
) -> list[dict[str, object]]:
    hourly = dict(payload.get("hourly") or {})
    times = hourly.get("time") or []
    rows: list[dict[str, object]] = []
    for index, raw_time in enumerate(times):
        valid_time = pd.Timestamp(raw_time, tz="UTC")
        lead = int(round((valid_time - issue_time) / pd.Timedelta(hours=1)))
        if lead not in lead_hours:
            continue
        row: dict[str, object] = {
            "station_id": station_id,
            "issue_time": issue_time.isoformat(),
            "valid_time": valid_time.isoformat(),
            "lead_hour": lead,
            "nwp_model": model,
        }
        for variable in variables:
            value = _value_at(hourly, variable, index, model)
            if value is None:
                continue
            canonical = VARIABLE_TO_CANONICAL.get(variable, f"nwp_{variable}")
            row[canonical] = value
        _append_derived_wind(row)
        rows.append(row)
    return rows


def _value_at(hourly: dict, variable: str, index: int, model: str):
    candidates = [variable, f"{variable}_{model}"]
    for key in candidates:
        values = hourly.get(key)
        if values is None or index >= len(values):
            continue
        value = values[index]
        if value is not None:
            return value
    return None


def _append_derived_wind(row: dict[str, object]) -> None:
    speed_kmh = row.get("nwp_wind_speed_10m_kmh")
    direction_deg = row.get("nwp_wind_direction_10m_deg")
    if speed_kmh is None:
        return
    speed_ms = float(speed_kmh) / 3.6
    row["nwp_wind_speed"] = speed_ms
    if direction_deg is None:
        return
    direction = np.deg2rad(float(direction_deg))
    # Meteorological direction is where wind comes from.  Store simple vector-like
    # components for ML features; exact sign is less important than consistency.
    row["nwp_u10"] = -speed_ms * np.sin(direction)
    row["nwp_v10"] = -speed_ms * np.cos(direction)


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(value) for value in part.split("-", 1)]
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Open-Meteo Single Runs forecast covariates for station MOS.")
    parser.add_argument("--station-metadata-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--model", default="ecmwf_ifs")
    parser.add_argument("--station-ids", default="", help="Comma-separated station ids; default all stations")
    parser.add_argument("--cycles", default="00,06,12,18")
    parser.add_argument("--lead-hours", default="1-24")
    parser.add_argument("--hourly", default=",".join(DEFAULT_HOURLY_VARIABLES))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    station_ids = [value.strip() for value in args.station_ids.split(",") if value.strip()] or None
    output = export_open_meteo_single_runs(
        station_metadata_csv=args.station_metadata_csv,
        output_csv=args.output_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        model=args.model,
        station_ids=station_ids,
        cycles=[value.strip() for value in args.cycles.split(",") if value.strip()],
        lead_hours=_parse_int_list(args.lead_hours),
        hourly_variables=[value.strip() for value in args.hourly.split(",") if value.strip()],
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
    )
    print(output)


if __name__ == "__main__":
    main()

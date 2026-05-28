from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KMA_CATEGORY_TO_COLUMN = {
    "TMP": "nwp_t2m",
    "REH": "nwp_humidity",
    "WSD": "nwp_wind_speed",
    "VEC": "nwp_wind_direction",
    "SKY": "nwp_sky_code",
    "PTY": "nwp_precip_type",
    "POP": "nwp_precip_probability",
    "PCP": "nwp_precip_amount",
    "SNO": "nwp_snow_amount",
}
KMA_OUTPUT_COLUMNS = [
    "station_id",
    "forecast_init_time",
    "valid_time",
    "horizon_step",
    "nwp_t2m",
    "nwp_humidity",
    "nwp_wind_speed",
    "nwp_wind_direction",
    "nwp_cloud_cover",
    "nwp_sky_code",
    "nwp_precip_type",
    "nwp_precip_probability",
    "nwp_precip_amount",
    "nwp_snow_amount",
    "source",
    "issue_time",
]


def build_kma_prepared_forecast(
    raw_forecast: pd.DataFrame,
    station_metadata: pd.DataFrame,
    *,
    forecast_grid: pd.DataFrame | None = None,
    mapping_output: str | Path | None = None,
) -> pd.DataFrame:
    """Convert KMA short-term forecast category rows to prepared forecast schema.

    The function accepts local CSV/API-shaped KMA records. Expected fields are
    KMA's `baseDate/baseTime/fcstDate/fcstTime/nx/ny/category/fcstValue` names;
    snake_case aliases are accepted to keep local exports easy to use.
    """

    raw = _normalize_kma_columns(raw_forecast)
    mapping = build_station_grid_mapping(station_metadata, forecast_grid=forecast_grid)
    if mapping_output is not None:
        Path(mapping_output).parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(mapping_output, index=False)
    if raw.empty:
        return pd.DataFrame(columns=KMA_OUTPUT_COLUMNS)

    raw["forecast_init_time"] = _parse_kma_time(raw["base_date"], raw["base_time"])
    raw["valid_time"] = _parse_kma_time(raw["fcst_date"], raw["fcst_time"])
    raw["horizon_step"] = ((raw["valid_time"] - raw["forecast_init_time"]) / pd.Timedelta(hours=1)).round().astype(int)
    raw["category"] = raw["category"].astype(str).str.upper()
    raw["standard_column"] = raw["category"].map(KMA_CATEGORY_TO_COLUMN)
    raw = raw.loc[raw["standard_column"].notna()].copy()
    raw["standard_value"] = raw.apply(lambda row: _parse_kma_value(row["category"], row["fcst_value"]), axis=1)

    pivot_index = ["nx", "ny", "forecast_init_time", "valid_time", "horizon_step"]
    pivot = (
        raw.pivot_table(index=pivot_index, columns="standard_column", values="standard_value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    joined = mapping.merge(pivot, left_on=["forecast_grid_x", "forecast_grid_y"], right_on=["nx", "ny"], how="inner")
    if joined.empty:
        return pd.DataFrame(columns=KMA_OUTPUT_COLUMNS)
    joined["station_id"] = joined["station_id"].astype(str)
    if "nwp_sky_code" in joined.columns:
        joined["nwp_cloud_cover"] = joined["nwp_sky_code"].map(_sky_code_to_cloud_cover)
    joined["source"] = "kma_forecast"
    joined["issue_time"] = joined["forecast_init_time"]
    for column in KMA_OUTPUT_COLUMNS:
        if column not in joined.columns:
            joined[column] = pd.NA
    return joined[KMA_OUTPUT_COLUMNS].sort_values(["station_id", "forecast_init_time", "horizon_step"]).reset_index(drop=True)


def build_station_grid_mapping(station_metadata: pd.DataFrame, *, forecast_grid: pd.DataFrame | None = None) -> pd.DataFrame:
    stations = station_metadata.copy()
    if "station_id" not in stations.columns:
        raise ValueError("station metadata requires station_id")
    stations["station_id"] = stations["station_id"].astype(str)
    x_col = _first_existing(stations, ["grid_x", "nx", "forecast_grid_x", "kma_grid_x"])
    y_col = _first_existing(stations, ["grid_y", "ny", "forecast_grid_y", "kma_grid_y"])
    rows: list[dict[str, Any]] = []
    if x_col and y_col:
        for row in stations.itertuples(index=False):
            data = row._asdict()
            rows.append(
                {
                    "station_id": data["station_id"],
                    "lat": data.get("lat"),
                    "lon": data.get("lon"),
                    "forecast_grid_x": int(data[x_col]),
                    "forecast_grid_y": int(data[y_col]),
                    "mapping_method": "station_metadata_grid",
                }
            )
        return pd.DataFrame(rows)
    if forecast_grid is None:
        raise ValueError("station metadata must include grid_x/grid_y or forecast_grid with lat/lon for nearest mapping")
    grid = forecast_grid.copy()
    nx_col = _first_existing(grid, ["nx", "grid_x", "forecast_grid_x"])
    ny_col = _first_existing(grid, ["ny", "grid_y", "forecast_grid_y"])
    if not nx_col or not ny_col or not {"lat", "lon"}.issubset(grid.columns) or not {"lat", "lon"}.issubset(stations.columns):
        raise ValueError("nearest grid mapping requires station lat/lon and forecast_grid nx/ny/lat/lon")
    for station in stations.itertuples(index=False):
        data = station._asdict()
        distances = np.square(grid["lat"].astype(float) - float(data["lat"])) + np.square(grid["lon"].astype(float) - float(data["lon"]))
        nearest = grid.loc[distances.idxmin()]
        rows.append(
            {
                "station_id": data["station_id"],
                "lat": data.get("lat"),
                "lon": data.get("lon"),
                "forecast_grid_x": int(nearest[nx_col]),
                "forecast_grid_y": int(nearest[ny_col]),
                "mapping_method": "nearest_lat_lon",
            }
        )
    return pd.DataFrame(rows)


def _normalize_kma_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "baseDate": "base_date",
        "baseTime": "base_time",
        "fcstDate": "fcst_date",
        "fcstTime": "fcst_time",
        "fcstValue": "fcst_value",
    }
    normalized = frame.rename(columns={source: target for source, target in aliases.items() if source in frame.columns}).copy()
    required = {"base_date", "base_time", "fcst_date", "fcst_time", "nx", "ny", "category", "fcst_value"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"KMA forecast table missing required columns: {missing}")
    normalized["nx"] = normalized["nx"].astype(int)
    normalized["ny"] = normalized["ny"].astype(int)
    return normalized


def _parse_kma_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    date_text = date_series.astype(str).str.replace("-", "", regex=False).str.zfill(8)
    time_text = time_series.astype(str).str.replace(":", "", regex=False).str.zfill(4).str[:4]
    local = pd.to_datetime(date_text + time_text, format="%Y%m%d%H%M", errors="raise")
    return local.dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")


def _parse_kma_value(category: str, value: Any) -> float:
    text = str(value).strip()
    if text in {"강수없음", "적설없음", "없음", "-"}:
        return 0.0
    cleaned = text.replace("mm", "").replace("cm", "").replace("이상", "").strip()
    if cleaned.startswith("1mm 미만") or cleaned.startswith("1.0mm 미만"):
        return 0.0
    range_match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*[~\-]\s*(-?\d+(?:\.\d+)?)\s*", cleaned)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2.0
    try:
        return float(cleaned)
    except ValueError:
        number = "".join(ch for ch in cleaned if ch.isdigit() or ch in ".-")
        return float(number) if number else float("nan")


def _sky_code_to_cloud_cover(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return {1: 0.0, 3: 75.0, 4: 100.0}.get(int(float(value)), float(value))


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert KMA short-term forecast CSV to prepared forecast archive schema.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--station-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mapping-output", default="data/raw/nwp/archive/station_forecast_grid_mapping.csv")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    raw = pd.read_csv(args.input)
    stations = pd.read_csv(args.station_metadata)
    prepared = build_kma_prepared_forecast(raw, stations, mapping_output=args.mapping_output)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()

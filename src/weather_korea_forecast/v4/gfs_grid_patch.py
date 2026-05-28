from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from weather_korea_forecast.v4.patch_extraction import extract_nwp_patches, patches_to_feature_table

GFS_MESSAGE_COLUMNS = {
    ("2t", "heightAboveGround", 2): "nwp_t2m",
    ("2d", "heightAboveGround", 2): "nwp_dew_point",
    ("2r", "heightAboveGround", 2): "nwp_humidity",
    ("sp", "surface", 0): "nwp_sp",
    ("10u", "heightAboveGround", 10): "nwp_u10",
    ("10v", "heightAboveGround", 10): "nwp_v10",
    ("tp", "surface", 0): "nwp_tp",
    ("tcc", "atmosphere", 0): "nwp_cloud_cover",
}
PATCH_FEATURE_MODE = "true_gfs_grid_patch"


def grib_to_grid_table(
    grib_path: str | Path,
    *,
    source: str = "gfs_forecast",
    variable_columns: dict[tuple[str, str, int], str] | None = None,
) -> pd.DataFrame:
    """Read a GFS GRIB subset into a compact grid table for patch extraction.

    The function accepts either NOMADS regional subset files or whole-message
    files downloaded from the NOAA GFS public S3 bucket. It keeps only surface
    variables needed by the operational MOS path and infers issue/lead metadata
    from the repository's `gfs_YYYYMMDD_CC_fFFF_*.grib2` cache naming contract.
    """

    path = Path(grib_path)
    issue_time, horizon_step = parse_gfs_grib_filename(path)
    valid_time = issue_time + pd.Timedelta(hours=horizon_step)
    rows_by_cell: dict[tuple[float, float], dict[str, Any]] = {}
    mapping = variable_columns or GFS_MESSAGE_COLUMNS
    try:
        import eccodes
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("GFS grid patch extraction requires the optional eccodes package.") from exc

    with path.open("rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                break
            try:
                short = str(eccodes.codes_get(gid, "shortName"))
                level_type = str(eccodes.codes_get(gid, "typeOfLevel"))
                level = int(eccodes.codes_get(gid, "level"))
                column = mapping.get((short, level_type, level))
                if column is None:
                    continue
                if column == "nwp_tp" and any(column in row for row in rows_by_cell.values()):
                    continue
                latitudes = np.asarray(eccodes.codes_get_array(gid, "latitudes"), dtype=float)
                longitudes = np.asarray(eccodes.codes_get_array(gid, "longitudes"), dtype=float)
                values = np.asarray(eccodes.codes_get_array(gid, "values"), dtype=float)
                longitudes = np.where(longitudes > 180.0, longitudes - 360.0, longitudes)
                for lat, lon, value in zip(latitudes, longitudes, values, strict=True):
                    key = (round(float(lat), 6), round(float(lon), 6))
                    row = rows_by_cell.setdefault(
                        key,
                        {
                            "forecast_init_time": issue_time,
                            "issue_time": issue_time,
                            "valid_time": valid_time,
                            "horizon_step": int(horizon_step),
                            "source": source,
                            "lat": float(lat),
                            "lon": float(lon),
                        },
                    )
                    row[column] = normalize_gfs_grid_value(column, float(value))
            finally:
                eccodes.codes_release(gid)
    if not rows_by_cell:
        raise ValueError(f"No supported surface GFS messages found in {path}.")
    return pd.DataFrame(rows_by_cell.values()).sort_values(["lat", "lon"]).reset_index(drop=True)


def parse_gfs_grib_filename(path: str | Path) -> tuple[pd.Timestamp, int]:
    name = Path(path).name
    parts = name.split("_")
    if len(parts) < 4 or parts[0] != "gfs":
        raise ValueError(f"Cannot infer GFS issue/lead metadata from filename: {name}")
    date = parts[1]
    cycle = parts[2]
    lead = int(parts[3].removeprefix("f")[:3])
    return pd.to_datetime(f"{date}{cycle}", format="%Y%m%d%H", utc=True), lead


def normalize_gfs_grid_value(column: str, value: float) -> float:
    if column in {"nwp_t2m", "nwp_dew_point"} and value > 150.0:
        return float(value - 273.15)
    if column == "nwp_sp" and value > 2000.0:
        return float(value / 100.0)
    return float(value)


def extract_true_gfs_grid_patch_features(
    grid_frames: Iterable[pd.DataFrame],
    stations: pd.DataFrame,
    *,
    patch_size: int,
    variables: Sequence[str] | None = None,
) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []
    for grid in grid_frames:
        patches = extract_nwp_patches(
            grid,
            stations,
            variables=variables,
            patch_size=patch_size,
            source=PATCH_FEATURE_MODE,
        )
        features = patches_to_feature_table(patches)
        if not features.empty:
            features["patch_feature_mode"] = PATCH_FEATURE_MODE
            feature_frames.append(features)
    if not feature_frames:
        return pd.DataFrame()
    return pd.concat(feature_frames, ignore_index=True)


def extract_true_gfs_grid_patch_features_from_gribs(
    grib_paths: Iterable[str | Path],
    stations: pd.DataFrame,
    *,
    patch_size: int,
    variables: Sequence[str] | None = None,
) -> pd.DataFrame:
    feature_frames = [
        grib_to_patch_feature_table(path, stations, patch_size=patch_size, variables=variables)
        for path in grib_paths
    ]
    feature_frames = [frame for frame in feature_frames if not frame.empty]
    return pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()


def grib_to_patch_feature_table(
    grib_path: str | Path,
    stations: pd.DataFrame,
    *,
    patch_size: int,
    variables: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Extract patch summaries directly from a GRIB file without materializing the full grid table."""

    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError("patch_size must be a positive odd integer.")
    path = Path(grib_path)
    issue_time, horizon_step = parse_gfs_grib_filename(path)
    valid_time = issue_time + pd.Timedelta(hours=horizon_step)
    wanted = set(variables or [])
    station_rows = _station_rows(stations)
    rows: dict[str, dict[str, Any]] = {
        sid: {
            "station_id": sid,
            "forecast_init_time": issue_time,
            "valid_time": valid_time,
            "horizon_step": int(horizon_step),
            "source": "gfs_forecast",
            "patch_size": int(patch_size),
        }
        for sid in station_rows
    }
    try:
        import eccodes
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("GFS grid patch extraction requires the optional eccodes package.") from exc

    with path.open("rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                break
            try:
                short = str(eccodes.codes_get(gid, "shortName"))
                level_type = str(eccodes.codes_get(gid, "typeOfLevel"))
                level = int(eccodes.codes_get(gid, "level"))
                column = GFS_MESSAGE_COLUMNS.get((short, level_type, level))
                if column is None or (wanted and column not in wanted):
                    continue
                if column == "nwp_tp" and any(f"{column}_patch_center" in row for row in rows.values()):
                    continue
                ni = int(eccodes.codes_get(gid, "Ni"))
                nj = int(eccodes.codes_get(gid, "Nj"))
                values = np.asarray(eccodes.codes_get_array(gid, "values"), dtype=float).reshape(nj, ni)
                lat_axis, lon_axis = _regular_lat_lon_axes(gid, ni=ni, nj=nj)
                for sid, station in station_rows.items():
                    center_row = int(np.abs(lat_axis - station["lat"]).argmin())
                    center_col = int(np.abs(lon_axis - station["lon"]).argmin())
                    patch = _constant_shape_patch(values, center_row=center_row, center_col=center_col, patch_size=patch_size)
                    _append_patch_summary(rows[sid], column, patch)
            finally:
                eccodes.codes_release(gid)
    frame = pd.DataFrame(rows.values())
    if not frame.empty:
        frame["patch_feature_mode"] = PATCH_FEATURE_MODE
    return frame


def write_true_gfs_grid_patch_features(
    grib_paths: Iterable[str | Path],
    station_metadata_csv: str | Path,
    output_csv: str | Path,
    *,
    patch_size: int,
    variables: Sequence[str] | None = None,
) -> Path:
    stations = pd.read_csv(station_metadata_csv, dtype={"station_id": str})
    features = extract_true_gfs_grid_patch_features_from_gribs(
        grib_paths,
        stations,
        patch_size=patch_size,
        variables=variables,
    )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    return output


def _station_rows(stations: pd.DataFrame) -> dict[str, dict[str, float]]:
    frame = stations.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    return {
        str(row.station_id): {"lat": float(row.lat), "lon": float(row.lon)}
        for row in frame[["station_id", "lat", "lon"]].dropna().drop_duplicates("station_id").itertuples(index=False)
    }


def _regular_lat_lon_axes(gid: Any, *, ni: int, nj: int) -> tuple[np.ndarray, np.ndarray]:
    """Return regular-lat/lon axes without materializing full lat/lon arrays."""

    import eccodes

    lat0 = float(eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
    lon0 = float(eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"))
    dlat = float(eccodes.codes_get(gid, "jDirectionIncrementInDegrees"))
    dlon = float(eccodes.codes_get(gid, "iDirectionIncrementInDegrees"))
    j_positive = int(eccodes.codes_get(gid, "jScansPositively")) == 1
    i_negative = int(eccodes.codes_get(gid, "iScansNegatively")) == 1
    lat_step = dlat if j_positive else -dlat
    lon_step = -dlon if i_negative else dlon
    lat_axis = lat0 + np.arange(nj, dtype=float) * lat_step
    lon_axis = lon0 + np.arange(ni, dtype=float) * lon_step
    lon_axis = ((lon_axis + 180.0) % 360.0) - 180.0
    return lat_axis, lon_axis


def _constant_shape_patch(values: np.ndarray, *, center_row: int, center_col: int, patch_size: int) -> np.ndarray:
    radius = patch_size // 2
    patch = np.full((patch_size, patch_size), np.nan, dtype=float)
    for out_r, row in enumerate(range(center_row - radius, center_row + radius + 1)):
        if row < 0 or row >= values.shape[0]:
            continue
        for out_c, col in enumerate(range(center_col - radius, center_col + radius + 1)):
            if col < 0 or col >= values.shape[1]:
                continue
            patch[out_r, out_c] = values[row, col]
    return patch


def _append_patch_summary(row: dict[str, Any], variable: str, patch: np.ndarray) -> None:
    prefix = variable
    normalized = np.full_like(patch, np.nan, dtype=float)
    finite_mask = np.isfinite(patch)
    normalized[finite_mask] = [normalize_gfs_grid_value(variable, float(value)) for value in patch[finite_mask]]
    finite = normalized[np.isfinite(normalized)]
    center = normalized[normalized.shape[0] // 2, normalized.shape[1] // 2]
    min_value = float(np.nanmin(normalized)) if finite.size else np.nan
    max_value = float(np.nanmax(normalized)) if finite.size else np.nan
    summary = {
        "center": float(center) if np.isfinite(center) else np.nan,
        "mean": float(np.nanmean(normalized)) if finite.size else np.nan,
        "std": float(np.nanstd(normalized)) if finite.size else np.nan,
        "min": min_value,
        "max": max_value,
        "sum": float(np.nansum(normalized)) if finite.size else np.nan,
        "range": max_value - min_value if finite.size else np.nan,
        "gradient_x": _axis_gradient(normalized, axis=1),
        "gradient_y": _axis_gradient(normalized, axis=0),
        "upwind_mean": _upwind_mean(normalized),
    }
    for name, value in summary.items():
        row[f"{prefix}_patch_{name}"] = value
        row[f"patch_{prefix}_{name}"] = value
    if variable == "nwp_tp":
        row[f"{prefix}_patch_coverage_fraction"] = float((finite > 0.0).mean()) if finite.size else np.nan
        row[f"patch_{prefix}_coverage_fraction"] = row[f"{prefix}_patch_coverage_fraction"]


def _axis_gradient(patch: np.ndarray, *, axis: int) -> float:
    center = patch.shape[axis] // 2
    neg_idx = center - 1
    pos_idx = center + 1
    if neg_idx < 0 or pos_idx >= patch.shape[axis]:
        return np.nan
    negative = patch[:, neg_idx] if axis == 1 else patch[neg_idx, :]
    positive = patch[:, pos_idx] if axis == 1 else patch[pos_idx, :]
    neg = negative[np.isfinite(negative)]
    pos = positive[np.isfinite(positive)]
    if not len(neg) or not len(pos):
        return np.nan
    return float(pos.mean() - neg.mean())


def _upwind_mean(patch: np.ndarray) -> float:
    radius = patch.shape[0] // 2
    values = patch[: radius + 1, : radius + 1]
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.nan
    return float(np.mean(finite))


def _parse_variables(raw: str) -> list[str] | None:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return values or None


def _expand_glob(pattern: str) -> list[Path]:
    return [Path(path) for path in sorted(glob.glob(pattern))]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract true GFS grid patch summary features from GRIB files.")
    parser.add_argument("--grib", action="append", default=[], help="GRIB path; may be passed multiple times.")
    parser.add_argument("--grib-glob", action="append", default=[], help="Glob pattern for GRIB files.")
    parser.add_argument("--station-metadata", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--patch-size", type=int, choices=[3, 5, 7, 9], required=True)
    parser.add_argument("--variables", default="", help="Comma-separated nwp_* variable columns; default all supported numeric columns.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths = [Path(path) for path in args.grib]
    for pattern in args.grib_glob:
        paths.extend(_expand_glob(pattern))
    if not paths:
        raise ValueError("Pass at least one --grib or --grib-glob path.")
    output = write_true_gfs_grid_patch_features(
        paths,
        args.station_metadata,
        args.output_csv,
        patch_size=args.patch_size,
        variables=_parse_variables(args.variables),
    )
    print(output)


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


PATCH_ID_COLUMNS = [
    "station_id",
    "forecast_init_time",
    "valid_time",
    "horizon_step",
    "source",
    "patch_size",
]
PATCH_LONG_COLUMNS = [
    *PATCH_ID_COLUMNS,
    "variable",
    "row_offset",
    "col_offset",
    "value",
    "lat",
    "lon",
]
GRID_METADATA_COLUMNS = {
    "station_id",
    "lat",
    "lon",
    "latitude",
    "longitude",
    "forecast_init_time",
    "issue_time",
    "valid_time",
    "datetime",
    "timestamp",
    "horizon_step",
    "lead_hour",
    "source",
}
PRECIPITATION_VARIABLE_HINTS = ("tp", "precip", "rain", "snow")


def extract_nwp_patches(
    grid: pd.DataFrame,
    stations: pd.DataFrame,
    *,
    variables: Sequence[str] | None = None,
    patch_size: int = 5,
    source: str = "nwp_forecast",
    lat_column: str = "lat",
    lon_column: str = "lon",
    station_lat_column: str = "lat",
    station_lon_column: str = "lon",
) -> pd.DataFrame:
    """Extract centered NWP grid patches around stations into a long schema.

    The returned frame follows the V4 patch contract documented in
    ``docs/V4_IMPLEMENTATION_PLAN.md``:
    ``station_id, forecast_init_time, valid_time, horizon_step, source,
    patch_size, variable, row_offset, col_offset, value, lat, lon``.

    Grid edges are padded with ``NaN`` values instead of shrinking the patch so
    downstream tensor builders can rely on a constant ``patch_size x patch_size``
    shape for every station/time/variable.
    """

    _validate_patch_size(patch_size)
    normalized_grid = _normalize_grid_frame(grid, lat_column=lat_column, lon_column=lon_column, source=source)
    normalized_stations = _normalize_station_frame(
        stations,
        station_lat_column=station_lat_column,
        station_lon_column=station_lon_column,
    )
    selected_variables = _select_variables(normalized_grid, variables)
    radius = patch_size // 2
    offsets = range(-radius, radius + 1)
    rows: list[dict[str, Any]] = []

    for keys, group in normalized_grid.groupby(_grid_group_columns(normalized_grid), dropna=False, sort=True):
        key_values = _group_key_values(_grid_group_columns(normalized_grid), keys)
        lat_values = np.array(sorted(group["lat"].dropna().unique()), dtype=float)
        lon_values = np.array(sorted(group["lon"].dropna().unique()), dtype=float)
        if len(lat_values) == 0 or len(lon_values) == 0:
            continue

        indexed = group.drop_duplicates(["lat", "lon"], keep="last").set_index(["lat", "lon"])
        for station in normalized_stations.itertuples(index=False):
            center_row = int(np.abs(lat_values - float(station.lat)).argmin())
            center_col = int(np.abs(lon_values - float(station.lon)).argmin())
            for variable in selected_variables:
                for row_offset in offsets:
                    grid_row = center_row + row_offset
                    cell_lat = float(lat_values[grid_row]) if 0 <= grid_row < len(lat_values) else np.nan
                    for col_offset in offsets:
                        grid_col = center_col + col_offset
                        cell_lon = float(lon_values[grid_col]) if 0 <= grid_col < len(lon_values) else np.nan
                        value = np.nan
                        if not np.isnan(cell_lat) and not np.isnan(cell_lon):
                            try:
                                value = indexed.at[(cell_lat, cell_lon), variable]
                            except KeyError:
                                value = np.nan
                        rows.append(
                            {
                                "station_id": str(station.station_id),
                                "forecast_init_time": key_values.get("forecast_init_time", pd.NaT),
                                "valid_time": key_values.get("valid_time", pd.NaT),
                                "horizon_step": _coerce_nullable_int(key_values.get("horizon_step")),
                                "source": str(key_values.get("source", source)),
                                "patch_size": int(patch_size),
                                "variable": str(variable),
                                "row_offset": int(row_offset),
                                "col_offset": int(col_offset),
                                "value": float(value) if pd.notna(value) else np.nan,
                                "lat": cell_lat,
                                "lon": cell_lon,
                            }
                        )

    return pd.DataFrame(rows, columns=PATCH_LONG_COLUMNS)


def patches_to_feature_table(
    patches: pd.DataFrame,
    *,
    precipitation_variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Convert long patch rows into one wide summary-feature row per sample.

    Feature names are stable and model-friendly, for example
    ``nwp_t2m_patch_center`` and ``nwp_t2m_patch_mean``.  Precipitation-like
    variables also get ``*_patch_coverage_fraction`` (fraction of finite patch
    cells with value > 0).
    """

    if patches.empty:
        return pd.DataFrame(columns=PATCH_ID_COLUMNS)
    missing = sorted(set(PATCH_LONG_COLUMNS) - set(patches.columns))
    if missing:
        raise ValueError(f"Patch frame is missing required columns: {missing}")

    precip_names = {str(v) for v in precipitation_variables or []}
    output_rows: list[dict[str, Any]] = []
    for keys, sample in patches.groupby(PATCH_ID_COLUMNS, dropna=False, sort=True):
        row = _group_key_values(PATCH_ID_COLUMNS, keys)
        for variable, variable_rows in sample.groupby("variable", sort=True):
            prefix = _safe_feature_prefix(str(variable))
            values = pd.to_numeric(variable_rows["value"], errors="coerce")
            center = variable_rows.loc[
                variable_rows["row_offset"].astype(int).eq(0) & variable_rows["col_offset"].astype(int).eq(0),
                "value",
            ]
            finite = values.dropna()
            patch_center = (
                float(center.iloc[-1]) if not center.empty and pd.notna(center.iloc[-1]) else np.nan
            )
            patch_mean = float(finite.mean()) if not finite.empty else np.nan
            patch_std = float(finite.std(ddof=0)) if not finite.empty else np.nan
            patch_min = float(finite.min()) if not finite.empty else np.nan
            patch_max = float(finite.max()) if not finite.empty else np.nan
            patch_range = patch_max - patch_min if pd.notna(patch_max) and pd.notna(patch_min) else np.nan
            gradient_x = _patch_axis_gradient(variable_rows, axis="col_offset")
            gradient_y = _patch_axis_gradient(variable_rows, axis="row_offset")
            for feature_name, feature_value in {
                "center": patch_center,
                "mean": patch_mean,
                "std": patch_std,
                "min": patch_min,
                "max": patch_max,
                "range": patch_range,
                "gradient_x": gradient_x,
                "gradient_y": gradient_y,
            }.items():
                row[f"{prefix}_patch_{feature_name}"] = feature_value
                row[f"patch_{prefix}_{feature_name}"] = feature_value
            if str(variable) in precip_names or _looks_like_precipitation(str(variable)):
                coverage = float((finite > 0.0).mean()) if not finite.empty else np.nan
                row[f"{prefix}_patch_coverage_fraction"] = coverage
                row[f"patch_{prefix}_coverage_fraction"] = coverage
        output_rows.append(row)
    return pd.DataFrame(output_rows)


def _patch_axis_gradient(variable_rows: pd.DataFrame, *, axis: str) -> float:
    positive = pd.to_numeric(variable_rows.loc[variable_rows[axis].astype(int).eq(1), "value"], errors="coerce").dropna()
    negative = pd.to_numeric(variable_rows.loc[variable_rows[axis].astype(int).eq(-1), "value"], errors="coerce").dropna()
    if positive.empty or negative.empty:
        return np.nan
    return float(positive.mean() - negative.mean())


def _validate_patch_size(patch_size: int) -> None:
    if int(patch_size) != patch_size or patch_size < 1 or patch_size % 2 != 1:
        raise ValueError("patch_size must be a positive odd integer such as 3, 5, or 9.")


def _normalize_grid_frame(grid: pd.DataFrame, *, lat_column: str, lon_column: str, source: str) -> pd.DataFrame:
    if grid.empty:
        raise ValueError("NWP grid frame is empty; cannot extract patches.")
    frame = grid.copy()
    for source_column, target_column in ((lat_column, "lat"), (lon_column, "lon")):
        if source_column not in frame.columns:
            raise ValueError(f"NWP grid frame is missing required column {source_column!r}.")
        if source_column != target_column:
            frame[target_column] = frame[source_column]
    valid_time_column = _first_existing_column(frame, ["valid_time", "datetime", "timestamp"])
    if valid_time_column is None:
        raise ValueError("NWP grid frame requires one of valid_time, datetime, or timestamp.")
    frame["valid_time"] = pd.to_datetime(frame[valid_time_column], utc=True)
    issue_time_column = _first_existing_column(
        frame,
        ["forecast_init_time", "issue_time", "model_run_time", "run_time", "reference_time"],
    )
    if issue_time_column is not None:
        frame["forecast_init_time"] = pd.to_datetime(frame[issue_time_column], utc=True)
    elif "forecast_init_time" not in frame.columns:
        frame["forecast_init_time"] = pd.NaT
    if "horizon_step" in frame.columns:
        frame["horizon_step"] = frame["horizon_step"].astype("Int64")
    elif "lead_hour" in frame.columns:
        frame["horizon_step"] = frame["lead_hour"].astype("Int64")
    elif frame["forecast_init_time"].notna().all():
        lead = (frame["valid_time"] - frame["forecast_init_time"]) / pd.Timedelta(hours=1)
        frame["horizon_step"] = lead.round().astype("Int64")
    else:
        frame["horizon_step"] = pd.Series([pd.NA] * len(frame), dtype="Int64")
    if "source" not in frame.columns:
        frame["source"] = source
    frame["lat"] = frame["lat"].astype(float)
    frame["lon"] = frame["lon"].astype(float)
    return frame


def _normalize_station_frame(
    stations: pd.DataFrame,
    *,
    station_lat_column: str,
    station_lon_column: str,
) -> pd.DataFrame:
    if stations.empty:
        raise ValueError("Station frame is empty; cannot extract patches.")
    frame = stations.copy()
    if "station_id" not in frame.columns:
        raise ValueError("Station frame requires a station_id column.")
    for source_column, target_column in ((station_lat_column, "lat"), (station_lon_column, "lon")):
        if source_column not in frame.columns:
            raise ValueError(f"Station frame is missing required column {source_column!r}.")
        if source_column != target_column:
            frame[target_column] = frame[source_column]
    frame["station_id"] = frame["station_id"].astype(str)
    frame["lat"] = frame["lat"].astype(float)
    frame["lon"] = frame["lon"].astype(float)
    return frame[["station_id", "lat", "lon"]].drop_duplicates("station_id")


def _select_variables(frame: pd.DataFrame, variables: Sequence[str] | None) -> list[str]:
    if variables is not None:
        selected = [str(variable) for variable in variables]
        missing = sorted(set(selected) - set(frame.columns))
        if missing:
            raise ValueError(f"NWP grid frame is missing requested patch variables: {missing}")
        return selected
    selected = [
        column
        for column in frame.columns
        if column not in GRID_METADATA_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not selected:
        raise ValueError("No numeric patch variables were found; pass variables=[...].")
    return selected


def _grid_group_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["forecast_init_time", "valid_time", "horizon_step", "source"]
    return [column for column in columns if column in frame.columns]


def _group_key_values(columns: Sequence[str], keys: Any) -> dict[str, Any]:
    if len(columns) == 1:
        keys = (keys,)
    return dict(zip(columns, keys))


def _coerce_nullable_int(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    return int(value)


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _safe_feature_prefix(variable: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in variable).strip("_")


def _looks_like_precipitation(variable: str) -> bool:
    lowered = variable.lower()
    return any(hint in lowered for hint in PRECIPITATION_VARIABLE_HINTS)

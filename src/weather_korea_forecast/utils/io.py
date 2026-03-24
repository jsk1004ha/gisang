from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from weather_korea_forecast.utils.paths import resolve_path


def read_table(path_like: str | Path) -> pd.DataFrame:
    path = resolve_path(path_like)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix == ".nc":
        return _read_era5_netcdf_or_archive(path)
    raise ValueError(f"Unsupported table extension: {path.suffix}")


def write_table(df: pd.DataFrame, path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {path.suffix}")
    return path


def _read_era5_netcdf_or_archive(path: Path) -> pd.DataFrame:
    import xarray as xr

    if _is_zip_file(path):
        extract_dir = path.parent / f"{path.stem}_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(path) as archive:
            archive.extractall(extract_dir)
        nc_files = sorted(extract_dir.glob("*.nc"))
        if not nc_files:
            raise ValueError(f"No .nc files found inside archive: {path}")
        frames = [_netcdf_to_frame(nc_path, xr) for nc_path in nc_files]
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["datetime", "lat", "lon"], how="outer")
        return merged.sort_values(["datetime", "lat", "lon"]).reset_index(drop=True)

    return _netcdf_to_frame(path, xr)


def _netcdf_to_frame(path: Path, xr_module) -> pd.DataFrame:
    ds = xr_module.open_dataset(path, engine="netcdf4")
    rename_dims = {}
    if "valid_time" in ds.coords:
        rename_dims["valid_time"] = "datetime"
    if "latitude" in ds.coords:
        rename_dims["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_dims["longitude"] = "lon"
    ds = ds.rename(rename_dims)
    drop_columns = [column for column in ("expver", "number") if column in ds.coords]
    if drop_columns:
        ds = ds.drop_vars(drop_columns)

    rename_vars = {"t2m": "era5_t2m", "sp": "era5_sp", "u10": "era5_u10", "v10": "era5_v10", "tp": "era5_tp"}
    ds = ds.rename({key: value for key, value in rename_vars.items() if key in ds.data_vars})
    frame = ds.to_dataframe().reset_index()
    return frame


def _is_zip_file(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == b"PK\x03\x04"


def write_json(payload: Any, path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)
    return path

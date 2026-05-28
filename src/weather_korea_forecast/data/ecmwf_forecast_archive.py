from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from weather_korea_forecast.data.gfs_forecast_archive import build_gfs_prepared_forecast


def build_ecmwf_prepared_forecast(raw_forecast: pd.DataFrame, station_metadata: pd.DataFrame) -> pd.DataFrame:
    """Skeleton ECMWF adapter using the same local-table contract as GFS.

    Live ECMWF Open Data retrieval is intentionally not implemented here; use a
    local extracted station/grid table and convert it through this stable seam.
    """
    prepared = build_gfs_prepared_forecast(raw_forecast, station_metadata).copy()
    prepared["source"] = "ecmwf_forecast"
    return prepared


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert local ECMWF extracted forecast table to prepared forecast archive schema (skeleton).")
    parser.add_argument("--input", required=True)
    parser.add_argument("--station-metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    prepared = build_ecmwf_prepared_forecast(pd.read_csv(args.input), pd.read_csv(args.station_metadata))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()

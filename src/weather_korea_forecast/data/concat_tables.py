from __future__ import annotations

import argparse

import pandas as pd

from weather_korea_forecast.utils.io import read_table, write_table


def concat_tables(input_paths: list[str], output_path: str, sort_columns: list[str] | None = None) -> None:
    frames = [read_table(path) for path in input_paths]
    merged = pd.concat(frames, ignore_index=True)
    if sort_columns:
        available = [column for column in sort_columns if column in merged.columns]
        if available:
            merged = merged.sort_values(available).reset_index(drop=True)
    write_table(merged, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate multiple CSV/parquet/netCDF-derived tables.")
    parser.add_argument("--input-path", action="append", required=True, dest="input_paths")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--sort-column", action="append", default=[])
    args = parser.parse_args()

    concat_tables(args.input_paths, args.output_path, sort_columns=args.sort_column or None)


if __name__ == "__main__":
    main()

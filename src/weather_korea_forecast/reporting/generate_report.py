from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from weather_korea_forecast.reporting.collect_experiments import (
    collect_all,
    main_leaderboard_records,
    write_best_models_csv,
    write_failed_csv,
    write_summary_csv,
    write_summary_json,
)
from weather_korea_forecast.reporting.html_template import render_report
from weather_korea_forecast.reporting.schema import ExperimentRecord


def build_report(
    *,
    experiments_root: Path,
    output_dir: Path,
    title: str = "기상 V1-V3 실험 리포트",
    include_images: bool = True,
    embed_images: str = "full",
    max_experiments: int | None = None,
    track_filter: str | None = None,
    target_filter: str | None = None,
    sort_by: str = "rmse",
) -> dict[str, Path | int | float | None]:
    experiments_root = Path(experiments_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_all(experiments_root)
    records = _apply_filters(records, track_filter=track_filter, target_filter=target_filter)
    records = _sort(records, sort_by)
    if max_experiments is not None:
        records = records[: max(0, max_experiments)]

    summary_csv = write_summary_csv(records, output_dir / "experiment_summary.csv")
    summary_json = write_summary_json(records, output_dir / "experiment_summary.json")
    best_csv = write_best_models_csv(records, output_dir / "best_models.csv")
    failed_csv = write_failed_csv(records, output_dir / "failed_or_incomplete_experiments.csv")
    html_path = output_dir / "experiment_report.html"
    image_mode = embed_images if include_images else "none"
    html_path.write_text(
        render_report(records, title=title, experiments_root=experiments_root, include_images=include_images, image_mode=image_mode),
        encoding="utf-8",
    )
    complete = [r for r in records if r.complete and r.rmse is not None]
    main_records = main_leaderboard_records(records)
    return {
        "total_experiments": len(records),
        "complete_experiments": len(complete),
        "incomplete_experiments": len([r for r in records if not r.complete or r.error]),
        "main_leaderboard_experiments": len(main_records),
        "diagnostic_experiments": len([r for r in records if r.is_diagnostic]),
        "best_temp_rmse": _best_rmse(main_records, target="temp"),
        "best_humidity_rmse": _best_rmse(main_records, target="humidity"),
        "report_path": html_path,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "best_models_csv": best_csv,
        "failed_csv": failed_csv,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a standalone HTML/CSV report for Gisang experiments.")
    parser.add_argument("--experiments-root", type=Path, default=Path("data/artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--title", default="기상 V1-V3 실험 리포트")
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument("--include-images", action="store_true", default=True)
    image_group.add_argument("--no-images", action="store_false", dest="include_images")
    parser.add_argument(
        "--embed-images",
        choices=["full", "thumbnail", "external-assets"],
        default="full",
        help="HTML plot handling: full base64 embed, thumbnail base64 links, or external image paths.",
    )
    parser.add_argument("--max-experiments", type=int)
    parser.add_argument("--track-filter")
    parser.add_argument("--target-filter")
    parser.add_argument("--sort-by", default="rmse")
    parser.add_argument("--open-after-build", action="store_true")
    args = parser.parse_args(argv)

    result = build_report(
        experiments_root=args.experiments_root,
        output_dir=args.output_dir,
        title=args.title,
        include_images=args.include_images,
        embed_images=args.embed_images,
        max_experiments=args.max_experiments,
        track_filter=args.track_filter,
        target_filter=args.target_filter,
        sort_by=args.sort_by,
    )
    print(f"total experiments: {result['total_experiments']}")
    print(f"complete experiments: {result['complete_experiments']}")
    print(f"incomplete experiments: {result['incomplete_experiments']}")
    print(f"main leaderboard experiments: {result['main_leaderboard_experiments']}")
    print(f"diagnostic experiments: {result['diagnostic_experiments']}")
    print(f"best temp RMSE: {_fmt(result['best_temp_rmse'])}")
    print(f"best humidity RMSE: {_fmt(result['best_humidity_rmse'])}")
    print(f"report path: {result['report_path']}")
    if args.open_after_build:
        _open_path(Path(result["report_path"]))


def _apply_filters(records: list[ExperimentRecord], *, track_filter: str | None, target_filter: str | None) -> list[ExperimentRecord]:
    output = records
    if track_filter:
        output = [r for r in output if r.track == track_filter]
    if target_filter:
        output = [r for r in output if r.target_name == target_filter]
    return output


def _sort(records: list[ExperimentRecord], sort_by: str) -> list[ExperimentRecord]:
    def key(record: ExperimentRecord):
        value = getattr(record, sort_by, None)
        if isinstance(value, (int, float)) and pd.notna(value):
            return (0, float(value))
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return (1, "")
        return (0, str(value))
    return sorted(records, key=key)


def _best_rmse(records: list[ExperimentRecord], *, target: str) -> float | None:
    values = [r.rmse for r in records if r.target_name == target and r.rmse is not None]
    return min(values) if values else None


def _fmt(value: object) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _open_path(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:
        print(f"Could not open report automatically: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()

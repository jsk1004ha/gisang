from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from weather_korea_forecast.evaluation.plots import plot_forecast_vs_actual
from weather_korea_forecast.evaluation.regional_report import build_breakdown_reports
from weather_korea_forecast.evaluation.weatherbenchx_adapter import to_weatherbenchx_sparse_frame
from weather_korea_forecast.training.metrics import compute_point_metrics
from weather_korea_forecast.utils.io import read_table, write_json, write_table
from weather_korea_forecast.utils.paths import resolve_path


def evaluate_experiment(experiment_dir: str | Path) -> dict[str, object]:
    experiment_path = resolve_path(experiment_dir)
    predictions = read_table(experiment_path / "predictions_test.csv")
    overall_metrics = compute_point_metrics(predictions["actual"].to_numpy(), predictions["prediction"].to_numpy())
    breakdowns = build_breakdown_reports(predictions)
    metrics_path = write_json(overall_metrics, experiment_path / "metrics_summary.json")
    for name, report in breakdowns.items():
        write_table(report, experiment_path / f"metrics_{name}.csv")
    write_table(to_weatherbenchx_sparse_frame(predictions), experiment_path / "weatherbenchx_sparse.csv")
    plot_forecast_vs_actual(predictions, experiment_path / "forecast_vs_actual.png")
    return {
        "metrics": overall_metrics,
        "metrics_path": str(metrics_path),
        "breakdown_names": sorted(breakdowns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved experiment directory.")
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()
    summary = evaluate_experiment(args.experiment_dir)
    print(summary)


if __name__ == "__main__":
    main()

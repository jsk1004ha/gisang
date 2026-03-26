from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from weather_korea_forecast.evaluation.plots import plot_forecast_vs_actual
from weather_korea_forecast.training.metrics import compute_group_metrics, compute_prediction_metrics
from weather_korea_forecast.utils.io import read_table, write_json, write_table
from weather_korea_forecast.utils.paths import resolve_path


def evaluate_prediction_frame(predictions: pd.DataFrame, experiment_dir: str | Path) -> dict[str, object]:
    experiment_path = resolve_path(experiment_dir)
    metrics = compute_prediction_metrics(predictions)
    raw_metrics = None
    if "prediction_raw" in predictions.columns:
        raw_metrics = compute_prediction_metrics(predictions.rename(columns={"prediction_raw": "_prediction_raw"}), predicted_column="_prediction_raw")
    metrics_payload = {"metrics": metrics, "raw_metrics": raw_metrics}
    write_json(metrics_payload, experiment_path / "metrics_test.json")
    write_json(metrics, experiment_path / "metrics_summary.json")

    for name, report in build_v2_breakdown_reports(predictions).items():
        write_table(report, experiment_path / f"metrics_{name}.csv")

    worst_cases = predictions.assign(abs_error=lambda df: (df["prediction"] - df["actual"]).abs()).sort_values("abs_error", ascending=False).head(100)
    write_table(worst_cases, experiment_path / "worst_case_samples.csv")
    plot_forecast_vs_actual(predictions, experiment_path / "forecast_vs_actual.png")
    return {"metrics": metrics, "raw_metrics": raw_metrics}


def evaluate_experiment(experiment_dir: str | Path) -> dict[str, object]:
    experiment_path = resolve_path(experiment_dir)
    predictions = read_table(experiment_path / "predictions_test.csv")
    summary = evaluate_prediction_frame(predictions, experiment_path)
    return {
        "metrics": summary["metrics"],
        "raw_metrics": summary["raw_metrics"],
        "experiment_dir": str(experiment_path),
    }


def build_v2_breakdown_reports(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    normalized = predictions.copy()
    if "target_name" not in normalized.columns:
        normalized["target_name"] = "unknown"
    if "region" not in normalized.columns and "region_class" in normalized.columns:
        normalized["region"] = normalized["region_class"]

    reports: dict[str, pd.DataFrame] = {}
    group_sets = [
        ["target_name"],
        ["target_name", "horizon_step"],
        ["target_name", "station_id"],
        ["target_name", "region"],
        ["target_name", "season"],
    ]
    for group_columns in group_sets:
        if all(column in normalized.columns for column in group_columns):
            reports["_".join(group_columns)] = compute_group_metrics(normalized, group_columns)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a V2 experiment directory.")
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()
    summary = evaluate_experiment(args.experiment_dir)
    print(summary)


if __name__ == "__main__":
    main()

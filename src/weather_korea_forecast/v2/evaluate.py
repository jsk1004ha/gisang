from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    for name, report in build_v2_raw_breakdown_reports(predictions).items():
        write_table(report, experiment_path / f"metrics_raw_{name}.csv")

    worst_cases = predictions.assign(abs_error=lambda df: (df["prediction"] - df["actual"]).abs()).sort_values("abs_error", ascending=False).head(100)
    write_table(worst_cases, experiment_path / "worst_case_samples.csv")
    plot_forecast_vs_actual(predictions, experiment_path / "forecast_vs_actual.png")
    plot_horizon_error(predictions, experiment_path / "horizon_error.png")
    plot_prediction_scatter(predictions, experiment_path / "prediction_scatter.png")
    plot_raw_vs_corrected(predictions, experiment_path / "raw_vs_corrected.png")
    rolling_origin = build_rolling_origin_reports(predictions)
    for name, report in rolling_origin.items():
        write_table(report, experiment_path / f"metrics_{name}.csv")
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


def build_v2_raw_breakdown_reports(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "prediction_raw" not in predictions.columns:
        return {}
    raw_frame = predictions.rename(columns={"prediction_raw": "_prediction_raw"})
    reports: dict[str, pd.DataFrame] = {}
    group_sets = [
        ["target_name"],
        ["target_name", "horizon_step"],
        ["target_name", "station_id"],
        ["target_name", "region"],
        ["target_name", "season"],
    ]
    for group_columns in group_sets:
        if all(column in raw_frame.columns for column in group_columns):
            reports["_".join(group_columns)] = compute_group_metrics(raw_frame, group_columns, predicted_column="_prediction_raw")
    return reports


def build_rolling_origin_reports(predictions: pd.DataFrame, num_folds: int = 3) -> dict[str, pd.DataFrame]:
    if "prediction_start" not in predictions.columns or predictions.empty:
        return {}
    origin_frame = predictions.copy()
    unique_origins = sorted(pd.to_datetime(origin_frame["prediction_start"], utc=True).drop_duplicates().tolist())
    if len(unique_origins) < num_folds:
        num_folds = len(unique_origins)
    if num_folds <= 1:
        return {}
    origin_series = pd.Series(unique_origins, dtype="datetime64[ns, UTC]")
    fold_labels = np.array_split(origin_series.index.to_numpy(), num_folds)
    mapping_rows: list[dict[str, object]] = []
    for fold_index, origin_indices in enumerate(fold_labels, start=1):
        for origin_index in origin_indices.tolist():
            mapping_rows.append({"prediction_start": origin_series.iloc[int(origin_index)], "rolling_origin_fold": f"fold{fold_index}"})
    mapping = pd.DataFrame(mapping_rows)
    origin_frame["prediction_start"] = pd.to_datetime(origin_frame["prediction_start"], utc=True)
    origin_frame = origin_frame.merge(mapping, on="prediction_start", how="left")
    return {
        "target_name_rolling_origin_fold": compute_group_metrics(origin_frame, ["target_name", "rolling_origin_fold"])
        if {"target_name", "rolling_origin_fold"}.issubset(origin_frame.columns)
        else pd.DataFrame()
    }


def plot_horizon_error(predictions: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    horizon_metrics = compute_group_metrics(predictions.assign(error=lambda df: df["prediction"] - df["actual"]), ["horizon_step"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(horizon_metrics["horizon_step"], horizon_metrics["rmse"], label="RMSE")
    axes[0].plot(horizon_metrics["horizon_step"], horizon_metrics["mae"], label="MAE")
    axes[0].set_title("Horizon Error")
    axes[0].legend()
    axes[1].plot(horizon_metrics["horizon_step"], horizon_metrics["bias"], color="tab:red", label="Bias")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Horizon Bias")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_prediction_scatter(predictions: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5))
    sample = predictions.head(1000)
    axis.scatter(sample["actual"], sample["prediction"], s=10, alpha=0.5)
    min_value = float(min(sample["actual"].min(), sample["prediction"].min()))
    max_value = float(max(sample["actual"].max(), sample["prediction"].max()))
    axis.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1.0)
    axis.set_xlabel("Actual")
    axis.set_ylabel("Prediction")
    axis.set_title("Actual vs Prediction")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_raw_vs_corrected(predictions: pd.DataFrame, output_path: str | Path) -> Path | None:
    if "prediction_raw" not in predictions.columns:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 4))
    sample = predictions.head(200)
    axis.plot(sample["valid_time"], sample["actual"], label="actual")
    axis.plot(sample["valid_time"], sample["prediction_raw"], label="raw")
    axis.plot(sample["valid_time"], sample["prediction"], label="corrected")
    axis.tick_params(axis="x", rotation=30)
    axis.legend()
    axis.set_title("Raw vs Corrected Prediction")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a V2 experiment directory.")
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()
    summary = evaluate_experiment(args.experiment_dir)
    print(summary)


if __name__ == "__main__":
    main()

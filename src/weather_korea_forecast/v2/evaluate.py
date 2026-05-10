from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
    write_json(build_worst_case_summary(predictions), experiment_path / "worst_case_summary.json")
    daily_reports = build_daily_temperature_reports(predictions)
    for name, report in daily_reports.items():
        write_table(report, experiment_path / f"{name}.csv")
    plot_forecast_vs_actual(predictions, experiment_path / "forecast_vs_actual.png")
    plot_horizon_error(predictions, experiment_path / "horizon_error.png")
    plot_prediction_scatter(predictions, experiment_path / "prediction_scatter.png")
    plot_raw_vs_corrected(predictions, experiment_path / "raw_vs_corrected.png")
    plot_horizon_station_heatmap(predictions, experiment_path / "horizon_station_heatmap.png")
    plot_group_rmse_bar(predictions, "station_id", experiment_path / "station_rmse_bar.png", title="Station RMSE")
    plot_group_rmse_bar(predictions, "region", experiment_path / "region_rmse_bar.png", title="Region RMSE")
    plot_daily_max_min_error(daily_reports.get("daily_temperature_errors", pd.DataFrame()), experiment_path / "daily_max_min_error.png")
    plot_extreme_temperature_scatter(predictions, experiment_path / "extreme_temperature_scatter.png")
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


def build_daily_temperature_reports(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if predictions.empty or "valid_time" not in predictions.columns:
        return {}
    frame = predictions.copy()
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
    frame["valid_date"] = frame["valid_time"].dt.date.astype(str)
    group_columns = [column for column in ["target_name", "station_id", "valid_date"] if column in frame.columns]
    if "station_id" not in group_columns:
        group_columns.append("valid_date")
    grouped = frame.groupby(group_columns, dropna=False)
    rows: list[dict[str, object]] = []
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: value for column, value in zip(group_columns, group_key)}
        actual = group["actual"].astype(float)
        prediction = group["prediction"].astype(float)
        row.update(
            {
                "actual_max": float(actual.max()),
                "prediction_max": float(prediction.max()),
                "max_error": float(prediction.max() - actual.max()),
                "actual_min": float(actual.min()),
                "prediction_min": float(prediction.min()),
                "min_error": float(prediction.min() - actual.min()),
                "actual_diurnal_range": float(actual.max() - actual.min()),
                "prediction_diurnal_range": float(prediction.max() - prediction.min()),
                "diurnal_range_error": float((prediction.max() - prediction.min()) - (actual.max() - actual.min())),
                "sample_count": int(len(group)),
            }
        )
        rows.append(row)
    errors = pd.DataFrame(rows)
    if errors.empty:
        return {}
    metric_rows = []
    for metric_name, error_column in (
        ("daily_max_temp", "max_error"),
        ("daily_min_temp", "min_error"),
        ("diurnal_range", "diurnal_range_error"),
    ):
        values = errors[error_column].astype(float).to_numpy()
        metric_rows.append(
            {
                "metric": metric_name,
                "rmse": float(np.sqrt(np.mean(np.square(values)))),
                "mae": float(np.mean(np.abs(values))),
                "bias": float(np.mean(values)),
                "sample_count": int(len(values)),
            }
        )
    return {
        "daily_temperature_errors": errors,
        "metrics_daily_temperature": pd.DataFrame(metric_rows),
    }


def build_worst_case_summary(predictions: pd.DataFrame) -> dict[str, object]:
    if predictions.empty:
        return {"sample_count": 0}
    frame = predictions.copy()
    frame["abs_error"] = (frame["prediction"].astype(float) - frame["actual"].astype(float)).abs()
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True) if "valid_time" in frame.columns else pd.NaT
    frame["hour_of_day"] = frame["valid_time"].dt.hour if "valid_time" in frame.columns else pd.NA
    summary: dict[str, object] = {
        "sample_count": int(len(frame)),
        "max_abs_error": float(frame["abs_error"].max()),
    }
    for column in ["horizon_step", "station_id", "region", "season", "hour_of_day"]:
        if column not in frame.columns:
            continue
        metrics = compute_group_metrics(frame, [column])
        if metrics.empty or "rmse" not in metrics.columns:
            continue
        worst = metrics.sort_values("rmse", ascending=False).iloc[0].to_dict()
        summary[f"worst_{column}"] = _json_safe_row(worst)
    return summary


def _json_safe_row(row: dict[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in row.items():
        if pd.isna(value):
            safe[key] = None
        elif isinstance(value, (np.integer,)):
            safe[key] = int(value)
        elif isinstance(value, (np.floating,)):
            safe[key] = float(value)
        else:
            safe[key] = value
    return safe


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


def plot_horizon_station_heatmap(predictions: pd.DataFrame, output_path: str | Path) -> Path | None:
    if predictions.empty or not {"station_id", "horizon_step"}.issubset(predictions.columns):
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = compute_group_metrics(predictions, ["station_id", "horizon_step"])
    if metrics.empty:
        return None
    heatmap = metrics.pivot(index="station_id", columns="horizon_step", values="rmse").sort_index()
    figure, axis = plt.subplots(figsize=(max(8, heatmap.shape[1] * 0.35), max(4, heatmap.shape[0] * 0.35)))
    image = axis.imshow(heatmap.to_numpy(dtype=float), aspect="auto", cmap="magma")
    axis.set_xticks(range(len(heatmap.columns)))
    axis.set_xticklabels([str(column) for column in heatmap.columns], rotation=90)
    axis.set_yticks(range(len(heatmap.index)))
    axis.set_yticklabels([str(index) for index in heatmap.index])
    axis.set_xlabel("Horizon step")
    axis.set_ylabel("Station")
    axis.set_title("Station x Horizon RMSE")
    figure.colorbar(image, ax=axis, label="RMSE")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_group_rmse_bar(predictions: pd.DataFrame, group_column: str, output_path: str | Path, title: str) -> Path | None:
    if predictions.empty:
        return None
    normalized = predictions.copy()
    if group_column not in normalized.columns:
        if group_column == "region" and "region_class" in normalized.columns:
            normalized["region"] = normalized["region_class"]
        else:
            return None
    metrics = compute_group_metrics(normalized, [group_column]).sort_values("rmse", ascending=False)
    if metrics.empty:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(6, len(metrics) * 0.6), 4))
    axis.bar(metrics[group_column].astype(str), metrics["rmse"].astype(float))
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylabel("RMSE")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_daily_max_min_error(daily_errors: pd.DataFrame, output_path: str | Path) -> Path | None:
    if daily_errors.empty or "valid_date" not in daily_errors.columns:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = daily_errors.sort_values("valid_date").copy()
    plot_frame["label"] = plot_frame["valid_date"].astype(str)
    if "station_id" in plot_frame.columns and plot_frame["station_id"].nunique() > 1:
        plot_frame = plot_frame.groupby("valid_date", as_index=False)[["max_error", "min_error", "diurnal_range_error"]].mean()
        plot_frame["label"] = plot_frame["valid_date"].astype(str)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(plot_frame["label"], plot_frame["max_error"], marker="o", label="daily max error")
    axis.plot(plot_frame["label"], plot_frame["min_error"], marker="o", label="daily min error")
    axis.plot(plot_frame["label"], plot_frame["diurnal_range_error"], marker="o", label="diurnal range error")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    axis.set_title("Daily Temperature Error")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_extreme_temperature_scatter(predictions: pd.DataFrame, output_path: str | Path) -> Path | None:
    if predictions.empty:
        return None
    actual = predictions["actual"].astype(float)
    lower = actual.quantile(0.1)
    upper = actual.quantile(0.9)
    extreme = predictions.loc[(actual <= lower) | (actual >= upper)].copy()
    if extreme.empty:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(extreme["actual"], extreme["prediction"], s=12, alpha=0.6)
    min_value = float(min(extreme["actual"].min(), extreme["prediction"].min()))
    max_value = float(max(extreme["actual"].max(), extreme["prediction"].max()))
    axis.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1.0)
    axis.set_xlabel("Actual")
    axis.set_ylabel("Prediction")
    axis.set_title("Extreme Actual Quantiles Scatter")
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

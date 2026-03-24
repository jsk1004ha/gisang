from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from weather_korea_forecast.data.build_training_table import build_training_table
from weather_korea_forecast.data.dataset_tft import build_dataset_bundle
from weather_korea_forecast.evaluation.plots import plot_forecast_vs_actual
from weather_korea_forecast.evaluation.regional_report import build_breakdown_reports
from weather_korea_forecast.models.baselines import PersistenceBaseline
from weather_korea_forecast.models.registry import build_model
from weather_korea_forecast.training.metrics import compute_point_metrics
from weather_korea_forecast.utils.config import dump_yaml, load_yaml
from weather_korea_forecast.utils.io import read_table, write_json, write_table
from weather_korea_forecast.utils.logger import get_logger
from weather_korea_forecast.utils.paths import ensure_dir, resolve_path, timestamp_slug
from weather_korea_forecast.utils.seed import seed_everything

LOGGER = get_logger(__name__)


def train_experiment(data_config: dict, model_config: dict, train_config: dict) -> Path:
    seed_everything(int(train_config.get("seed", 42)))
    training_table = _load_or_build_training_table(data_config)
    bundle = build_dataset_bundle(training_table, data_config, backend=model_config["model"].get("backend", "fallback_torch"))

    batch_size = int(train_config["training"].get("batch_size", 32))
    num_workers = int(train_config["training"].get("num_workers", 0))
    train_loader = bundle.make_dataloader("train", batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = bundle.make_dataloader("val", batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = bundle.make_dataloader("test", batch_size=batch_size, num_workers=num_workers, shuffle=False)

    experiment_dir = _create_experiment_dir(train_config)
    _snapshot_configs(experiment_dir, data_config, model_config, train_config)

    if model_config["model"]["type"] == "persistence":
        model = PersistenceBaseline(seasonal_period=model_config["model"].get("seasonal_period"))
        test_predictions = _predict_baseline(model, test_loader)
        history = []
    else:
        model = build_model(model_config, bundle)
        train_result = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=int(train_config["training"].get("max_epochs", 5)),
            learning_rate=float(model_config["model"].get("learning_rate", 1e-3)),
            device=train_config["training"].get("device", "cpu"),
            early_stopping_patience=int(train_config["training"].get("early_stopping_patience", 3)),
        )
        history = train_result.history
        model.save(experiment_dir / "model.pt", extra_state={"bundle_metadata": bundle.metadata})
        test_predictions = _predict_trainable_model(model, test_loader)

    test_frame = build_prediction_frame(
        prediction_tensor=test_predictions["prediction"],
        target_tensor=test_predictions["target"],
        metadata=test_predictions["metadata"],
        base_frame=bundle.test_frame,
        bundle=bundle,
    )
    metrics = compute_point_metrics(test_frame["actual"].to_numpy(), test_frame["prediction"].to_numpy())
    write_table(test_frame, experiment_dir / "predictions_test.csv")
    write_json(metrics, experiment_dir / "metrics_test.json")
    write_json(history, experiment_dir / "training_history.json")
    plot_forecast_vs_actual(test_frame, experiment_dir / "forecast_vs_actual.png")
    for name, report in build_breakdown_reports(test_frame).items():
        write_table(report, experiment_dir / f"metrics_{name}.csv")
    _write_experiment_summary(experiment_dir, data_config, model_config, train_config, metrics)
    _refresh_latest_pointer(experiment_dir)
    LOGGER.info("Finished experiment at %s", experiment_dir)
    return experiment_dir


def build_prediction_frame(
    prediction_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    metadata: dict[str, list],
    base_frame: pd.DataFrame,
    bundle,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    region_lookup = base_frame.drop_duplicates("station_id").set_index("station_id").to_dict("index")
    target_column = bundle.target_columns[0]
    for sample_index in range(prediction_tensor.shape[0]):
        station_id = metadata["station_id"][sample_index]
        prediction_start = _ensure_utc_timestamp(metadata["prediction_start"][sample_index])
        station_meta = region_lookup.get(station_id, {})
        for horizon_index in range(prediction_tensor.shape[1]):
            valid_time = prediction_start + pd.Timedelta(hours=horizon_index)
            pred_value = float(prediction_tensor[sample_index, horizon_index, 0].item())
            actual_value = float(target_tensor[sample_index, horizon_index, 0].item())
            if target_column in bundle.scaler.means:
                pred_value = float(bundle.scaler.inverse_values(target_column, pred_value))
                actual_value = float(bundle.scaler.inverse_values(target_column, actual_value))
            rows.append(
                {
                    "station_id": station_id,
                    "prediction_start": prediction_start,
                    "valid_time": valid_time,
                    "horizon_step": horizon_index + 1,
                    "prediction": pred_value,
                    "actual": actual_value,
                    "region": station_meta.get("region", "unknown"),
                    "season": _season_from_timestamp(valid_time),
                }
            )
    return pd.DataFrame(rows)


def _season_from_timestamp(timestamp: pd.Timestamp) -> str:
    month = timestamp.month
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _ensure_utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _load_or_build_training_table(data_config: dict) -> pd.DataFrame:
    output_path = resolve_path(data_config["paths"]["output_training_table"])
    if output_path.exists():
        return read_table(output_path)
    training_table = build_training_table(data_config)
    write_table(training_table, output_path)
    return training_table


def _create_experiment_dir(train_config: dict) -> Path:
    root_dir = ensure_dir(train_config["artifacts"]["root_dir"])
    experiment_name = train_config["experiment"]["name"]
    experiment_dir = root_dir / f"{experiment_name}_{timestamp_slug()}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def _snapshot_configs(experiment_dir: Path, data_config: dict, model_config: dict, train_config: dict) -> None:
    dump_yaml(experiment_dir / "data_config.yaml", data_config)
    dump_yaml(experiment_dir / "model_config.yaml", model_config)
    dump_yaml(experiment_dir / "train_config.yaml", train_config)


def _predict_baseline(model: PersistenceBaseline, loader) -> dict[str, object]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    metadata = {"station_id": [], "prediction_start": []}
    for batch in loader:
        predictions.append(model.predict_batch(batch))
        targets.append(batch["target"])
        metadata["station_id"].extend(batch["station_id"])
        metadata["prediction_start"].extend(batch["prediction_start"])
    return {"prediction": torch.cat(predictions), "target": torch.cat(targets), "metadata": metadata}


def _predict_trainable_model(model, loader) -> dict[str, object]:
    prediction, target, metadata = model.predict_loader(loader)
    return {"prediction": prediction, "target": target, "metadata": metadata}


def _write_experiment_summary(
    experiment_dir: Path,
    data_config: dict,
    model_config: dict,
    train_config: dict,
    metrics: dict[str, float],
) -> None:
    summary = {
        "experiment_name": train_config["experiment"]["name"],
        "dataset_version": Path(data_config["paths"]["output_training_table"]).stem,
        "model_name": model_config["model"]["name"],
        "metrics": metrics,
    }
    write_json(summary, experiment_dir / "experiment_summary.json")


def _refresh_latest_pointer(experiment_dir: Path) -> None:
    latest_dir = experiment_dir.parent / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"latest_experiment": experiment_dir.name}
    with (latest_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    for source_name in (
        "predictions_test.csv",
        "metrics_test.json",
        "experiment_summary.json",
        "forecast_vs_actual.png",
        "data_config.yaml",
        "model_config.yaml",
        "train_config.yaml",
        "model.pt",
        "training_history.json",
    ):
        source = experiment_dir / source_name
        if source.exists():
            target = latest_dir / source_name
            target.write_bytes(source.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a weather forecasting experiment.")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    args = parser.parse_args()

    experiment_dir = train_experiment(
        data_config=load_yaml(args.data_config),
        model_config=load_yaml(args.model_config),
        train_config=load_yaml(args.train_config),
    )
    print(experiment_dir)


if __name__ == "__main__":
    main()

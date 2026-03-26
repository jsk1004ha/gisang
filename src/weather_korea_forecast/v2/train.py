from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from weather_korea_forecast.models.registry import build_model, resolve_model_config
from weather_korea_forecast.training.metrics import compute_prediction_metrics
from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.utils.io import write_json, write_table
from weather_korea_forecast.v2.artifacts import (
    create_experiment_dir,
    refresh_aliases,
    snapshot_config,
    update_leaderboard,
    write_experiment_summary,
    write_feature_importance,
    write_scaler_artifact,
)
from weather_korea_forecast.v2.data import load_or_prepare_v2_training_table
from weather_korea_forecast.v2.dataset import V2DatasetBundle, build_v2_dataset_bundle
from weather_korea_forecast.v2.evaluate import evaluate_prediction_frame


def train_v2_experiment(config: dict) -> Path:
    resolved_model_config = resolve_model_config({"model": dict(config["model"])})
    training_table = load_or_prepare_v2_training_table(config)
    bundle = build_v2_dataset_bundle(training_table, config, backend=resolved_model_config["model"].get("backend", "fallback_torch"))

    batch_size = int(config["training"].get("batch_size", 32))
    num_workers = int(config["training"].get("num_workers", 0))
    device = str(config["training"].get("device", "cpu"))
    train_loader = bundle.make_dataloader("train", batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = bundle.make_dataloader("val", batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = bundle.make_dataloader("test", batch_size=batch_size, num_workers=num_workers, shuffle=False)

    experiment_dir = create_experiment_dir(config)
    snapshot_config(experiment_dir, config)
    write_scaler_artifact(experiment_dir, bundle.scaler)

    model = build_model(resolved_model_config, bundle)
    model_type = resolved_model_config["model"]["type"]
    if model_type in {"persistence", "seasonal_persistence"}:
        model.save(experiment_dir / "model.pt", extra_state={"bundle_metadata": bundle.metadata})
        history = []
        best_val_loss = float("nan")
        val_predictions = _predict_baseline(model, val_loader)
        test_predictions = _predict_baseline(model, test_loader)
    else:
        fit_kwargs = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "max_epochs": int(config["training"].get("max_epochs", 10)),
            "learning_rate": float(resolved_model_config["model"].get("learning_rate", config["training"].get("learning_rate", 1e-3))),
            "device": device,
            "early_stopping_patience": int(config["training"].get("early_stopping_patience", 3)),
        }
        if model_type == "tft":
            fit_kwargs["gradient_clip_val"] = float(config["training"].get("gradient_clip_val", 0.0))
        train_result = model.fit(**fit_kwargs)
        history = train_result.history
        best_val_loss = train_result.best_val_loss
        model.save(experiment_dir / "model.pt", extra_state={"bundle_metadata": bundle.metadata})
        val_predictions = model.predict_loader(val_loader, device=device)
        test_predictions = model.predict_loader(test_loader, device=device)

    write_json(history, experiment_dir / "training_history.json")
    val_frame = build_v2_prediction_frame(*val_predictions, bundle=bundle)
    test_frame = build_v2_prediction_frame(*test_predictions, bundle=bundle)

    bias_payload = compute_bias_correction(val_frame, config)
    write_json(bias_payload, experiment_dir / "bias_correction.json")

    val_frame = apply_postprocessing(val_frame, config, bias_payload)
    test_frame = apply_postprocessing(test_frame, config, bias_payload)
    write_table(test_frame, experiment_dir / "predictions_test.csv")

    raw_metrics = None
    if "prediction_raw" in test_frame.columns:
        raw_metrics = compute_prediction_metrics(test_frame.rename(columns={"prediction_raw": "_prediction_raw"}), predicted_column="_prediction_raw")
    summary = evaluate_prediction_frame(test_frame, experiment_dir)

    feature_importance = export_feature_importance(model, bundle)
    write_feature_importance(experiment_dir, feature_importance)
    write_experiment_summary(
        experiment_dir=experiment_dir,
        config=config,
        metrics=summary["metrics"],
        raw_metrics=raw_metrics,
        val_metrics=compute_prediction_metrics(val_frame) if not val_frame.empty else None,
        best_val_loss=best_val_loss,
        training_history=history,
    )
    update_leaderboard(experiment_dir, config, summary["metrics"])
    refresh_aliases(experiment_dir)
    print(experiment_dir)
    return experiment_dir


def build_v2_prediction_frame(
    prediction_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    metadata: dict[str, list],
    bundle: V2DatasetBundle,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    station_lookup = (
        bundle.full_frame[["station_id", "region_class", "region"]]
        .drop_duplicates("station_id")
        .set_index("station_id")
        .to_dict("index")
    )
    for sample_index in range(prediction_tensor.shape[0]):
        station_id = str(metadata["station_id"][sample_index])
        prediction_start = _ensure_utc_timestamp(metadata["prediction_start"][sample_index])
        station_meta = station_lookup.get(station_id, {})
        for horizon_index in range(prediction_tensor.shape[1]):
            valid_time = prediction_start + pd.Timedelta(hours=horizon_index)
            scaled_prediction = float(prediction_tensor[sample_index, horizon_index, 0].item())
            scaled_actual = float(target_tensor[sample_index, horizon_index, 0].item())
            prediction = float(bundle.scaler.inverse_values("target_value", [scaled_prediction], groups=[station_id])[0])
            actual = float(bundle.scaler.inverse_values("target_value", [scaled_actual], groups=[station_id])[0])
            rows.append(
                {
                    "station_id": station_id,
                    "prediction_start": prediction_start,
                    "valid_time": valid_time,
                    "horizon_step": horizon_index + 1,
                    "target_name": bundle.target_name,
                    "prediction": prediction,
                    "actual": actual,
                    "region_class": station_meta.get("region_class", "unknown"),
                    "region": station_meta.get("region", station_meta.get("region_class", "unknown")),
                    "season": _season_from_timestamp(valid_time),
                }
            )
    return pd.DataFrame(rows)


def compute_bias_correction(prediction_frame: pd.DataFrame, config: dict) -> dict[str, object]:
    bias_config = config.get("evaluation", {}).get("bias_correction", {})
    enabled = bool(bias_config.get("enabled", False))
    mode = str(bias_config.get("mode", "global"))
    if not enabled or prediction_frame.empty:
        return {"enabled": False, "mode": mode, "values": []}

    frame = prediction_frame.copy()
    frame["bias"] = frame["prediction"] - frame["actual"]
    if mode == "global":
        return {"enabled": True, "mode": mode, "values": [{"bias": float(frame["bias"].mean())}]}
    if mode == "per_horizon":
        grouped = frame.groupby("horizon_step")["bias"].mean().reset_index()
        return {"enabled": True, "mode": mode, "values": grouped.to_dict(orient="records")}
    if mode == "per_station_horizon":
        grouped = frame.groupby(["station_id", "horizon_step"])["bias"].mean().reset_index()
        return {"enabled": True, "mode": mode, "values": grouped.to_dict(orient="records")}
    raise ValueError(f"Unsupported bias correction mode: {mode}")


def apply_postprocessing(prediction_frame: pd.DataFrame, config: dict, bias_payload: dict[str, object]) -> pd.DataFrame:
    frame = prediction_frame.copy()
    frame["prediction_raw"] = frame["prediction"]
    if bool(bias_payload.get("enabled", False)):
        mode = str(bias_payload.get("mode", "global"))
        if mode == "global":
            correction = float(bias_payload["values"][0]["bias"])
            frame["prediction"] = frame["prediction"] - correction
        elif mode == "per_horizon":
            correction_frame = pd.DataFrame(bias_payload["values"])
            frame = frame.merge(correction_frame.rename(columns={"bias": "_bias_correction"}), on="horizon_step", how="left")
            frame["prediction"] = frame["prediction"] - frame["_bias_correction"].fillna(0.0)
            frame = frame.drop(columns=["_bias_correction"])
        elif mode == "per_station_horizon":
            correction_frame = pd.DataFrame(bias_payload["values"])
            frame = frame.merge(correction_frame.rename(columns={"bias": "_bias_correction"}), on=["station_id", "horizon_step"], how="left")
            frame["prediction"] = frame["prediction"] - frame["_bias_correction"].fillna(0.0)
            frame = frame.drop(columns=["_bias_correction"])

    clip_range = config["data"].get("postprocess", {}).get("clip_prediction")
    if clip_range:
        frame["prediction"] = frame["prediction"].clip(lower=float(clip_range[0]), upper=float(clip_range[1]))
    return frame


def export_feature_importance(model, bundle: V2DatasetBundle) -> pd.DataFrame | None:
    if not hasattr(model, "feature_importance_frame"):
        return None
    return model.feature_importance_frame(flattened_feature_names(bundle))


def flattened_feature_names(bundle: V2DatasetBundle) -> list[str]:
    names: list[str] = []
    for encoder_step in range(bundle.encoder_length):
        for column in bundle.encoder_columns:
            names.append(f"encoder_t-{bundle.encoder_length - encoder_step}:{column}")
    for horizon_step in range(1, bundle.prediction_length + 1):
        for column in bundle.decoder_columns:
            names.append(f"decoder_t+{horizon_step}:{column}")
    for column in bundle.static_baseline_columns:
        names.append(f"static:{column}")
    return names


def _predict_baseline(model, loader) -> tuple[torch.Tensor, torch.Tensor, dict[str, list]]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    metadata = {"station_id": [], "prediction_start": []}
    for batch in loader:
        predictions.append(model.predict_batch(batch))
        targets.append(batch["target"])
        metadata["station_id"].extend(batch["station_id"])
        metadata["prediction_start"].extend(batch["prediction_start"])
    return torch.cat(predictions), torch.cat(targets), metadata


def _ensure_utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _season_from_timestamp(timestamp: pd.Timestamp) -> str:
    month = timestamp.month
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a V2 experiment from a unified config.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    train_v2_experiment(config)


if __name__ == "__main__":
    main()

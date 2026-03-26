from __future__ import annotations

from copy import deepcopy

from weather_korea_forecast.models.baselines import LightGBMBaseline, PersistenceBaseline, RidgeRegressionBaseline
from weather_korea_forecast.models.tft_model import TFTModelWrapper, can_use_pytorch_forecasting


def build_model(model_config: dict, bundle):
    resolved_config = resolve_model_config(model_config)
    model_type = resolved_config["model"]["type"]
    if model_type in {"persistence", "seasonal_persistence"}:
        seasonal_period = resolved_config["model"].get("seasonal_period")
        if model_type == "seasonal_persistence" and seasonal_period is None:
            seasonal_period = int(resolved_config["model"].get("default_seasonal_period", 24))
        return PersistenceBaseline(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            seasonal_period=seasonal_period,
            target_source_features=resolved_config["model"].get("target_source_features"),
        )
    if model_type == "ridge":
        return RidgeRegressionBaseline(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            alpha=float(resolved_config["model"].get("alpha", 1.0)),
        )
    if model_type == "lightgbm":
        return LightGBMBaseline(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            params=dict(resolved_config["model"].get("params", {})),
        )
    if model_type == "tft":
        return TFTModelWrapper.from_dataset_bundle(bundle, resolved_config)
    raise ValueError(f"Unknown model type: {model_type}")


def resolve_model_config(model_config: dict) -> dict:
    resolved = deepcopy(model_config)
    model_section = resolved.setdefault("model", {})
    if model_section.get("type") != "tft":
        return resolved

    requested_backend = model_section.get("backend", "auto")
    allow_fallback = bool(model_section.get("allow_fallback_backend", requested_backend == "auto"))
    if requested_backend == "auto":
        actual_backend = "pytorch_forecasting" if can_use_pytorch_forecasting() else "fallback_torch"
    elif requested_backend == "pytorch_forecasting":
        if can_use_pytorch_forecasting():
            actual_backend = "pytorch_forecasting"
        elif allow_fallback:
            actual_backend = "fallback_torch"
        else:
            raise RuntimeError("pytorch_forecasting backend was requested but optional dependencies are not installed.")
    else:
        actual_backend = requested_backend

    model_section["requested_backend"] = requested_backend
    model_section["backend"] = actual_backend
    return resolved

from __future__ import annotations

from copy import deepcopy

from weather_korea_forecast.models.baselines import (
    CatBoostBaseline,
    DecoderFeatureBaseline,
    HorizonWiseCatBoostBaseline,
    HorizonWiseLightGBMBaseline,
    HorizonWiseRidgeRegressionBaseline,
    LightGBMBaseline,
    PersistenceBaseline,
    ResidualForecastModel,
    RidgeRegressionBaseline,
)
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
    if model_type in {"decoder_feature", "decoder_feature_baseline", "future_feature_baseline"}:
        target_source_features = resolved_config["model"].get("target_source_features")
        if not target_source_features:
            raise ValueError("decoder_feature_baseline requires model.target_source_features.")
        return DecoderFeatureBaseline(
            decoder_feature_names=bundle.decoder_columns,
            target_columns=bundle.target_columns,
            target_source_features=[str(feature) for feature in target_source_features],
        )
    if model_type == "ridge":
        return RidgeRegressionBaseline(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            alpha=float(resolved_config["model"].get("alpha", 1.0)),
            alpha_grid=[float(value) for value in resolved_config["model"].get("alpha_grid", [])] or None,
            alpha_selection=dict(resolved_config["model"].get("alpha_selection", {})),
        )
    if model_type in {"horizon_wise_ridge", "horizonwise_ridge"}:
        return HorizonWiseRidgeRegressionBaseline(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            alpha=float(resolved_config["model"].get("alpha", 1.0)),
            alpha_grid=[float(value) for value in resolved_config["model"].get("alpha_grid", [])] or None,
            alpha_selection=dict(resolved_config["model"].get("alpha_selection", {})),
        )
    if model_type == "lightgbm":
        return LightGBMBaseline(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            params=dict(resolved_config["model"].get("params", {})),
            param_grid=dict(resolved_config["model"].get("param_grid", {})),
        )
    if model_type in {"horizon_wise_lightgbm", "horizonwise_lightgbm"}:
        return HorizonWiseLightGBMBaseline(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            params=dict(resolved_config["model"].get("params", {})),
            param_grid=dict(resolved_config["model"].get("param_grid", {})),
        )
    if model_type in {"catboost", "catboost_regressor"}:
        return CatBoostBaseline(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            params=dict(resolved_config["model"].get("params", {})),
        )
    if model_type in {"horizon_wise_catboost", "horizonwise_catboost"}:
        return HorizonWiseCatBoostBaseline(
            encoder_feature_names=bundle.encoder_columns,
            target_columns=bundle.target_columns,
            prediction_length=bundle.prediction_length,
            params=dict(resolved_config["model"].get("params", {})),
        )
    if model_type == "residual":
        baseline_section = dict(resolved_config["model"].get("baseline", {}))
        residual_section = dict(resolved_config["model"].get("residual", {}))
        if not baseline_section or not residual_section:
            raise ValueError("Residual model requires model.baseline and model.residual sections.")
        baseline_config = resolve_model_config({"model": baseline_section})
        residual_config = resolve_model_config({"model": residual_section})
        if residual_config["model"].get("backend") == "pytorch_forecasting":
            raise ValueError("Residual forecasting currently supports fallback_torch/baseline residual learners, not pytorch_forecasting residual datasets.")
        return ResidualForecastModel(
            baseline_model=build_model(baseline_config, bundle),
            residual_model=build_model(residual_config, bundle),
            baseline_config=baseline_config,
            residual_config=residual_config,
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

from __future__ import annotations

from weather_korea_forecast.models.baselines import PersistenceBaseline
from weather_korea_forecast.models.tft_model import TFTModelWrapper


def build_model(model_config: dict, bundle):
    model_type = model_config["model"]["type"]
    if model_type == "persistence":
        return PersistenceBaseline(seasonal_period=model_config["model"].get("seasonal_period"))
    if model_type == "tft":
        return TFTModelWrapper.from_dataset_bundle(bundle, model_config)
    raise ValueError(f"Unknown model type: {model_type}")

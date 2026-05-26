from __future__ import annotations

import pandas as pd
import pytest
import torch

from weather_korea_forecast.models.baselines import ResidualForecastModel


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict_batch(self, batch: dict) -> torch.Tensor:
        shape = batch["target"].shape
        return torch.full(shape, self.value, dtype=torch.float32)


def test_residual_mos_reconstructs_baseline_plus_residual() -> None:
    batch = {"target": torch.zeros((2, 3, 1), dtype=torch.float32)}
    model = ResidualForecastModel(_ConstantModel(10.0), _ConstantModel(-0.75), {}, {})
    components = model.predict_components_batch(batch)

    assert torch.allclose(components["baseline"], torch.full((2, 3, 1), 10.0))
    assert torch.allclose(components["residual"], torch.full((2, 3, 1), -0.75))
    assert torch.allclose(components["final"], torch.full((2, 3, 1), 9.25))


def test_residual_prediction_columns_are_consistent() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["108"],
            "prediction_start": ["2024-01-01T00:00:00Z"],
            "valid_time": ["2024-01-01T01:00:00Z"],
            "horizon_step": [1],
            "actual": [9.8],
            "baseline_prediction": [10.0],
            "predicted_residual": [-0.25],
            "prediction_raw": [9.75],
            "prediction_corrected": [9.75],
            "error": [-0.05],
            "abs_error": [0.05],
        }
    )
    required = {"station_id", "prediction_start", "valid_time", "horizon_step", "actual", "baseline_prediction", "predicted_residual", "prediction_raw", "prediction_corrected", "error", "abs_error"}
    assert required.issubset(frame.columns)
    assert frame.loc[0, "actual"] - frame.loc[0, "baseline_prediction"] == pytest.approx(-0.2)
    assert frame.loc[0, "baseline_prediction"] + frame.loc[0, "predicted_residual"] == pytest.approx(frame.loc[0, "prediction_raw"])

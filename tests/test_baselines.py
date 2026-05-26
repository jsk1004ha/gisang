from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from weather_korea_forecast.models.baselines import (
    DecoderFeatureBaseline,
    LightGBMBaseline,
    RidgeRegressionBaseline,
    _ridge_alpha_selection_loss,
)


def test_ridge_regression_selects_alpha_from_validation_loss() -> None:
    train_batch = {
        "encoder_cont": torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[3.0]]]),
        "decoder_known": torch.zeros((4, 1, 0)),
        "static_real": torch.zeros((4, 0)),
        "target": torch.tensor([[[1.0]], [[3.0]], [[5.0]], [[7.0]]]),
    }
    val_batch = {
        "encoder_cont": torch.tensor([[[4.0]], [[5.0]]]),
        "decoder_known": torch.zeros((2, 1, 0)),
        "static_real": torch.zeros((2, 0)),
        "target": torch.tensor([[[9.0]], [[11.0]]]),
    }
    model = RidgeRegressionBaseline(
        encoder_feature_names=["x"],
        target_columns=["target_value"],
        prediction_length=1,
        alpha=1000.0,
        alpha_grid=[0.01, 1000.0],
    )

    result = model.fit([train_batch], [val_batch], max_epochs=1, learning_rate=0.0)

    assert model.alpha == 0.01
    assert result.best_val_loss < 0.01
    assert [row["alpha"] for row in result.history if row["epoch"] == "closed_form_alpha_search"] == [0.01, 1000.0]


def test_ridge_alpha_selection_scores_bias_corrected_holdout() -> None:
    prediction = torch.tensor([[5.0], [6.0], [7.0], [8.0]])
    target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    metadata = {
        "station_id": ["108", "108", "108", "108"],
        "prediction_start": [
            "2025-01-01T00:00:00Z",
            "2025-01-01T01:00:00Z",
            "2025-01-01T02:00:00Z",
            "2025-01-01T03:00:00Z",
        ],
    }

    loss = _ridge_alpha_selection_loss(
        prediction=prediction,
        target=target,
        metadata=metadata,
        prediction_length=1,
        target_count=1,
        selection_config={
            "metric": "bias_corrected_holdout_mse",
            "calibration_fraction": 0.5,
            "correction_mode": "per_station_horizon",
        },
        default_val_loss=16.0,
    )

    assert loss == 0.0


def test_decoder_feature_baseline_copies_future_known_feature() -> None:
    batch = {
        "encoder_cont": torch.zeros((2, 3, 1)),
        "decoder_known": torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
            ]
        ),
        "static_real": torch.zeros((2, 0)),
        "target": torch.tensor([[[10.0], [20.0]], [[30.0], [40.0]]]),
        "station_id": ["108", "109"],
        "prediction_start": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
    }
    model = DecoderFeatureBaseline(
        decoder_feature_names=["hour", "future_humidity"],
        target_columns=["target_value"],
        target_source_features=["future_humidity"],
    )

    prediction = model.predict_batch(batch)
    result = model.fit([batch], [batch], max_epochs=1, learning_rate=0.0)

    assert torch.equal(prediction, batch["target"])
    assert result.best_val_loss == 0.0


def test_lightgbm_load_reports_optional_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def raise_missing_lightgbm(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'lightgbm'", name="lightgbm")

    monkeypatch.setattr(torch, "load", raise_missing_lightgbm)
    bundle = SimpleNamespace(
        encoder_columns=["x"],
        target_columns=["target_value"],
        prediction_length=1,
    )

    with pytest.raises(RuntimeError, match="optional 'lightgbm' dependency"):
        LightGBMBaseline.load(tmp_path / "model.pt", bundle, {"model": {"params": {}}})

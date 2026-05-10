from __future__ import annotations

from types import SimpleNamespace

import torch

from weather_korea_forecast.models.tft_model import FallbackSeqForecaster, TFTModelWrapper, _resolve_torch_device


def test_fallback_residual_shortcut_repeats_last_encoder_values() -> None:
    model = FallbackSeqForecaster(
        encoder_length=2,
        encoder_dim=3,
        decoder_length=3,
        decoder_dim=1,
        static_dim=0,
        target_dim=2,
        hidden_size=4,
        dropout=0.0,
        residual_source_indices=[0, 2],
    )
    for parameter in model.network.parameters():
        parameter.data.zero_()

    encoder = torch.tensor([[[1.0, 10.0, 2.0], [3.0, 20.0, 4.0]]])
    decoder = torch.zeros((1, 3, 1))
    static_real = torch.zeros((1, 0))

    prediction = model(encoder, decoder, static_real)

    expected = torch.tensor([[[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]])
    assert torch.allclose(prediction, expected)


def test_tft_wrapper_infers_residual_sources_from_target_names() -> None:
    bundle = SimpleNamespace(
        encoder_length=2,
        prediction_length=2,
        encoder_columns=["obs_temp", "obs_humidity", "hour_sin"],
        decoder_columns=["hour_sin"],
        target_columns=["target_temp", "target_humidity"],
        static_columns=[],
    )

    wrapper = TFTModelWrapper.from_dataset_bundle(
        bundle,
        {
            "model": {
                "type": "tft",
                "backend": "fallback_torch",
                "hidden_size": 4,
                "dropout": 0.0,
                "residual_baseline": True,
            }
        },
    )

    assert wrapper.model.residual_source_indices.tolist() == [0, 1]


def test_tft_wrapper_accepts_v2_static_baseline_columns() -> None:
    bundle = SimpleNamespace(
        encoder_length=2,
        prediction_length=2,
        encoder_columns=["target_value", "hour_sin"],
        decoder_columns=["hour_sin"],
        target_columns=["target_value"],
        static_baseline_columns=["lat", "lon"],
    )

    wrapper = TFTModelWrapper.from_dataset_bundle(
        bundle,
        {
            "model": {
                "type": "tft",
                "backend": "fallback_torch",
                "hidden_size": 4,
                "dropout": 0.0,
                "residual_baseline": True,
            }
        },
    )

    first_linear = wrapper.model.network[0]
    assert first_linear.in_features == 2 * 2 + 2 * 1 + 2


def test_gpu_device_alias_resolves_to_available_torch_device() -> None:
    resolved = _resolve_torch_device("gpu")

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolved == expected

from __future__ import annotations

from weather_korea_forecast.data.build_training_table import build_training_table
from weather_korea_forecast.data.dataset_tft import build_dataset_bundle


def test_build_dataset_bundle_creates_windows(synthetic_project: dict) -> None:
    training_table = build_training_table(synthetic_project["data_config"])
    bundle = build_dataset_bundle(training_table, synthetic_project["data_config"], backend="fallback_torch")

    assert len(bundle.train_dataset) > 0
    sample = bundle.train_dataset[0]
    assert sample["encoder_cont"].shape == (6, 10)
    assert sample["decoder_known"].shape == (3, 4)
    assert sample["target"].shape == (3, 1)

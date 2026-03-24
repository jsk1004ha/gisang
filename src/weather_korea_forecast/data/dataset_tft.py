from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from weather_korea_forecast.features.scaling import ColumnScaler, fit_standard_scaler


@dataclass
class PreparedDatasetBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    scaler: ColumnScaler
    encoder_columns: list[str]
    decoder_columns: list[str]
    unknown_columns: list[str]
    target_columns: list[str]
    static_columns: list[str]
    encoder_length: int
    prediction_length: int
    backend: str
    metadata: dict[str, Any]

    def make_dataloader(self, split: str, batch_size: int, num_workers: int = 0, shuffle: bool = False) -> DataLoader:
        dataset = {"train": self.train_dataset, "val": self.val_dataset, "test": self.test_dataset}[split]
        if self.backend == "pytorch_forecasting":
            return dataset.to_dataloader(train=shuffle, batch_size=batch_size, num_workers=num_workers)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        encoder_columns: list[str],
        decoder_columns: list[str],
        target_columns: list[str],
        static_columns: list[str],
        encoder_length: int,
        prediction_length: int,
    ) -> None:
        self.samples: list[dict[str, torch.Tensor]] = []
        self.encoder_columns = encoder_columns
        self.decoder_columns = decoder_columns
        self.target_columns = target_columns
        self.static_columns = static_columns
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self._build_samples(frame)

    def _build_samples(self, frame: pd.DataFrame) -> None:
        for station_id, group in frame.sort_values(["station_id", "datetime"]).groupby("station_id"):
            group = group.reset_index(drop=True)
            total = len(group)
            window_size = self.encoder_length + self.prediction_length
            for start in range(total - window_size + 1):
                encoder_slice = group.iloc[start : start + self.encoder_length]
                decoder_slice = group.iloc[start + self.encoder_length : start + window_size]
                sample = {
                    "encoder_cont": torch.tensor(encoder_slice[self.encoder_columns].to_numpy(dtype="float32")),
                    "decoder_known": torch.tensor(decoder_slice[self.decoder_columns].to_numpy(dtype="float32")),
                    "static_real": torch.tensor(encoder_slice.iloc[0][self.static_columns].to_numpy(dtype="float32"))
                    if self.static_columns
                    else torch.zeros(0, dtype=torch.float32),
                    "target": torch.tensor(decoder_slice[self.target_columns].to_numpy(dtype="float32")),
                    "station_id": station_id,
                    "encoder_start": encoder_slice.iloc[0]["datetime"].isoformat(),
                    "prediction_start": decoder_slice.iloc[0]["datetime"].isoformat(),
                }
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def build_dataset_bundle(
    training_table: pd.DataFrame,
    config: dict,
    backend: str = "fallback_torch",
) -> PreparedDatasetBundle:
    training_table = training_table.copy()
    training_table["station_id"] = training_table["station_id"].astype(str)
    training_table["datetime"] = pd.to_datetime(training_table["datetime"], utc=True)
    if backend == "pytorch_forecasting":
        training_table = training_table.sort_values(["station_id", "datetime"]).reset_index(drop=True)
        training_table["time_idx"] = training_table.groupby("station_id").cumcount()
    encoder_columns = config["features"]["encoder_continuous"]
    decoder_columns = config["features"]["decoder_known"]
    unknown_columns = [column for column in encoder_columns if column not in decoder_columns]
    static_columns = config["features"].get("static_real", [])
    target_columns = [f"target_{target}" if not target.startswith("target_") else target for target in config["targets"]]
    encoder_length = int(config["window"]["encoder_length"])
    prediction_length = int(config["window"]["prediction_length"])
    scaling_columns = config.get("scaling", {}).get("columns", [])

    split_frames = {
        "train": training_table.loc[training_table["split"] == "train"].copy(),
        "val": training_table.loc[training_table["split"] == "val"].copy(),
        "test": training_table.loc[training_table["split"] == "test"].copy(),
    }
    scaler = fit_standard_scaler(split_frames["train"], scaling_columns)
    for split_name, frame in split_frames.items():
        split_frames[split_name] = scaler.transform(frame, [column for column in scaling_columns if column in frame.columns])

    if backend == "pytorch_forecasting":
        bundle = _build_pytorch_forecasting_bundle(
            split_frames=split_frames,
            encoder_columns=encoder_columns,
            decoder_columns=decoder_columns,
            unknown_columns=unknown_columns,
            static_columns=static_columns,
            target_columns=target_columns,
            encoder_length=encoder_length,
            prediction_length=prediction_length,
            scaler=scaler,
        )
        return bundle

    train_dataset = SlidingWindowDataset(
        split_frames["train"], encoder_columns, decoder_columns, target_columns, static_columns, encoder_length, prediction_length
    )
    val_dataset = SlidingWindowDataset(
        split_frames["val"], encoder_columns, decoder_columns, target_columns, static_columns, encoder_length, prediction_length
    )
    test_dataset = SlidingWindowDataset(
        split_frames["test"], encoder_columns, decoder_columns, target_columns, static_columns, encoder_length, prediction_length
    )
    return PreparedDatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_frame=split_frames["train"],
        val_frame=split_frames["val"],
        test_frame=split_frames["test"],
        scaler=scaler,
        encoder_columns=encoder_columns,
        decoder_columns=decoder_columns,
        unknown_columns=unknown_columns,
        target_columns=target_columns,
        static_columns=static_columns,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        backend=backend,
        metadata={"scaling_columns": scaling_columns},
    )


def _build_pytorch_forecasting_bundle(
    split_frames: dict[str, pd.DataFrame],
    encoder_columns: list[str],
    decoder_columns: list[str],
    unknown_columns: list[str],
    static_columns: list[str],
    target_columns: list[str],
    encoder_length: int,
    prediction_length: int,
    scaler: ColumnScaler,
) -> PreparedDatasetBundle:
    try:
        from pytorch_forecasting import TimeSeriesDataSet  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pytorch_forecasting is not installed.") from exc

    train_frame = split_frames["train"].copy()
    val_frame = split_frames["val"].copy()
    test_frame = split_frames["test"].copy()

    target_column = target_columns[0]
    common_kwargs = dict(
        time_idx="time_idx",
        target=target_column,
        group_ids=["station_id"],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_reals=static_columns,
        time_varying_known_reals=decoder_columns,
        time_varying_unknown_reals=unknown_columns,
        allow_missing_timesteps=False,
    )
    train_dataset = TimeSeriesDataSet(train_frame, **common_kwargs)
    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_frame, predict=False, stop_randomization=True)
    test_dataset = TimeSeriesDataSet.from_dataset(train_dataset, test_frame, predict=False, stop_randomization=True)
    return PreparedDatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        scaler=scaler,
        encoder_columns=encoder_columns,
        decoder_columns=decoder_columns,
        unknown_columns=unknown_columns,
        target_columns=target_columns,
        static_columns=static_columns,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        backend="pytorch_forecasting",
        metadata={},
    )

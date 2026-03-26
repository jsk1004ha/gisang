from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from weather_korea_forecast.v2.scaling import SplitAwareScaler, fit_split_aware_scaler


@dataclass
class V2DatasetBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    full_frame: pd.DataFrame
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    scaler: SplitAwareScaler
    encoder_columns: list[str]
    decoder_columns: list[str]
    unknown_columns: list[str]
    target_columns: list[str]
    target_name: str
    static_real_columns: list[str]
    static_categorical_columns: list[str]
    static_baseline_columns: list[str]
    encoder_length: int
    prediction_length: int
    backend: str
    metadata: dict[str, Any]

    def make_dataloader(self, split: str, batch_size: int, num_workers: int = 0, shuffle: bool = False) -> DataLoader:
        dataset = {"train": self.train_dataset, "val": self.val_dataset, "test": self.test_dataset}[split]
        if self.backend == "pytorch_forecasting":
            return dataset.to_dataloader(train=shuffle, batch_size=batch_size, num_workers=num_workers)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


class DirectForecastWindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        encoder_columns: list[str],
        decoder_columns: list[str],
        target_columns: list[str],
        static_baseline_columns: list[str],
        encoder_length: int,
        prediction_length: int,
        split_name: str,
    ) -> None:
        self.samples: list[dict[str, Any]] = []
        self.encoder_columns = encoder_columns
        self.decoder_columns = decoder_columns
        self.target_columns = target_columns
        self.static_baseline_columns = static_baseline_columns
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.split_name = split_name
        self._build_samples(frame)

    def _build_samples(self, frame: pd.DataFrame) -> None:
        window_size = self.encoder_length + self.prediction_length
        for station_id, group in frame.sort_values(["station_id", "datetime"]).groupby("station_id"):
            group = group.reset_index(drop=True)
            if len(group) < window_size:
                continue
            for start in range(len(group) - window_size + 1):
                window = group.iloc[start : start + window_size].copy()
                if not _is_hourly_contiguous(window["datetime"]):
                    continue
                encoder_slice = window.iloc[: self.encoder_length]
                decoder_slice = window.iloc[self.encoder_length :]
                if decoder_slice["split"].nunique() != 1 or decoder_slice["split"].iloc[0] != self.split_name:
                    continue
                sample = {
                    "encoder_cont": torch.tensor(encoder_slice[self.encoder_columns].to_numpy(dtype="float32")),
                    "decoder_known": torch.tensor(decoder_slice[self.decoder_columns].to_numpy(dtype="float32")),
                    "static_real": (
                        torch.tensor(encoder_slice.iloc[0][self.static_baseline_columns].to_numpy(dtype="float32"))
                        if self.static_baseline_columns
                        else torch.zeros(0, dtype=torch.float32)
                    ),
                    "target": torch.tensor(decoder_slice[self.target_columns].to_numpy(dtype="float32")),
                    "station_id": str(station_id),
                    "region_class": str(encoder_slice.iloc[0].get("region_class", "unknown")),
                    "prediction_start": decoder_slice.iloc[0]["datetime"].isoformat(),
                }
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def build_v2_dataset_bundle(training_table: pd.DataFrame, config: dict, backend: str = "fallback_torch") -> V2DatasetBundle:
    frame = training_table.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame.sort_values(["station_id", "datetime"]).reset_index(drop=True)
    if "region_class" in frame.columns:
        frame["region_class"] = frame["region_class"].fillna("unknown").astype(str)
    elif "region" in frame.columns:
        frame["region_class"] = frame["region"].fillna("unknown").astype(str)
    else:
        frame["region_class"] = "unknown"
    global_start = frame["datetime"].min()
    frame["time_idx"] = ((frame["datetime"] - global_start) / pd.Timedelta(hours=1)).astype(int)

    data_config = config["data"]
    features_config = data_config["features"]
    encoder_columns = list(features_config["encoder_continuous"])
    decoder_columns = list(features_config["decoder_known"])
    unknown_columns = [column for column in encoder_columns if column not in decoder_columns]
    target_name = str(data_config["target_name"])
    target_columns = ["target_value"]
    static_real_columns = [column for column in features_config.get("static_real", []) if column in frame.columns]
    static_categorical_columns = [column for column in features_config.get("static_categoricals", []) if column in frame.columns]
    encoder_length = int(data_config["window"]["encoder_length"])
    prediction_length = int(data_config["window"]["prediction_length"])
    scaling_config = data_config.get("scaling", {})
    scaling_columns = [column for column in scaling_config.get("columns", []) if column in frame.columns]
    scaling_mode = str(scaling_config.get("mode", "global"))

    train_frame_raw = frame.loc[frame["split"] == "train"].copy()
    scaler = fit_split_aware_scaler(
        train_frame=train_frame_raw,
        columns=scaling_columns,
        mode=scaling_mode,
        group_column=str(scaling_config.get("group_column", "station_id")),
    )
    frame = scaler.transform(frame, scaling_columns)

    category_levels = _fit_static_category_levels(train_frame_raw, static_categorical_columns)
    frame, static_baseline_columns = _append_static_one_hot_columns(frame, category_levels)

    required_columns = sorted(
        set(encoder_columns + decoder_columns + target_columns + static_real_columns + ["station_id", "datetime", "split", "region_class"])
    )
    drop_columns = [column for column in required_columns if column in frame.columns]
    before_drop = len(frame)
    frame = frame.dropna(subset=drop_columns).reset_index(drop=True)

    train_frame = frame.loc[frame["split"] == "train"].copy()
    val_frame = frame.loc[frame["split"] == "val"].copy()
    test_frame = frame.loc[frame["split"] == "test"].copy()

    train_dataset = DirectForecastWindowDataset(
        frame, encoder_columns, decoder_columns, target_columns, static_real_columns + static_baseline_columns, encoder_length, prediction_length, "train"
    )
    val_dataset = DirectForecastWindowDataset(
        frame, encoder_columns, decoder_columns, target_columns, static_real_columns + static_baseline_columns, encoder_length, prediction_length, "val"
    )
    test_dataset = DirectForecastWindowDataset(
        frame, encoder_columns, decoder_columns, target_columns, static_real_columns + static_baseline_columns, encoder_length, prediction_length, "test"
    )

    metadata: dict[str, Any] = {
        "scaling_columns": scaling_columns,
        "category_levels": category_levels,
        "static_baseline_columns": static_baseline_columns,
        "dropped_row_count": before_drop - len(frame),
        "global_start": str(global_start),
    }

    if backend == "pytorch_forecasting":
        train_dataset_pf, val_dataset_pf, test_dataset_pf = _build_pytorch_forecasting_datasets(
            full_frame=frame,
            train_frame=train_frame,
            val_frame=val_frame,
            test_frame=test_frame,
            target_column="target_value",
            encoder_length=encoder_length,
            prediction_length=prediction_length,
            static_real_columns=static_real_columns,
            static_categorical_columns=static_categorical_columns,
            decoder_columns=decoder_columns,
            unknown_columns=unknown_columns,
        )
        train_dataset = train_dataset_pf
        val_dataset = val_dataset_pf
        test_dataset = test_dataset_pf
        metadata["pf_ready"] = True

    return V2DatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        full_frame=frame,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        scaler=scaler,
        encoder_columns=encoder_columns,
        decoder_columns=decoder_columns,
        unknown_columns=unknown_columns,
        target_columns=target_columns,
        target_name=target_name,
        static_real_columns=static_real_columns,
        static_categorical_columns=static_categorical_columns,
        static_baseline_columns=static_real_columns + static_baseline_columns,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        backend=backend,
        metadata=metadata,
    )


def _build_pytorch_forecasting_datasets(
    full_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_column: str,
    encoder_length: int,
    prediction_length: int,
    static_real_columns: list[str],
    static_categorical_columns: list[str],
    decoder_columns: list[str],
    unknown_columns: list[str],
):
    try:
        from pytorch_forecasting import TimeSeriesDataSet  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pytorch_forecasting is not installed.") from exc

    common_kwargs = dict(
        time_idx="time_idx",
        target=target_column,
        group_ids=["station_id"],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_reals=static_real_columns,
        static_categoricals=static_categorical_columns,
        time_varying_known_reals=decoder_columns,
        time_varying_unknown_reals=unknown_columns,
        allow_missing_timesteps=True,
    )
    train_dataset = TimeSeriesDataSet(train_frame, **common_kwargs)
    val_min_idx = int(val_frame["time_idx"].min())
    val_context = full_frame.loc[full_frame["datetime"] <= val_frame["datetime"].max()].copy()
    test_min_idx = int(test_frame["time_idx"].min())
    test_context = full_frame.loc[full_frame["datetime"] <= test_frame["datetime"].max()].copy()
    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_context, predict=False, stop_randomization=True, min_prediction_idx=val_min_idx)
    test_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        test_context,
        predict=False,
        stop_randomization=True,
        min_prediction_idx=test_min_idx,
    )
    return train_dataset, val_dataset, test_dataset


def _fit_static_category_levels(train_frame: pd.DataFrame, columns: list[str]) -> dict[str, list[str]]:
    levels: dict[str, list[str]] = {}
    for column in columns:
        if column in train_frame.columns:
            values = sorted(train_frame[column].fillna("unknown").astype(str).unique().tolist())
            levels[column] = values
    return levels


def _append_static_one_hot_columns(frame: pd.DataFrame, category_levels: dict[str, list[str]]) -> tuple[pd.DataFrame, list[str]]:
    enriched = frame.copy()
    added_columns: list[str] = []
    for column, levels in category_levels.items():
        values = enriched[column].fillna("unknown").astype(str)
        for level in levels:
            encoded_column = f"static_cat__{column}__{level}"
            enriched[encoded_column] = (values == level).astype(float)
            added_columns.append(encoded_column)
    return enriched, added_columns


def _is_hourly_contiguous(datetimes: pd.Series) -> bool:
    if len(datetimes) <= 1:
        return True
    diffs = pd.to_datetime(datetimes, utc=True).sort_values().diff().dropna()
    return bool((diffs == pd.Timedelta(hours=1)).all())

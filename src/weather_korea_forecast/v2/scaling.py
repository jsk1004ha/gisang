from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SplitAwareScaler:
    mode: str
    columns: list[str]
    group_column: str = "station_id"
    global_means: dict[str, float] = field(default_factory=dict)
    global_stds: dict[str, float] = field(default_factory=dict)
    group_means: dict[str, dict[str, float]] = field(default_factory=dict)
    group_stds: dict[str, dict[str, float]] = field(default_factory=dict)

    def transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        frame = df.copy()
        target_columns = [column for column in (columns or self.columns) if column in frame.columns]
        if self.mode == "none":
            return frame

        if self.mode == "global":
            for column in target_columns:
                frame[column] = (frame[column].astype(float) - self.global_means[column]) / self.global_stds[column]
            return frame

        if self.mode == "station_wise":
            groups = frame[self.group_column].astype(str)
            for column in target_columns:
                mean_map = pd.Series(self.group_means.get(column, {}), dtype=float)
                std_map = pd.Series(self.group_stds.get(column, {}), dtype=float)
                means = groups.map(mean_map).fillna(self.global_means[column]).astype(float)
                stds = groups.map(std_map).fillna(self.global_stds[column]).astype(float)
                frame[column] = (frame[column].astype(float) - means) / stds
            return frame

        raise ValueError(f"Unsupported scaling mode: {self.mode}")

    def inverse_values(self, column: str, values, groups=None):
        array = np.asarray(values, dtype=float)
        if self.mode == "none" or column not in self.columns:
            return array
        if self.mode == "global" or groups is None:
            return array * self.global_stds[column] + self.global_means[column]

        group_values = pd.Series(groups).astype(str)
        mean_map = pd.Series(self.group_means.get(column, {}), dtype=float)
        std_map = pd.Series(self.group_stds.get(column, {}), dtype=float)
        means = group_values.map(mean_map).fillna(self.global_means[column]).to_numpy(dtype=float)
        stds = group_values.map(std_map).fillna(self.global_stds[column]).to_numpy(dtype=float)
        return array * stds + means

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "columns": self.columns,
            "group_column": self.group_column,
            "global_means": self.global_means,
            "global_stds": self.global_stds,
            "group_means": self.group_means,
            "group_stds": self.group_stds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SplitAwareScaler":
        return cls(
            mode=str(payload["mode"]),
            columns=list(payload.get("columns", [])),
            group_column=str(payload.get("group_column", "station_id")),
            global_means={str(key): float(value) for key, value in dict(payload.get("global_means", {})).items()},
            global_stds={str(key): float(value) for key, value in dict(payload.get("global_stds", {})).items()},
            group_means={
                str(column): {str(group): float(value) for group, value in dict(stats).items()}
                for column, stats in dict(payload.get("group_means", {})).items()
            },
            group_stds={
                str(column): {str(group): float(value) for group, value in dict(stats).items()}
                for column, stats in dict(payload.get("group_stds", {})).items()
            },
        )


def fit_split_aware_scaler(
    train_frame: pd.DataFrame,
    columns: list[str],
    mode: str = "global",
    group_column: str = "station_id",
) -> SplitAwareScaler:
    existing_columns = [column for column in columns if column in train_frame.columns]
    global_means: dict[str, float] = {}
    global_stds: dict[str, float] = {}
    group_means: dict[str, dict[str, float]] = {}
    group_stds: dict[str, dict[str, float]] = {}

    for column in existing_columns:
        series = train_frame[column].astype(float)
        global_means[column] = float(series.mean())
        std = float(series.std(ddof=0))
        global_stds[column] = std if std > 0 else 1.0

        if mode == "station_wise":
            grouped = train_frame.groupby(group_column)[column].agg(["mean", "std"]).reset_index()
            grouped["std"] = grouped["std"].fillna(0.0).replace(0.0, 1.0)
            group_means[column] = {
                str(row[group_column]): float(row["mean"])
                for _, row in grouped.iterrows()
            }
            group_stds[column] = {
                str(row[group_column]): float(row["std"])
                for _, row in grouped.iterrows()
            }

    return SplitAwareScaler(
        mode=mode,
        columns=existing_columns,
        group_column=group_column,
        global_means=global_means,
        global_stds=global_stds,
        group_means=group_means,
        group_stds=group_stds,
    )

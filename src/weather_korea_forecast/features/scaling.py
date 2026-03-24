from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ColumnScaler:
    means: dict[str, float]
    stds: dict[str, float]

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        frame = df.copy()
        for column in columns:
            mean = self.means[column]
            std = self.stds[column]
            frame[column] = (frame[column] - mean) / std
        return frame

    def inverse_values(self, column: str, values):
        return values * self.stds[column] + self.means[column]

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, payload: dict[str, dict[str, float]]) -> "ColumnScaler":
        return cls(means=payload["means"], stds=payload["stds"])


def fit_standard_scaler(df: pd.DataFrame, columns: list[str]) -> ColumnScaler:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column in columns:
        series = df[column].astype(float)
        means[column] = float(series.mean())
        std = float(series.std(ddof=0))
        stds[column] = std if std > 0 else 1.0
    return ColumnScaler(means=means, stds=stds)

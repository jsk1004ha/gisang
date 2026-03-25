from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast_vs_actual(predictions: pd.DataFrame, output_path: str | Path, max_points: int = 200) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "target_name" not in predictions.columns or predictions["target_name"].nunique() <= 1:
        sample = predictions.head(max_points)
        plt.figure(figsize=(10, 4))
        plt.plot(sample["valid_time"], sample["actual"], label="actual")
        plt.plot(sample["valid_time"], sample["prediction"], label="prediction")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.legend()
    else:
        target_names = list(predictions["target_name"].dropna().unique())
        figure, axes = plt.subplots(len(target_names), 1, figsize=(10, 4 * len(target_names)), squeeze=False)
        for axis, target_name in zip(axes.flatten(), target_names):
            sample = predictions.loc[predictions["target_name"] == target_name].head(max_points)
            axis.plot(sample["valid_time"], sample["actual"], label="actual")
            axis.plot(sample["valid_time"], sample["prediction"], label="prediction")
            axis.set_title(str(target_name))
            axis.tick_params(axis="x", rotation=30)
            axis.legend()
        figure.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

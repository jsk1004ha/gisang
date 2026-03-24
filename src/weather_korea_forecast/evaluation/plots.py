from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast_vs_actual(predictions: pd.DataFrame, output_path: str | Path, max_points: int = 200) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = predictions.head(max_points)
    plt.figure(figsize=(10, 4))
    plt.plot(sample["valid_time"], sample["actual"], label="actual")
    plt.plot(sample["valid_time"], sample["prediction"], label="prediction")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.legend()
    plt.savefig(path)
    plt.close()
    return path

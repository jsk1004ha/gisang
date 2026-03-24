from __future__ import annotations

import pandas as pd


def assign_time_splits(
    df: pd.DataFrame,
    split_config: dict[str, str],
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    frame = df.copy()
    dt = pd.to_datetime(frame[datetime_col], utc=True)
    train_end = pd.Timestamp(split_config["train_end"], tz="UTC")
    val_end = pd.Timestamp(split_config["val_end"], tz="UTC")
    test_end = pd.Timestamp(split_config["test_end"], tz="UTC")

    frame["split"] = "holdout"
    frame.loc[dt <= train_end, "split"] = "train"
    frame.loc[(dt > train_end) & (dt <= val_end), "split"] = "val"
    frame.loc[(dt > val_end) & (dt <= test_end), "split"] = "test"
    return frame

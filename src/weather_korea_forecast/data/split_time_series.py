from __future__ import annotations

import pandas as pd


def assign_time_splits(
    df: pd.DataFrame,
    split_config: dict[str, str],
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    frame = df.copy()
    dt = pd.to_datetime(frame[datetime_col], utc=True)
    train_end = _ensure_utc_timestamp(split_config["train_end"])
    val_end = _ensure_utc_timestamp(split_config["val_end"])
    test_end = _ensure_utc_timestamp(split_config["test_end"])

    frame["split"] = "holdout"
    frame.loc[dt <= train_end, "split"] = "train"
    frame.loc[(dt > train_end) & (dt <= val_end), "split"] = "val"
    frame.loc[(dt > val_end) & (dt <= test_end), "split"] = "test"
    return frame


def _ensure_utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")

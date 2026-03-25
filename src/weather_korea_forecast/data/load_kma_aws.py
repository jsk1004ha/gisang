from __future__ import annotations

from pathlib import Path

import pandas as pd

from weather_korea_forecast.data.load_observations import load_observation_table


def load_kma_aws(
    path: str | Path,
    column_mapping: dict[str, str] | None = None,
    source_tz: str = "Asia/Seoul",
) -> pd.DataFrame:
    return load_observation_table(path=path, column_mapping=column_mapping, source_tz=source_tz, source_name="aws")

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.utils.env import load_dotenv
from weather_korea_forecast.utils.io import write_table
from weather_korea_forecast.utils.logger import get_logger
from weather_korea_forecast.utils.paths import resolve_path

LOGGER = get_logger(__name__)

ASOS_HOURLY_URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
AWS_MINUTELY_URL = "https://apis.data.go.kr/1360000/Aws1miInfoService/getAws1miList"


@dataclass
class ServiceSpec:
    url: str
    datetime_format: str
    make_params: Any


SERVICE_SPECS = {
    "asos_hourly": ServiceSpec(url=ASOS_HOURLY_URL, datetime_format="%Y-%m-%dT%H:%M:%S%z", make_params=lambda s, e, station, cfg: {
        "ServiceKey": _require_kma_key(),
        "numOfRows": cfg.get("page_size", 999),
        "dataType": cfg.get("data_type", "JSON"),
        "dataCd": "ASOS",
        "dateCd": "HR",
        "startDt": s.strftime("%Y%m%d"),
        "startHh": s.strftime("%H"),
        "endDt": e.strftime("%Y%m%d"),
        "endHh": e.strftime("%H"),
        "stnIds": str(station),
    }),
    "aws_minutely": ServiceSpec(url=AWS_MINUTELY_URL, datetime_format="%Y-%m-%dT%H:%M:%S%z", make_params=lambda s, e, station, cfg: {
        "ServiceKey": _require_kma_key(),
        "numOfRows": cfg.get("page_size", 999),
        "dataType": cfg.get("data_type", "JSON"),
        "awsDt": s.strftime("%Y%m%d%H%M"),
        "awsId": str(station),
    }),
}


def download_kma_observations(config: dict[str, Any]) -> pd.DataFrame:
    load_dotenv()
    service_name = config["service"]
    if service_name not in SERVICE_SPECS:
        raise ValueError(f"Unsupported KMA service: {service_name}")

    spec = SERVICE_SPECS[service_name]
    stations = [str(station) for station in config["stations"]]
    start = pd.Timestamp(config["start"])
    end = pd.Timestamp(config["end"])
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("KMA download config start/end must include timezone offsets.")

    frames: list[pd.DataFrame] = []
    for station in stations:
        if service_name == "aws_minutely":
            cursor = start
            while cursor <= end:
                next_day = min(cursor + timedelta(days=1) - timedelta(minutes=1), end)
                frame = _request_paginated_json(spec.url, spec.make_params(cursor, next_day, station, config), page_size=int(config.get("page_size", 999)))
                if not frame.empty:
                    frame["station_id"] = station
                    frames.append(frame)
                cursor += timedelta(days=1)
        else:
            frame = _request_paginated_json(spec.url, spec.make_params(start, end, station, config), page_size=int(config.get("page_size", 999)))
            if not frame.empty:
                frame["station_id"] = station
                frames.append(frame)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = _normalize_kma_frame(result, service_name)
    output_path = config["output_path"]
    write_table(result, output_path)
    LOGGER.info("Wrote %s rows to %s", len(result), output_path)
    return result


def _request_paginated_json(url: str, params: dict[str, Any], page_size: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    page = 1
    while True:
        paged_params = dict(params)
        paged_params["pageNo"] = page
        paged_params["numOfRows"] = page_size
        frame, total_count = _request_json(url, paged_params)
        if frame.empty:
            break
        frames.append(frame)
        if total_count is None or page * page_size >= total_count:
            break
        page += 1
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _request_json(url: str, params: dict[str, Any]) -> tuple[pd.DataFrame, int | None]:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    body = payload.get("response", {}).get("body", {})
    items = body.get("items", {}).get("item", [])
    total_count = body.get("totalCount")
    if isinstance(items, dict):
        items = [items]
    parsed_total_count = int(total_count) if total_count is not None else None
    return pd.DataFrame(items), parsed_total_count


def _normalize_kma_frame(df: pd.DataFrame, service_name: str) -> pd.DataFrame:
    frame = df.copy()
    if service_name == "asos_hourly":
        rename_map = {
            "tm": "datetime",
            "ta": "temp",
            "hm": "humidity",
            "pa": "pressure",
            "ws": "wind_speed",
            "rn": "precipitation",
        }
    else:
        rename_map = {
            "awsDt": "datetime",
            "ta": "temp",
            "hm": "humidity",
            "ps": "pressure",
            "ws10M": "wind_speed",
            "rn60M": "precipitation",
            "awsId": "station_id",
        }
    frame = frame.rename(columns=rename_map)
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for column in ("temp", "humidity", "pressure", "wind_speed", "precipitation"):
        if column not in frame.columns:
            frame[column] = pd.NA
    if "quality_flag" not in frame.columns:
        frame["quality_flag"] = ""
    keep_columns = ["station_id", "datetime", "temp", "humidity", "pressure", "wind_speed", "precipitation", "quality_flag"]
    extras = [column for column in frame.columns if column not in keep_columns]
    return frame[keep_columns + extras]


def _require_kma_key() -> str:
    key = os.getenv("KMA_API_KEY")
    if not key:
        raise RuntimeError("KMA_API_KEY is not set. A data.go.kr service key is required for KMA downloads.")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description="Download KMA ASOS/AWS observation data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    download_kma_observations(config)


if __name__ == "__main__":
    main()

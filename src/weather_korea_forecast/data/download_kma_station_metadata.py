from __future__ import annotations

import argparse
import os
from io import StringIO
from typing import Any

import pandas as pd
import requests

from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.utils.env import load_dotenv
from weather_korea_forecast.utils.io import write_table
from weather_korea_forecast.utils.logger import get_logger
from weather_korea_forecast.utils.paths import resolve_path

LOGGER = get_logger(__name__)

KMA_STATION_INFO_URL = "https://apihub.kma.go.kr/api/typ01/url/stn_inf.php"
KMA_DATAWIKI_STATION_URL = "https://datawiki.kma.go.kr/doku.php?id=%EA%B8%B0%EC%83%81%EA%B4%80%EC%B8%A1%3A%EC%A7%80%EC%83%81%3A%EC%9E%90%EB%8F%99%EA%B8%B0%EC%83%81%EA%B4%80%EC%B8%A1_aws"


def download_kma_station_metadata(config: dict[str, Any]) -> pd.DataFrame:
    load_dotenv()
    source = str(config.get("source", "datawiki"))
    if source == "datawiki":
        metadata = _download_station_metadata_from_datawiki(config)
    else:
        response_text = _request_station_metadata_text(config)
        metadata = parse_kma_station_metadata_text(response_text, config)
    output_path = resolve_path(config["output_path"])
    write_table(metadata, output_path)
    LOGGER.info("Wrote %s station rows to %s", len(metadata), output_path)
    return metadata


def parse_kma_station_metadata_text(text: str, config: dict[str, Any]) -> pd.DataFrame:
    station_filter = {str(station) for station in config.get("stations", [])}
    region_class_map = {str(key): value for key, value in config.get("region_class_map", {}).items()}
    coastal_distance_map = {str(key): value for key, value in config.get("coastal_distance_km_map", {}).items()}

    rows: list[dict[str, object]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not line[0].isdigit():
            continue
        tokens = line.split()
        if len(tokens) < 6:
            continue
        station_id = tokens[0]
        if station_filter and station_id not in station_filter:
            continue
        rows.append(
            {
                "station_id": station_id,
                "station_name_ko": tokens[1],
                "station_name_en": tokens[2],
                "lon": float(tokens[3]),
                "lat": float(tokens[4]),
                "elevation": float(tokens[5]),
                "region_class": region_class_map.get(station_id, "unknown"),
                "coastal_distance_km": coastal_distance_map.get(station_id),
            }
        )

    if not rows:
        raise ValueError("No station metadata rows were parsed from the KMA response.")

    metadata = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
    metadata["station_id"] = metadata["station_id"].astype(str)
    return metadata


def _request_station_metadata_text(config: dict[str, Any]) -> str:
    params = {
        "inf": str(config.get("inf", "SFC")),
        "help": int(config.get("help", 1)),
        "authKey": _require_kma_key(),
    }
    response = requests.get(KMA_STATION_INFO_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.content.decode(config.get("encoding", "euc-kr"), errors="ignore")


def _download_station_metadata_from_datawiki(config: dict[str, Any]) -> pd.DataFrame:
    response = requests.get(str(config.get("datawiki_url", KMA_DATAWIKI_STATION_URL)), timeout=60)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    station_frame = _select_station_table(tables)
    station_frame["station_id"] = station_frame["station_id"].astype(str)

    station_filter = {str(station) for station in config.get("stations", [])}
    if station_filter:
        station_frame = station_frame[station_frame["station_id"].isin(station_filter)].copy()
    if station_frame.empty:
        raise ValueError("No requested stations were found in the KMA DataWiki station table.")

    region_class_map = {str(key): value for key, value in config.get("region_class_map", {}).items()}
    coastal_distance_map = {str(key): value for key, value in config.get("coastal_distance_km_map", {}).items()}
    station_frame["region_class"] = station_frame["station_id"].map(region_class_map).fillna("unknown")
    station_frame["coastal_distance_km"] = station_frame["station_id"].map(coastal_distance_map)
    return station_frame.sort_values("station_id").reset_index(drop=True)


def _select_station_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        columns = [str(column) for column in table.columns]
        if {"지점번호", "지점명(한글)", "지점명(영문)", "경도(degree)", "위도(degree)", "관측장비 해발고도(m)"}.issubset(columns):
            selected = table.rename(
                columns={
                    "지점번호": "station_id",
                    "지점명(한글)": "station_name_ko",
                    "지점명(영문)": "station_name_en",
                    "경도(degree)": "lon",
                    "위도(degree)": "lat",
                    "관측장비 해발고도(m)": "elevation",
                }
            )
            keep_columns = ["station_id", "station_name_ko", "station_name_en", "lon", "lat", "elevation"]
            return selected[keep_columns].copy()
    raise ValueError("Could not locate the station metadata table on the KMA DataWiki page.")


def _require_kma_key() -> str:
    key = os.getenv("KMA_API_KEY")
    if not key:
        raise RuntimeError("KMA_API_KEY is not set. A KMA API Hub auth key is required.")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize KMA station metadata.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    download_kma_station_metadata(config)


if __name__ == "__main__":
    main()

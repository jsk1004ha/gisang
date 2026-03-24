from __future__ import annotations

import argparse
import os
from pathlib import Path

from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.utils.logger import get_logger
from weather_korea_forecast.utils.paths import resolve_path

LOGGER = get_logger(__name__)


def download_or_register_era5(config: dict) -> Path:
    output_path = resolve_path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "local_source" in config:
        source = resolve_path(config["local_source"])
        if not source.exists():
            raise FileNotFoundError(source)
        output_path.write_bytes(source.read_bytes())
        LOGGER.info("Copied local ERA5 source to %s", output_path)
        return output_path

    try:
        import cdsapi  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "cdsapi is required for remote ERA5 downloads. Provide a local_source or install cdsapi."
        ) from exc

    _ensure_cdsapi_credentials()
    client = cdsapi.Client()
    client.retrieve(config["dataset"], config["request"], str(output_path))
    LOGGER.info("Downloaded ERA5 file to %s", output_path)
    return output_path


def _ensure_cdsapi_credentials() -> None:
    home_credential_path = Path.home() / ".cdsapirc"
    if home_credential_path.exists():
        return

    url = os.getenv("CDSAPI_URL")
    key = os.getenv("CDSAPI_KEY")
    if not url or not key:
        raise RuntimeError(
            "CDS credentials are missing. Set CDSAPI_URL and CDSAPI_KEY or provide an existing ~/.cdsapirc file."
        )

    home_credential_path.write_text(f"url: {url}\nkey: {key}\n", encoding="utf-8")
    LOGGER.info("Wrote CDS credentials file to %s", home_credential_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or register ERA5 raw files.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    download_or_register_era5(config)


if __name__ == "__main__":
    main()

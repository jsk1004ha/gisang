from __future__ import annotations

import argparse

from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.v2.data import prepare_v2_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a V2 multi-station training table.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_path, quality_path = prepare_v2_data(config)
    print({"training_table": str(output_path), "data_quality": str(quality_path) if quality_path else None})


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from weather_korea_forecast.reporting.report import run_pipeline_and_write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training, evaluation, inference, and report generation for V1.")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    result = run_pipeline_and_write_report(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        result_dir=args.result_dir,
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()

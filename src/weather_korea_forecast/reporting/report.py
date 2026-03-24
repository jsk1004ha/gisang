from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from weather_korea_forecast.evaluation.evaluate import evaluate_experiment
from weather_korea_forecast.inference.predict import generate_forecast
from weather_korea_forecast.training.train import train_experiment
from weather_korea_forecast.utils.config import load_yaml
from weather_korea_forecast.utils.io import read_table, write_json, write_table
from weather_korea_forecast.utils.paths import ensure_dir, resolve_path


@dataclass
class ReportBundle:
    experiment_dir: str
    result_dir: str
    metrics: dict[str, float]
    inference_init_time: str
    inference_rows: int
    split_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_pipeline_and_write_report(
    data_config_path: str | Path,
    model_config_path: str | Path,
    train_config_path: str | Path,
    result_dir: str | Path,
) -> ReportBundle:
    data_config = load_yaml(data_config_path)
    model_config = load_yaml(model_config_path)
    train_config = load_yaml(train_config_path)

    experiment_dir = train_experiment(data_config=data_config, model_config=model_config, train_config=train_config)
    evaluation = evaluate_experiment(experiment_dir)

    experiment_path = resolve_path(experiment_dir)
    result_root = ensure_dir(result_dir)
    result_path = ensure_dir(result_root / experiment_path.name)

    training_table = read_table(data_config["paths"]["output_training_table"])
    predictions = read_table(experiment_path / "predictions_test.csv")
    split_counts = training_table["split"].value_counts(dropna=False).sort_index().to_dict()

    prediction_starts = pd.to_datetime(predictions["prediction_start"], utc=True).sort_values().drop_duplicates()
    if prediction_starts.empty:
        raise ValueError("No prediction_start values found in predictions_test.csv.")
    inference_init_time = (prediction_starts.iloc[-1] - pd.Timedelta(hours=1)).isoformat()
    first_station = str(predictions["station_id"].astype(str).iloc[0])
    inference_frame = generate_forecast(
        experiment_dir=experiment_path,
        station_id=first_station,
        forecast_init_time=inference_init_time,
    )

    _copy_experiment_artifacts(experiment_path, result_path)
    write_table(predictions, result_path / "predictions_test.csv")
    write_table(training_table, result_path / "training_table_snapshot.csv")
    write_table(inference_frame, result_path / "inference_sample.csv")

    comparison_frame = _build_comparison_summary(predictions)
    write_table(comparison_frame, result_path / "comparison_summary.csv")
    write_table(predictions.head(36), result_path / "prediction_vs_actual_head.csv")
    write_table(predictions.tail(36), result_path / "prediction_vs_actual_tail.csv")

    report_bundle = ReportBundle(
        experiment_dir=str(experiment_path),
        result_dir=str(result_path),
        metrics=evaluation["metrics"],
        inference_init_time=inference_init_time,
        inference_rows=int(len(inference_frame)),
        split_counts={str(key): int(value) for key, value in split_counts.items()},
    )
    write_json(report_bundle.to_dict(), result_path / "report_summary.json")
    _write_markdown_report(
        result_path=result_path,
        report_bundle=report_bundle,
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        training_table=training_table,
        predictions=predictions,
        comparison_frame=comparison_frame,
        inference_frame=inference_frame,
    )
    return report_bundle


def _copy_experiment_artifacts(experiment_path: Path, result_path: Path) -> None:
    for name in (
        "data_config.yaml",
        "model_config.yaml",
        "train_config.yaml",
        "metrics_test.json",
        "metrics_summary.json",
        "metrics_horizon_step.csv",
        "metrics_region.csv",
        "metrics_season.csv",
        "metrics_station_id.csv",
        "forecast_vs_actual.png",
        "experiment_summary.json",
        "training_history.json",
        "weatherbenchx_sparse.csv",
    ):
        source = experiment_path / name
        if source.exists():
            shutil.copy2(source, result_path / name)


def _build_comparison_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["error"] = frame["prediction"] - frame["actual"]
    frame["abs_error"] = frame["error"].abs()
    top_worst = frame.sort_values("abs_error", ascending=False).head(20).reset_index(drop=True)
    return top_worst


def _write_markdown_report(
    result_path: Path,
    report_bundle: ReportBundle,
    data_config_path: str | Path,
    model_config_path: str | Path,
    train_config_path: str | Path,
    training_table: pd.DataFrame,
    predictions: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    inference_frame: pd.DataFrame,
) -> None:
    split_summary = (
        training_table.groupby("split")
        .agg(rows=("split", "size"), start=("datetime", "min"), end=("datetime", "max"))
        .reset_index()
    )
    horizon_summary = predictions.groupby("horizon_step").agg(
        actual_mean=("actual", "mean"),
        prediction_mean=("prediction", "mean"),
        mae=("actual", lambda s: 0.0),
    )
    horizon_summary = horizon_summary.reset_index()
    frame = predictions.copy()
    frame["abs_error"] = (frame["prediction"] - frame["actual"]).abs()
    horizon_mae = frame.groupby("horizon_step")["abs_error"].mean().rename("mae").reset_index()
    horizon_summary = horizon_summary.drop(columns=["mae"]).merge(horizon_mae, on="horizon_step", how="left")

    report_text = "\n".join(
        [
            "# V1 실험 보고서",
            "",
            "## 개요",
            f"- 실험 디렉터리: `{report_bundle.experiment_dir}`",
            f"- 결과 디렉터리: `{report_bundle.result_dir}`",
            f"- 데이터 설정: `{resolve_path(data_config_path)}`",
            f"- 모델 설정: `{resolve_path(model_config_path)}`",
            f"- 학습 설정: `{resolve_path(train_config_path)}`",
            "",
            "## 데이터 요약",
            f"- 전체 행 수: `{len(training_table)}`",
            f"- 관측소 수: `{training_table['station_id'].astype(str).nunique()}`",
            f"- split 분포: `{json.dumps(report_bundle.split_counts, ensure_ascii=False)}`",
            "",
            "```text",
            split_summary.to_string(index=False),
            "```",
            "",
            "## 테스트 성능",
            f"- RMSE: `{report_bundle.metrics['rmse']:.4f}`",
            f"- MAE: `{report_bundle.metrics['mae']:.4f}`",
            f"- Bias: `{report_bundle.metrics['bias']:.4f}`",
            f"- MAPE: `{report_bundle.metrics['mape']:.4f}`",
            "",
            "### Horizon별 요약",
            "```text",
            horizon_summary.to_string(index=False),
            "```",
            "",
            "## 실제값 vs 예측값 오차가 큰 샘플",
            "```text",
            comparison_frame[["station_id", "prediction_start", "valid_time", "horizon_step", "actual", "prediction", "error", "abs_error"]]
            .head(10)
            .to_string(index=False),
            "```",
            "",
            "## 샘플 추론 결과",
            f"- 추론 기준 시각: `{report_bundle.inference_init_time}`",
            "```text",
            inference_frame.to_string(index=False),
            "```",
            "",
            "## 산출물",
            "- `forecast_vs_actual.png`: 테스트 구간 실제값/예측값 그래프",
            "- `predictions_test.csv`: 테스트 예측 전체 결과",
            "- `comparison_summary.csv`: 절대오차 상위 샘플",
            "- `inference_sample.csv`: 샘플 추론 결과",
            "- `metrics_*.csv`: breakdown metric",
        ]
    )
    (result_path / "report.md").write_text(report_text, encoding="utf-8")

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from weather_korea_forecast.reporting.generate_report import build_report, main


def _make_exp(root: Path, name: str, rmse: float) -> None:
    exp = root / "v3_experiments" / name
    exp.mkdir(parents=True)
    (exp / "experiment_summary.json").write_text(
        json.dumps({"experiment_name": name, "version": "v3", "target_name": "temp", "track": "nwp_assisted_mos", "model_type": "ridge", "metrics": {"rmse": rmse, "mae": 0.7, "bias": 0.1}}),
        encoding="utf-8",
    )
    (exp / "predictions_test.csv").write_text("station_id,prediction,actual\n108,1,1.1\n", encoding="utf-8")


def _make_oracle_exp(root: Path) -> None:
    exp = root / "v3_experiments" / "oracle"
    exp.mkdir(parents=True)
    (exp / "experiment_summary.json").write_text(
        json.dumps(
            {
                "experiment_name": "oracle",
                "version": "v3",
                "target_name": "temp",
                "track": "oracle",
                "model_type": "decoder_feature_baseline",
                "future_feature_source": "observed_target_oracle",
                "backtest_only": True,
                "operational_valid": False,
                "metrics": {"rmse": 0.0, "mae": 0.0, "bias": 0.0},
            }
        ),
        encoding="utf-8",
    )
    (exp / "predictions_test.csv").write_text("station_id,prediction,actual\n108,1,1\n", encoding="utf-8")


def test_build_report_writes_all_outputs(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _make_exp(root, "a", 1.05)
    _make_exp(root, "b", 0.95)
    _make_oracle_exp(root)
    out = tmp_path / "reports"

    result = build_report(experiments_root=root, output_dir=out, title="Synthetic", include_images=False)

    assert result["total_experiments"] == 3
    assert result["diagnostic_experiments"] == 1
    assert result["main_leaderboard_experiments"] == 2
    assert (out / "experiment_report.html").exists()
    assert (out / "experiment_summary.csv").exists()
    assert (out / "experiment_summary.json").exists()
    assert (out / "best_models.csv").exists()
    assert (out / "failed_or_incomplete_experiments.csv").exists()
    best = pd.read_csv(out / "best_models.csv")
    assert best.iloc[0]["experiment_name"] == "b"
    assert "oracle" not in set(best["experiment_name"])


def test_generate_report_main_prints_summary(tmp_path: Path, capsys) -> None:
    root = tmp_path / "artifacts"
    _make_exp(root, "a", 1.05)
    out = tmp_path / "reports"

    main(["--experiments-root", str(root), "--output-dir", str(out), "--title", "Synthetic", "--no-images"])

    captured = capsys.readouterr().out
    assert "total experiments: 1" in captured
    assert str(out / "experiment_report.html") in captured

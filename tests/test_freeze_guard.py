from weather_korea_forecast.service.freeze_guard import FreezeGuardError, assert_post_freeze_paths_allowed


def _freeze_record() -> dict[str, object]:
    return {
        "allowed_post_freeze_change_scopes": ["web", "api", "docs", "schema_consumption"],
        "forbidden_post_freeze_change_scopes": ["training", "model_selection", "experiment_candidates"],
    }


def test_freeze_guard_allows_g031_site_api_schema_paths() -> None:
    result = assert_post_freeze_paths_allowed(
        [
            "web/app.js",
            "src/weather_korea_forecast/api/main.py",
            "src/weather_korea_forecast/service/export_forecast.py",
            "docs/SITE_FINALIZATION.md",
            "tests/test_api_forecast.py",
            "README.md",
        ],
        _freeze_record(),
    )

    assert "web/app.js" in result["allowed"]
    assert result["blocked"] == []


def test_freeze_guard_blocks_training_and_model_selection_paths() -> None:
    try:
        assert_post_freeze_paths_allowed(
            [
                "src/weather_korea_forecast/training/train.py",
                "src/weather_korea_forecast/v4/operational_performance.py",
                "configs/v3/experiments/new_model.yaml",
            ],
            _freeze_record(),
        )
    except FreezeGuardError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("FreezeGuardError was not raised")

    assert "training/train.py" in message
    assert "operational_performance.py" in message
    assert "configs/v3/experiments/new_model.yaml" in message

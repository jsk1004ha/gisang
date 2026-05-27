from weather_korea_forecast.service.confidence import estimate_confidence


def test_confidence_research_when_not_operational():
    result = estimate_confidence(target_name="temp", rmse=1.0, horizon_step=3, operational_valid=False, backtest_only=True)
    assert result.confidence == "research"
    assert result.confidence_score < 0.5


def test_confidence_operational_temp_and_humidity_beta():
    assert estimate_confidence(target_name="temp", rmse=1.1, horizon_step=6, operational_valid=True, backtest_only=False).confidence == "high"
    assert estimate_confidence(target_name="humidity", rmse=14, horizon_step=6, operational_valid=True, backtest_only=False, humidity_beta=True).confidence == "low"

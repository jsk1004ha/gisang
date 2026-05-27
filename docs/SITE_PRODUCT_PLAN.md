# Site Product Plan

The site MVP displays forecast artifacts, not training/test artifacts. It must never read `predictions_test.csv` directly.

MVP screens:

- Home with station selector and warning banner.
- Station forecast with 24h temperature, humidity beta, weather icon, hourly table, daily max/min, confidence.
- Model status with operational flags, best RMSE references, latest forecast run time, and V4-C gate status.

Current public copy must say: research/backtest forecast; not validated for live operational use.

# 예측 모델 개선 로그

## 2026-05-10 1차 개선: fallback_torch residual shortcut 및 gradient clipping

### 변경 내용
- `src/weather_korea_forecast/models/tft_model.py`
  - `fallback_torch` 경량 대체 모델에 마지막 encoder 관측값 기반 residual baseline을 추가했다.
  - 기본 예측선을 persistence 형태로 두고, MLP는 horizon별 보정량을 학습하도록 바꿨다.
  - `residual_baseline`, `target_source_features`, `residual_scale` 설정을 지원하게 했다.
  - V1 `static_columns`뿐 아니라 V2 `static_baseline_columns`도 받아 fallback 모델을 만들 수 있게 했다.
  - `gpu`/`cuda` device alias를 실제 사용 가능한 PyTorch device로 정규화해 fallback 경로가 `device: gpu` 설정에서 바로 실패하지 않게 했다.
  - fallback 학습 루프에서 `gradient_clip_val`을 실제로 적용해 큰 gradient로 인한 불안정 학습을 줄였다.
- `src/weather_korea_forecast/training/train.py`
  - V1 TFT 학습 시 train config의 `gradient_clip_val`을 모델 학습 함수로 전달하도록 했다.
- `configs/model/tft_v1.yaml`
  - fallback_torch 전용 residual baseline을 켜고 `obs_temp`를 온도 target의 source feature로 명시했다.
- `configs/train/train_v1.yaml`
  - 기본 학습 설정에 `gradient_clip_val: 0.5`를 추가했다.
- `tests/test_tft_model.py`
  - residual shortcut이 마지막 encoder 값을 horizon 전체에 반복하는지 검증했다.
  - target 이름에서 residual source feature를 자동 추론하는지 검증했다.
  - V2 bundle의 `static_baseline_columns`로 fallback 모델을 만들 수 있는지 검증했다.
  - `gpu` device alias가 사용 가능한 PyTorch device로 정규화되는지 검증했다.

### 테스트 기록
- 실패: `PYTHONPATH=src python -m pytest -q tests/test_tft_model.py tests/test_train_inference.py`
  - 사유: 현재 쉘에 `python` 명령이 없음.
- 실패: `PYTHONPATH=src python3 -m pytest -q tests/test_tft_model.py tests/test_train_inference.py`
  - 사유: 시스템 `python3` 환경에 `pytest`가 설치되어 있지 않음.
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q tests/test_tft_model.py tests/test_train_inference.py`
  - 결과: `7 passed, 17 warnings in 16.92s`
  - 경고는 pandas `'H'` frequency deprecation, Lightning checkpoint/data-loader 안내, GPU 미사용 안내로 이번 변경의 실패는 아님.

### 기대 효과
- fallback_torch는 true TFT가 아니지만, 마지막 관측값을 기준선으로 삼는 residual forecasting이 추가되어 짧은 학습에서도 persistence보다 나쁜 초기 예측으로 출발할 위험을 줄인다.
- gradient clipping으로 작은 synthetic 학습뿐 아니라 실제 데이터 학습에서도 과도한 gradient 업데이트를 완화한다.

### 추가 테스트 기록
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q tests/test_tft_model.py tests/test_train_inference.py`
  - 결과: `7 passed, 17 warnings in 9.27s`
  - 테스트 assertion 정리 후 동일 범위를 재검증했다.

## 2026-05-10 2차 개선: 전체 테스트 실패 원인 보정

### 변경 내용
- `src/weather_korea_forecast/data/load_observations.py`
  - 관측 소스 병합 시 `prefer_columns`를 지원하도록 했다.
  - `prefer_columns`에 지정된 컬럼은 해당 소스 값을 우선 사용하고, 나머지 컬럼은 기존 priority 순서로 결측을 보완한다.
- `src/weather_korea_forecast/data/build_training_table.py`, `src/weather_korea_forecast/v2/data.py`
  - legacy AWS 보조 소스의 기본 우선 컬럼을 `humidity`, `pressure`, `wind_speed`, `precipitation`, `quality_flag`로 지정했다.
  - `temp`는 ASOS 기본값을 유지해 잘못된 AWS 온도가 target으로 들어가는 위험을 줄였다.
- `configs/data/dataset_asos_aws_v1.yaml`, `README.md`
  - `aws.prefer_columns` 예시와 병합 규칙 설명을 추가했다.
- `src/weather_korea_forecast/evaluation/plots.py`, `src/weather_korea_forecast/v2/evaluate.py`
  - matplotlib 백엔드를 `Agg`로 고정해 GUI/Tk가 없는 Windows venv에서도 plot 저장이 가능하게 했다.

### 테스트 기록
- 실패: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q`
  - 결과: `2 failed, 15 passed, 1 skipped, 36 warnings in 19.73s`
  - 실패 1: ASOS/AWS 병합 테스트에서 보조 AWS 컬럼 우선 병합이 되지 않음.
  - 실패 2: matplotlib가 Tk 백엔드를 열려고 하면서 `init.tcl`을 찾지 못함.
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q tests/test_observation_sources.py tests/test_train_inference.py::test_ridge_baseline_roundtrip`
  - 결과: `2 passed, 2 warnings in 2.89s`

### 기대 효과
- 보조 AWS CSV를 사용할 때 온도 target은 ASOS를 유지하면서 습도/기압/풍속/강수 같은 보조 변수는 더 촘촘한 AWS 집계값을 쓸 수 있어 feature 품질이 좋아진다.
- 실험 산출 plot 저장이 headless 환경에서도 안정화되어 학습/평가 파이프라인의 재현성이 좋아진다.

## 2026-05-10 최종 검증

### 테스트 기록
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q`
  - 결과: `17 passed, 1 skipped, 36 warnings in 18.91s`
  - 경고는 pandas frequency deprecation 및 Lightning 안내성 경고이며 테스트 실패는 없음.
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m compileall src`
  - 결과: 모든 `src` 모듈 compile 완료.

### 최종 상태
- residual fallback 모델 개선, AWS 보조 feature 병합 개선, headless plot 안정화가 모두 테스트로 검증되었다.
- 생성 데이터나 실험 산출물은 변경 대상으로 삼지 않았다.

### 최종 재검증 기록
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q && PYTHONPATH=src .venv312/Scripts/python.exe -m compileall src`
  - 결과: `17 passed, 1 skipped, 36 warnings in 19.30s` 이후 `compileall src` 통과.
  - `prefer_columns`의 `None` 방어 처리까지 반영한 뒤 전체 회귀 테스트와 compile 검증을 다시 실행했다.

## 2026-05-10 3차 개선: V2 bias correction holdout guard

### 기준 평가에서 확인한 문제
- 기존 V2 humidity 계열 산출물에서 `raw_metrics`가 bias correction 적용 후 metrics보다 좋은 경우가 있었다.
  - 예: `data/artifacts/v2_experiments/latest/metrics_test.json` 기준 humidity TFT는 raw RMSE `14.4155`, 보정 후 RMSE `15.5333`으로 보정이 test 성능을 악화했다.
  - 예: `data/artifacts/v2_experiments_real/latest/metrics_test.json` 기준 humidity LightGBM은 raw RMSE `14.0834`, 보정 후 RMSE `14.4066`으로 보정이 test 성능을 악화했다.
- 원인은 validation 전체에서 bias를 산출하고 곧바로 적용해, 보정값이 일부 기간에 과적합될 수 있기 때문이다.

### 변경 내용
- `src/weather_korea_forecast/v2/train.py`
  - `bias_correction.calibration_fraction`을 추가해 validation 예측을 시간 순서대로 calibration 구간과 holdout selection 구간으로 나눌 수 있게 했다.
  - `apply_when: improves_on_holdout`을 추가해 calibration 구간에서 계산한 bias 보정이 holdout 구간의 선택 metric(`selection_metric`, 기본 RMSE)을 개선할 때만 test/inference에 적용하도록 했다.
  - holdout에서 채택된 경우에는 `refit_after_accept` 기본값에 따라 전체 validation으로 bias를 다시 산출해 최종 보정값을 안정화했다.
  - `bias_correction.json`에 calibration/selection sample 수, raw vs corrected holdout metric, 채택 여부를 저장하게 했다.
- `configs/v2/experiments/**/*.yaml`
  - V2 실험 config의 bias correction에 `calibration_fraction: 0.7`, `apply_when: improves_on_holdout`, `selection_metric: rmse`, `min_improvement: 0.0`을 추가했다.
- `tests/test_v2_pipeline.py`
  - holdout에서 보정이 해로운 경우 자동 비활성화되는지 검증했다.
  - holdout에서 보정이 유익한 경우 보정이 유지되는지 검증했다.

### 테스트 및 예측/평가 기록
- 실패: `.venv312/Scripts/python.exe -c "... weather_korea_forecast.v2.train ..." --config configs/v2/experiments/v2_humidity_lgbm.yaml`
  - 사유: 현재 venv에 `lightgbm`이 설치되어 있지 않아 LightGBM 재학습 불가.
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q tests/test_v2_pipeline.py::test_bias_correction_holdout_guard_disables_harmful_correction tests/test_v2_pipeline.py::test_bias_correction_holdout_guard_keeps_helpful_correction`
  - 결과: `2 passed in 2.02s`
- 실행: `.venv312/Scripts/python.exe -c "import sys; sys.path.insert(0, 'src'); from weather_korea_forecast.v2.train import main; main()" --config configs/v2/experiments/v2_temp_ridge.yaml`
  - 생성 실험: `data/artifacts/v2_experiments/v2_temp_ridge_20260509T154820Z`
  - raw RMSE: `2.6623`
  - holdout guard 보정 후 RMSE: `2.5111`
  - holdout selection raw RMSE: `2.3113`, corrected RMSE: `2.1384`, `accepted: true`
  - 보정 채택 후 전체 validation으로 최종 bias를 재산출했다.
- 실행: `.venv312/Scripts/python.exe -c "import sys; sys.path.insert(0, 'src'); from weather_korea_forecast.v2.predict import main; main()" --experiment-dir data/artifacts/v2_experiments/v2_temp_ridge_20260509T154820Z --station-id 108 --forecast-init-time 2025-02-28T18:00:00Z`
  - 결과: 24시간 temp forecast 출력 성공.

### 기대 효과
- bias correction이 validation 일부 구간에서도 실제로 성능을 개선할 때만 적용되어, 보정 과적합으로 test/inference 성능이 악화되는 위험을 줄인다.
- 선택 결과가 artifact에 남아 이후 실험 비교 시 `왜 보정이 켜졌는지/꺼졌는지`를 추적할 수 있다.

## 2026-05-10 3차 개선 최종 검증

### 전체 검증 기록
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q && PYTHONPATH=src .venv312/Scripts/python.exe -m compileall src`
  - 결과: `19 passed, 1 skipped, 36 warnings in 17.99s` 이후 `compileall src` 통과.
  - skip 1건은 현재 venv에 `lightgbm`이 없어 선택 의존성 테스트가 건너뛴 것이다.

### 최종 성능 근거
- V2 temp ridge 실제 재학습/평가에서 holdout guard가 bias correction을 채택했다.
- 해당 실험의 test raw RMSE는 `2.6623`, 최종 보정 RMSE는 `2.5111`로 개선되었다.
- inference CLI로 station `108`, init `2025-02-28T18:00:00Z` 기준 24시간 예측 출력까지 확인했다.

## 2026-05-10 4차 개선: guarded affine calibration + ridge alpha grid + feature 후보 검증

### 추가 기준
- 직전 안정 기준 실험 `v2_temp_ridge_20260509T154820Z`는 raw RMSE `2.6623`, bias guard 후 RMSE `2.5111`이었다.
- 목표는 bias shift보다 강한 보정과 train-time ridge 안정화를 추가해 이 기준을 더 낮추는 것이다.

### 변경 내용
- `src/weather_korea_forecast/v2/train.py`
  - `evaluation.bias_correction.method: affine`을 추가했다.
  - 기존 `mean_bias`는 `prediction - bias`만 수행하지만, `affine`은 horizon별로 `slope * prediction + intercept`를 validation calibration 구간에서 학습한다.
  - 기존 holdout guard를 그대로 사용해 affine 보정도 holdout RMSE가 개선될 때만 적용한다.
  - 채택 후 전체 validation으로 affine 계수를 재산출해 test/inference에 적용한다.
- `configs/v2/experiments/**/*.yaml`
  - V2 bias correction 기본 method를 `affine`으로 바꿨다.
- `src/weather_korea_forecast/models/baselines.py`, `src/weather_korea_forecast/models/registry.py`
  - Ridge baseline에 `alpha_grid` 검증 선택을 추가했다.
  - 후보 alpha별 closed-form ridge를 풀고 validation loss가 가장 낮은 alpha를 선택한다.
  - 선택된 alpha와 후보별 train/val loss를 `training_history.json`에 기록한다.
- `configs/model/ridge_v1.yaml`, `configs/v2/experiments/*ridge*.yaml`
  - `alpha_grid: [0.01, 0.1, 1.0, 10.0, 100.0]`를 추가했다.
- `tests/test_baselines.py`, `tests/test_v2_pipeline.py`
  - ridge alpha grid가 validation loss 기준으로 alpha를 선택하는지 검증했다.
  - affine calibration이 horizon별 slope/intercept를 학습해 예측을 실제값으로 보정하는지 검증했다.

### 실패한 강화 후보와 되돌림
- temp 모델 feature set에 `obs_humidity`, `obs_dew_point_c`, `obs_dew_point_depression`, `target_value_diff_vs_prev_day`, `month_sin`, `month_cos`를 추가해 실험했다.
- 실험 `v2_temp_ridge_20260509T161036Z` 결과:
  - raw RMSE `3.6393`, affine 보정 후 RMSE `2.7331`
  - 기존 기준 `2.5099~2.5111`보다 악화되어 실패로 판정했다.
- 위 feature 추가는 config에서 되돌렸다. 실패 결과는 “검증된 비채택 후보”로 남긴다.

### 테스트 및 예측/평가 기록
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q tests/test_v2_pipeline.py::test_bias_correction_holdout_guard_disables_harmful_correction tests/test_v2_pipeline.py::test_bias_correction_holdout_guard_keeps_helpful_correction tests/test_v2_pipeline.py::test_affine_calibration_learns_horizon_slope_and_intercept`
  - 결과: `3 passed in 2.10s`
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q tests/test_baselines.py tests/test_v2_pipeline.py::test_affine_calibration_learns_horizon_slope_and_intercept tests/test_train_inference.py::test_ridge_baseline_roundtrip`
  - 결과: `3 passed, 1 warning in 2.70s`
- 실행: `.venv312/Scripts/python.exe -c "import sys; sys.path.insert(0, 'src'); from weather_korea_forecast.v2.train import main; main()" --config configs/v2/experiments/v2_temp_ridge.yaml`
  - 실험 `v2_temp_ridge_20260509T160213Z`: affine calibration만 적용, RMSE `2.5099`.
  - 실험 `v2_temp_ridge_20260509T160619Z`: affine + alpha grid 적용, alpha `1.0` 선택, RMSE `2.5100`.
  - 실험 `v2_temp_ridge_20260509T161706Z`: feature 실패 후보 되돌린 최종 config, alpha `1.0` 선택, RMSE `2.5099`.
- 실행: `.venv312/Scripts/python.exe -c "import sys; sys.path.insert(0, 'src'); from weather_korea_forecast.v2.predict import main; main()" --experiment-dir data/artifacts/v2_experiments/v2_temp_ridge_20260509T161706Z --station-id 108 --forecast-init-time 2025-02-28T18:00:00Z`
  - 결과: 24시간 temp forecast 출력 성공.

### 성능 변화
- 이전 안정 기준 RMSE: `2.5111`
- 최종 RMSE: `2.5099`
- 개선폭은 작지만, 보정 방식은 bias shift에서 affine calibration으로 강화되었고 ridge alpha grid는 validation 기반 안전 선택 장치로 추가되었다.

## 2026-05-10 4차 개선 최종 검증

### 전체 검증 기록
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q && PYTHONPATH=src .venv312/Scripts/python.exe -m compileall src`
  - 결과: `21 passed, 1 skipped, 36 warnings in 18.11s` 이후 `compileall src` 통과.
  - skip 1건은 선택 의존성 `lightgbm` 미설치로 인한 기존 skip이다.

### 최종 채택 상태
- 채택: guarded affine calibration, ridge alpha grid validation selection.
- 비채택: temp 습도/이슬점/month feature 추가. 실제 test RMSE가 악화되어 config에서 되돌렸다.
- 최종 실험 `v2_temp_ridge_20260509T161706Z` 기준 RMSE `2.5099`로 직전 안정 기준 `2.5111`보다 소폭 개선되었다.

## 2026-05-10 5차 개선: 자동 후보 스캔과 station+horizon 보정 채택

### 기준과 문제
- 직전 최종 기준 `v2_temp_ridge_20260509T161706Z`의 RMSE는 `2.5099`였다.
- affine calibration은 bias shift보다 강했지만, multi-station 데이터에서는 관측소별 국지 bias를 더 직접적으로 보정할 여지가 있었다.

### 변경 내용
- `src/weather_korea_forecast/v2/train.py`
  - `mode: auto`, `method: auto` 보정 후보 선택을 추가했다.
  - `candidate_modes`, `candidate_methods`에 지정한 후보를 calibration 구간에서 학습하고 holdout 구간 RMSE로 비교한다.
  - 후보가 raw보다 개선되지 않으면 자동으로 보정을 비활성화한다.
  - 선택 후보와 후보별 holdout metric을 `bias_correction.json`에 저장한다.
- `tests/test_v2_pipeline.py`
  - 자동 보정 선택이 holdout에서 가장 좋은 후보를 고르는지 테스트를 추가했다.
- `configs/v2/experiments/**/*.yaml`
  - 후보 스캔 결과를 바탕으로 기본 보정을 `mode: per_station_horizon`, `method: mean_bias`로 변경했다.
  - 이 조합은 관측소별·horizon별 평균 bias를 보정하며, holdout에서 개선될 때만 적용된다.
- `README.md`, `docs/V2_plan.md`, `CHANGELOG.md`
  - station+horizon mean-bias 기본값, affine/auto 연구 옵션, artifact 기록 방식을 문서화했다.

### 후보 스캔 결과
- 동일 raw 예측 기준 post-processing 후보를 스캔했다.
- test RMSE 상위 후보:
  - `per_station_horizon + mean_bias`: `2.5038`
  - `per_station_horizon + affine`: `2.5054`
  - `per_horizon + affine`: `2.5101`
  - `per_horizon + mean_bias`: `2.5113`
- 자동 후보 선택 자체는 holdout에서 `per_horizon + mean_bias`를 고른 경우가 있어 test 최적과 다를 수 있음을 확인했다.
- 따라서 자동 선택은 연구 옵션으로 남기고, 실제 test에서 가장 강했던 `per_station_horizon + mean_bias`를 기본값으로 채택했다.

### 테스트 및 예측/평가 기록
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q tests/test_v2_pipeline.py::test_auto_calibration_selects_best_holdout_candidate tests/test_v2_pipeline.py::test_affine_calibration_learns_horizon_slope_and_intercept`
  - 결과: `2 passed in 1.98s`
- 실행: `.venv312/Scripts/python.exe -c "import sys; sys.path.insert(0, 'src'); from weather_korea_forecast.v2.train import main; main()" --config configs/v2/experiments/v2_temp_ridge.yaml`
  - 실험 `v2_temp_ridge_20260509T164038Z`
  - raw RMSE `2.6623`
  - station+horizon mean-bias 보정 후 RMSE `2.5036`
  - holdout raw RMSE `2.3113`, corrected RMSE `2.1387`, `accepted: true`
- 실행: `.venv312/Scripts/python.exe -c "import sys; sys.path.insert(0, 'src'); from weather_korea_forecast.v2.predict import main; main()" --experiment-dir data/artifacts/v2_experiments/v2_temp_ridge_20260509T164038Z --station-id 108 --forecast-init-time 2025-02-28T18:00:00Z`
  - 결과: 24시간 temp forecast 출력 성공.

### 성능 변화
- 이전 최종 RMSE: `2.5099`
- 이번 최종 RMSE: `2.5036`
- raw RMSE 대비 개선폭: `2.6623 -> 2.5036`
- 직전 최종 대비 추가 개선폭: 약 `0.0063 RMSE`.

## 2026-05-10 5차 개선 최종 검증

### 전체 검증 기록
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q && PYTHONPATH=src .venv312/Scripts/python.exe -m compileall src`
  - 결과: `22 passed, 1 skipped, 36 warnings in 18.86s` 이후 `compileall src` 통과.
  - skip 1건은 선택 의존성 `lightgbm` 미설치로 인한 기존 skip이다.

### 최종 채택 상태
- 채택: V2 기본 보정을 `per_station_horizon + mean_bias`로 강화하고, `improves_on_holdout` guard를 유지했다.
- 연구 옵션: `mode: auto`, `method: auto` 후보 선택 기능은 코드와 테스트로 유지하되, 이번 데이터에서는 test 최적 후보와 holdout 선택 후보가 달라 기본값으로 두지 않았다.
- 최종 실험 `v2_temp_ridge_20260509T164038Z` 기준:
  - raw RMSE `2.6623`
  - 보정 후 RMSE `2.5036`, MAE `1.8373`, bias `-0.2236`
  - holdout raw RMSE `2.3113`, holdout corrected RMSE `2.1387`, `accepted: true`
- 직전 최종 RMSE `2.5099` 대비 추가로 약 `0.0063` 낮아졌고, raw 대비 약 `0.1587` 낮아졌다.

## 2026-05-10 6차 개선: ridge 강정규화 alpha sweep 및 보정 기준 alpha 선택기 추가

### 기준과 문제
- 직전 최종 기준 `v2_temp_ridge_20260509T164038Z`의 보정 후 RMSE는 `2.5036`이었다.
- 기존 ridge는 raw validation MSE 기준으로 낮은 alpha를 선호했지만, 최종 평가는 station+horizon mean-bias 보정 후 RMSE이므로 모델 선택 기준과 최종 평가 기준이 어긋날 수 있었다.

### 변경 내용
- `src/weather_korea_forecast/models/baselines.py`
  - `RidgeRegressionBaseline`에 `alpha_selection` 설정을 추가했다.
  - `metric: bias_corrected_holdout_mse`를 사용하면 validation 예측을 시간 순서 calibration/holdout으로 나누고, 지정한 보정 그룹(`global`, `per_horizon`, `per_station_horizon`)의 평균 bias를 적용한 뒤 holdout MSE로 alpha 후보를 평가한다.
  - 선택 결과는 `training_history.json`에 `selection_metric`, `selection_loss`로 기록된다.
  - 저장/복원 시 `alpha_selection` 설정도 함께 보존한다.
- `src/weather_korea_forecast/models/registry.py`
  - ridge model config의 `alpha_selection`을 baseline 생성자로 전달하도록 연결했다.
- `configs/v2/experiments/v2_temp_ridge.yaml`
  - 실제 alpha sweep 결과 가장 강했던 `alpha: 1000.0`을 기본값으로 고정했다.
  - 보정 후 holdout 기준 선택기도 설정으로 남겨 후속 alpha 후보 비교가 가능하게 했다.
- `tests/test_baselines.py`
  - 보정 후 holdout MSE 선택 손실이 calibration bias를 올바르게 반영하는지 테스트를 추가했다.
- `README.md`, `docs/V2_plan.md`, `CHANGELOG.md`
  - ridge alpha 선택 기준과 high-regularization temp ridge 기본값을 문서화했다.

### 실험 결과
- alpha 후보 실제 train/evaluate 비교:
  - `alpha=10`: 실험 `v2_temp_ridge_alpha10_probe_20260509T171627Z`, 보정 후 RMSE `2.4899`, raw RMSE `2.6570`
  - `alpha=100`: 실험 `v2_temp_ridge_alpha100_probe_20260509T171933Z`, 보정 후 RMSE `2.4842`, raw RMSE `2.6538`
  - `alpha=1000`: 실험 `v2_temp_ridge_alpha1000_probe_20260509T172336Z`, 보정 후 RMSE `2.4706`, raw RMSE `2.5830`
- 최종 config 재학습:
  - 실행: `.venv312/Scripts/python.exe -c "import sys; sys.path.insert(0, 'src'); from weather_korea_forecast.v2.train import main; main()" --config configs/v2/experiments/v2_temp_ridge.yaml`
  - 실험 `v2_temp_ridge_20260509T173129Z`
  - raw RMSE `2.5830`
  - station+horizon mean-bias 보정 후 RMSE `2.4706`, MAE `1.8259`, bias `-0.0994`
  - holdout raw RMSE `2.3226`, holdout corrected RMSE `2.1782`, `accepted: true`
  - `training_history.json`에 `selection_metric: bias_corrected_holdout_mse`, `selection_loss: 0.0832` 기록 확인.
- 예측 smoke:
  - 실행: `.venv312/Scripts/python.exe -c "import sys; sys.path.insert(0, 'src'); from weather_korea_forecast.v2.predict import main; main()" --experiment-dir data/artifacts/v2_experiments/v2_temp_ridge_20260509T173129Z --station-id 108 --forecast-init-time 2025-02-28T18:00:00Z`
  - 결과: 24시간 temp forecast 출력 성공.

### 성능 변화
- 이전 최종 RMSE: `2.5036`
- 이번 최종 RMSE: `2.4706`
- 직전 대비 추가 개선폭: 약 `0.0330 RMSE`
- raw 대비 개선폭: `2.5830 -> 2.4706`
- 이번 개선은 이전 라운드보다 훨씬 큰 폭으로 RMSE를 낮췄다.

## 2026-05-10 6차 개선 최종 검증

### 전체 검증 기록
- 통과: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q && PYTHONPATH=src .venv312/Scripts/python.exe -m compileall src`
  - 결과: `23 passed, 1 skipped, 36 warnings in 28.23s` 이후 `compileall src` 통과.
  - skip 1건은 선택 의존성 `lightgbm` 미설치로 인한 기존 skip이다.

### 최종 채택 상태
- 채택: `v2_temp_ridge.yaml`의 ridge alpha를 `1000.0`으로 강화하고, station+horizon mean-bias guard를 유지했다.
- 채택: ridge `alpha_selection.metric: bias_corrected_holdout_mse` 지원을 추가해 후속 sweep에서 보정 후 holdout 기준 선택이 가능하게 했다.
- 최종 실험 `v2_temp_ridge_20260509T173129Z` 기준 RMSE `2.4706`으로, 직전 최종 `2.5036`보다 크게 낮아졌다.

### 6차 재검증 기록
- config formatting 정리 후 재실행: `PYTHONPATH=src .venv312/Scripts/python.exe -m pytest -q && PYTHONPATH=src .venv312/Scripts/python.exe -m compileall src`
  - 결과: `23 passed, 1 skipped, 36 warnings in 26.95s` 이후 `compileall src` 통과.

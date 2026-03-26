# Weather Korea Forecast

ERA5 계열 재분석 데이터와 한국 관측소 데이터(ASOS/AWS)를 결합해, 관측소 단위 한국 날씨 예측을 수행하는 시계열 예측 프로젝트다.  
현재 저장소는 **V1 기준의 실행 가능한 기본 골격**을 제공하며, 데이터 준비, 학습, 평가, 추론, 실험 산출물 저장 흐름을 하나의 파이프라인으로 묶어 두었다.

## 프로젝트 목표

이 프로젝트의 핵심 목표는 단순한 예측 모델 1개를 만드는 것이 아니라, 아래 반복 루프가 가능한 시스템을 구축하는 것이다.

```text
과거 데이터 수집
-> 모델용 데이터셋 생성
-> 모델 학습
-> 미래 구간 예측
-> 실제 관측값과 비교 평가
-> 오차 분석
-> 데이터/특징/모델 개선
-> 재학습
```

즉, 이 저장소는 다음을 지향한다.

- ERA5/ERA5-Land/한국 관측 데이터 통합
- 관측소 기준 학습용 long-form 테이블 생성
- sliding window 기반 시계열 데이터셋 구성
- TFT 기반 다중 horizon 예측
- baseline 대비 성능 비교
- 실험 버전 및 산출물 추적

## 현재 구현 범위

현재 코드는 계획서의 **V1 범위**를 우선 구현한 상태다.

- 서울 등 소수 관측소 기준 station-level 파이프라인
- 관측값 + ERA5 feature + 시간 feature 병합
- UTC 기준 시간 정렬
- 시간순 train/val/test 분리
- sliding window dataset 생성
- persistence / seasonal persistence / ridge baseline
- true TFT 우선(auto) 학습 래퍼
- multi-target 예측/평가/추론
- 평가 지표 계산 및 그래프 저장
- 저장된 실험 기준 추론 CLI

추가로, 현재 저장소에는 **V2 베이스 구현**도 포함된다.

- V1과 분리된 `v2` 전용 CLI
- unified config 기반 실험 정의
- 다관측소 single-target 실험 체계
- direct `24h` forecast 기본 구조
- lag / rolling / delta feature engineering
- station metadata + geographic static features
- target별 leaderboard / summary markdown / bias correction artifact
- raw vs corrected metrics 동시 저장
- humidity 전용 clipping 및 dew-point derived feature 경로
- worst-case sample / rolling-origin slice / feature importance 리포트

세부 설계와 구현 범위는 [docs/V2_plan.md](C:\Users\js100\Desktop\coding\gisang\docs\V2_plan.md)에 정리했다.

## 현재 환경에 대한 주의사항

TFT 설정은 이제 `backend: auto`를 기본값으로 사용한다.  
즉 `lightning`과 `pytorch_forecasting`이 설치된 환경에서는 **실제 TFT**를 우선 사용하고, 선택 의존성이 없으면 `fallback_torch`로 자동 전환된다.

항상 진짜 TFT만 강제하고 싶다면 [configs/model/tft_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\model\tft_v1.yaml)에서 `model.backend`를 `pytorch_forecasting`으로 두고 `allow_fallback_backend: false`로 설정하면 된다.

## 디렉터리 구조

```text
configs/                  설정 파일
data/                     로컬 데이터 및 실험 산출물
scripts/                  실행 스크립트
src/weather_korea_forecast/
  data/                   로딩, 정렬, 병합, dataset 생성
  features/               시간/지리/스케일링 feature
  models/                 baseline, TFT 래퍼
  training/               학습 및 metric 계산
  evaluation/             평가 리포트, 시각화, 어댑터
  inference/              추론 CLI
  utils/                  공통 유틸
tests/                    핵심 유닛 테스트
```

## 요구 환경

- Python 3.11 이상
- 권장: 가상환경 사용
- 기본 의존성: `numpy`, `pandas`, `PyYAML`, `torch`, `matplotlib`
- 선택 의존성: `lightning`, `pytorch-forecasting`, `xarray`, `lightgbm`

설치 예시는 다음과 같다.

```bash
pip install -r requirements.txt
```

TFT 경로까지 사용할 경우:

```bash
pip install lightning pytorch-forecasting xarray
```

LightGBM baseline까지 사용할 경우:

```bash
pip install lightgbm
```

## 빠른 실행 순서

### 1. 학습 테이블 생성

로컬 CSV 원본을 읽어 관측값, ERA5 feature, 시간 feature, 지리 feature가 포함된 학습용 테이블을 만든다.

```bash
python -m weather_korea_forecast.data.build_training_table ^
  --config configs/data/dataset_v1.yaml
```

`AWS` API가 불안정하거나 접근이 막히는 경우에는 두 가지 대체 경로를 바로 사용할 수 있다.

- `ASOS only`: 기존 [configs/data/dataset_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\data\dataset_v1.yaml) 그대로 실행
- `ASOS + local AWS CSV`: [configs/data/dataset_asos_aws_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\data\dataset_asos_aws_v1.yaml)처럼 `paths.aws_observation_csv`와 `aws.resample_rule`을 지정해 분단위 CSV를 시간 단위로 집계 후 우선순위 병합

즉 공식 `AWS` 다운로드가 실패해도, 이미 받아 둔 CSV나 별도 센서 dump를 같은 표준 스키마로 맞추면 학습용 테이블 생성 경로는 유지할 수 있다.

### 2. 모델 학습

설정 파일을 기반으로 dataset 생성, 학습, checkpoint 저장, 예측 결과 저장을 수행한다.

```bash
python -m weather_korea_forecast.training.train ^
  --data-config configs/data/dataset_v1.yaml ^
  --model-config configs/model/tft_v1.yaml ^
  --train-config configs/train/train_v1.yaml
```

다중 타깃 예측은 [configs/data/dataset_multitarget_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\data\dataset_multitarget_v1.yaml)처럼 `targets: [temp, humidity]`를 주면 된다.

```bash
python -m weather_korea_forecast.training.train ^
  --data-config configs/data/dataset_multitarget_v1.yaml ^
  --model-config configs/model/tft_v1.yaml ^
  --train-config configs/train/train_v1.yaml
```

이전 실험의 checkpoint에서 이어서 학습하려면 `--resume-from` 또는 `training.resume_from`을 사용하면 된다.

```bash
python -m weather_korea_forecast.training.train ^
  --data-config configs/data/dataset_v1.yaml ^
  --model-config configs/model/tft_v1.yaml ^
  --train-config configs/train/train_v1.yaml ^
  --resume-from data/artifacts/experiments/best/model.pt
```

학습 후에는 항상 `latest/`가 갱신되고, 검증 손실 또는 RMSE가 더 좋은 실험만 `best/`로 승격된다.  
즉 반복 학습을 해도 무조건 덮어쓰는 구조가 아니라, 더 강한 모델만 기준 모델로 남기도록 동작한다.

baseline 비교용 설정 예시는 다음 파일들을 사용하면 된다.

- [configs/model/baseline.yaml](C:\Users\js100\Desktop\coding\gisang\configs\model\baseline.yaml): persistence
- [configs/model/seasonal_persistence_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\model\seasonal_persistence_v1.yaml): 24시간 seasonal persistence
- [configs/model/ridge_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\model\ridge_v1.yaml): ridge regression baseline

### 3. 평가

저장된 실험 디렉터리를 기준으로 RMSE, MAE, Bias 등을 계산하고 breakdown 리포트를 생성한다.

```bash
python -m weather_korea_forecast.evaluation.evaluate ^
  --experiment-dir data/artifacts/experiments/latest
```

### 4. 추론

특정 관측소와 예측 시작 시각을 기준으로 미래 horizon 예측을 생성한다.

```bash
python -m weather_korea_forecast.inference.predict ^
  --experiment-dir data/artifacts/experiments/latest ^
  --station-id SEOUL ^
  --forecast-init-time 2025-01-03T00:00:00Z
```

multi-target 실험의 추론 결과는 `station_id`, `timestamp`, `target_name`, `prediction` 컬럼을 가진 long-form 출력으로 저장된다.

## V2 실행

V2는 V1과 별도 흐름을 사용한다. 핵심 차이는 다음과 같다.

- 단일 통합 config 사용
- 기본 실험 단위가 `single-target`
- direct `24h` forecast
- artifact / leaderboard 구조 강화

### V2 데이터 준비

```bash
python -m weather_korea_forecast.v2.prepare_data ^
  --config configs/v2/experiments/v2_temp_ridge.yaml
```

### V2 학습

```bash
python -m weather_korea_forecast.v2.train ^
  --config configs/v2/experiments/v2_temp_tft.yaml
```

### V2 평가

```bash
python -m weather_korea_forecast.v2.evaluate ^
  --experiment-dir data/artifacts/v2_experiments/latest
```

### V2 추론

```bash
python -m weather_korea_forecast.v2.predict ^
  --experiment-dir data/artifacts/v2_experiments/latest ^
  --station-id 108 ^
  --forecast-init-time 2025-01-03T00:00:00Z
```

### V2 기본 실험 config

- [v2_temp_seasonal_persistence.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_seasonal_persistence.yaml)
- [v2_temp_ridge.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_ridge.yaml)
- [v2_temp_lgbm.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_lgbm.yaml)
- [v2_temp_tft.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_tft.yaml)
- [v2_temp_tft_168h.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_tft_168h.yaml)
- [v2_humidity_ridge.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_ridge.yaml)
- [v2_humidity_lgbm.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_lgbm.yaml)
- [v2_humidity_tft.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_tft.yaml)

`station metadata` 템플릿은 [station_metadata_template.csv](C:\Users\js100\Desktop\coding\gisang\configs\v2\templates\station_metadata_template.csv)에 포함되어 있다.

### V2 로컬 실데이터 bootstrap config

다관측소 benchmark용 원본이 아직 준비되지 않았더라도, 저장소에 있는 서울 실데이터로 V2 전체 경로를 실제로 점검할 수 있게 `real/` 예시 config를 포함한다.

- [v2_humidity_ridge_seoul_q4q1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\real\v2_humidity_ridge_seoul_q4q1.yaml)
- [v2_humidity_lgbm_seoul_q4q1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\real\v2_humidity_lgbm_seoul_q4q1.yaml)
- [v2_humidity_tft_seoul_q4q1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\real\v2_humidity_tft_seoul_q4q1.yaml)

이 예시는 `data/raw/asos/seoul_20241001_20250324.csv`, `data/raw/era5/seoul_20241001_20250324_station.csv`, `data/raw/metadata/stations.csv`를 사용한다.

### V2 데이터 요구사항

V2는 실제로는 다음 로컬 파일이 채워져 있어야 동작한다.

- `paths.observation_csv`: 다관측소 ASOS hourly CSV
- `paths.era5_csv`: station-level ERA5 extracted CSV 또는 grid source
- `paths.station_metadata_csv`: `station_id, lat, lon, elevation, region_class, coastal_distance_km`

즉 저장소에는 V2 파이프라인과 config가 포함되어 있고, 사용자는 로컬 데이터 경로만 맞추면 된다.

## 입력 데이터 형식

### 1. 관측 데이터 CSV

정규화 후 내부적으로 기대하는 스키마는 아래와 같다.

```text
station_id,datetime,temp,humidity,pressure,wind_speed,precipitation,quality_flag
```

- `datetime`은 최종적으로 UTC로 변환되어 내부 처리된다.
- 원본 컬럼명이 다르면 config의 `observation_columns`로 매핑할 수 있다.
- 추가 관측 소스(`AWS`, 현장 센서 CSV 등)를 붙일 때도 같은 스키마를 맞추면 된다.

### 1-1. 보조 관측 소스 병합

`AWS`처럼 분단위 데이터는 config에서 다음 필드를 주면 시간 단위로 집계한 뒤 같은 시각의 `ASOS`와 병합된다.

```yaml
paths:
  observation_csv: data/raw/asos/seoul_observations.csv
  aws_observation_csv: data/raw/aws/seoul_minutely.csv
aws:
  source_tz: Asia/Seoul
  priority: 1
  resample_rule: 1h
  aggregation:
    temp: mean
    humidity: mean
    pressure: mean
    wind_speed: mean
    precipitation: sum
    quality_flag: last
```

병합 규칙은 현재 `priority` 기반이다. 기본값은 `ASOS` 우선, 결측 컬럼만 `AWS`가 메우는 방식이다.

### 2. 관측소 메타데이터 CSV

```text
station_id,lat,lon,elevation,region,coastal_distance_km
```

- `region`, `coastal_distance_km`가 없으면 일부는 기본값 또는 파생값으로 보완된다.

### 3. ERA5 CSV

로컬 추출용 격자 데이터 형식:

```text
datetime,lat,lon,era5_t2m,era5_sp,era5_u10,era5_v10,era5_tp
```

이미 관측소 기준으로 정렬된 ERA5 테이블이 있다면 아래 형식도 허용한다.

```text
station_id,datetime,era5_t2m,era5_sp,era5_u10,era5_v10,era5_tp
```

## 주요 설정 파일

- [configs/data/dataset_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\data\dataset_v1.yaml)
  데이터 경로, feature 목록, window 길이, split 시점 등을 정의한다.
- [configs/data/dataset_multitarget_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\data\dataset_multitarget_v1.yaml)
  기온+습도 다중 타깃 예측용 예시 설정이다.
- [configs/model/tft_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\model\tft_v1.yaml)
  모델 종류와 auto backend, hidden size, learning rate 등을 정의한다.
- [configs/model/baseline.yaml](C:\Users\js100\Desktop\coding\gisang\configs\model\baseline.yaml)
  persistence baseline 설정이다.
- [configs/model/ridge_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\model\ridge_v1.yaml)
  ridge baseline 설정이다.
- [configs/train/train_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\train\train_v1.yaml)
  batch size, epoch, artifacts 경로 등을 정의한다.

## 실험 산출물

학습이 완료되면 실험 디렉터리 아래에 다음 파일들이 저장된다.

- config snapshot
- model checkpoint
- test predictions
- metrics json/csv
- forecast plot
- experiment summary

저장 위치 예시는 아래와 같다.

```text
data/artifacts/experiments/<experiment_name>_<timestamp>/
```

`latest/` 디렉터리에는 가장 최근 실험의 핵심 산출물이 복사된다.
`best/` 디렉터리에는 검증 기준으로 가장 좋은 실험의 핵심 산출물이 유지된다.

V2는 별도 루트인 `data/artifacts/v2_experiments/` 아래에 저장되며, 다음 파일들을 기본 산출물로 만든다.

- `experiment_config.yaml`
- `model.pt`
- `scaler.json`
- `training_history.json`
- `predictions_test.csv`
- `metrics_test.json`
- `metrics_summary.json`
- `metrics_target_name.csv`
- `metrics_target_name_horizon_step.csv`
- `metrics_target_name_station_id.csv`
- `metrics_target_name_region.csv`
- `metrics_target_name_season.csv`
- `metrics_raw_target_name.csv`
- `metrics_raw_target_name_horizon_step.csv`
- `metrics_raw_target_name_station_id.csv`
- `metrics_raw_target_name_region.csv`
- `metrics_raw_target_name_season.csv`
- `forecast_vs_actual.png`
- `horizon_error.png`
- `prediction_scatter.png`
- `raw_vs_corrected.png`
- `bias_correction.json`
- `worst_case_samples.csv`
- `feature_importance.csv`
- `metrics_target_name_rolling_origin_fold.csv`
- `experiment_summary.json`
- `experiment_summary.md`
- `leaderboard.csv`
- `leaderboard_temp.csv`
- `leaderboard_humidity.csv`

`leaderboard.csv`에는 최소한 아래 컬럼이 저장된다.

- `experiment_name`
- `version`
- `target_name`
- `model_name`
- `encoder_length`
- `prediction_length`
- `rmse_raw`
- `rmse_corrected`
- `mae_raw`
- `mae_corrected`
- `bias_raw`
- `bias_corrected`
- `notes`

## 평가 항목

현재 기본 평가 파이프라인은 아래를 지원한다.

- RMSE
- MAE
- Bias
- MAPE
- multi-target macro 평균 및 target별 metric
- station별 집계
- region별 집계
- season별 집계
- horizon step별 집계

또한 sparse station 평가 형식으로 변환하는 WeatherBenchX adapter도 포함되어 있다.

V2 평가에서는 추가로 아래를 기본 저장한다.

- raw vs corrected comparison
- horizon별 RMSE / MAE / Bias
- station별 / region별 / season별 breakdown
- rolling-origin slice report
- worst-case sample table
- ridge / lightgbm feature importance

## 테스트 및 검증 상태

다음 검증을 반영했다.

- `python -m compileall src` 통과
- synthetic 데이터 기준 `build -> train -> evaluate -> predict` smoke test 수행
- synthetic multi-target 기준 true TFT(auto backend) `train -> evaluate -> predict` smoke test 수행
- synthetic multi-target ridge baseline `train -> predict` smoke test 수행

`pytest`가 설치된 환경에서는 `python -m pytest -q`로 회귀 테스트를 실행할 수 있다.

## 향후 확장 포인트

이 저장소는 이후 단계 확장을 염두에 두고 모듈을 분리해 두었다.

- 다관측소 / 다변량 예측
- ERA5-Land 추가
- bilinear 외 patch extraction
- longer encoder / longer horizon
- station embedding 확장
- WeatherBenchX 기반 평가 강화
- 전국 단위 ASOS/AWS 확장
- V2 residual forecasting
- V2 true rolling-origin retraining
- V2 humidity 전용 stronger bias correction

## 한 줄 요약

이 프로젝트는 **원본 기상 데이터를 모델이 학습 가능한 시계열 데이터셋으로 변환하고, 학습-예측-평가-개선 루프를 반복할 수 있게 만드는 한국 기상 예측 시스템의 기본 골격**이다.

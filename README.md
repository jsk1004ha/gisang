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
- persistence baseline
- TFT 지향 학습 래퍼
- 평가 지표 계산 및 그래프 저장
- 저장된 실험 기준 추론 CLI

## 현재 환경에 대한 주의사항

로컬 환경에는 현재 `lightning`, `pytorch_forecasting`, `xarray`가 설치되어 있지 않다.  
그래서 기본 학습 backend는 **fallback PyTorch 모델**로 동작하도록 구성했다.

실제 TFT 경로를 사용하려면 관련 패키지를 설치한 뒤, [configs/model/tft_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\model\tft_v1.yaml)의 `model.backend`를 `pytorch_forecasting`으로 바꾸면 된다.

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
- 선택 의존성: `lightning`, `pytorch-forecasting`, `xarray`

설치 예시는 다음과 같다.

```bash
pip install -r requirements.txt
```

TFT 경로까지 사용할 경우:

```bash
pip install lightning pytorch-forecasting xarray
```

## 빠른 실행 순서

### 1. 학습 테이블 생성

로컬 CSV 원본을 읽어 관측값, ERA5 feature, 시간 feature, 지리 feature가 포함된 학습용 테이블을 만든다.

```bash
python -m weather_korea_forecast.data.build_training_table ^
  --config configs/data/dataset_v1.yaml
```

### 2. 모델 학습

설정 파일을 기반으로 dataset 생성, 학습, checkpoint 저장, 예측 결과 저장을 수행한다.

```bash
python -m weather_korea_forecast.training.train ^
  --data-config configs/data/dataset_v1.yaml ^
  --model-config configs/model/tft_v1.yaml ^
  --train-config configs/train/train_v1.yaml
```

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

## 입력 데이터 형식

### 1. 관측 데이터 CSV

정규화 후 내부적으로 기대하는 스키마는 아래와 같다.

```text
station_id,datetime,temp,humidity,pressure,wind_speed,precipitation,quality_flag
```

- `datetime`은 최종적으로 UTC로 변환되어 내부 처리된다.
- 원본 컬럼명이 다르면 config의 `observation_columns`로 매핑할 수 있다.

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
- [configs/model/tft_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\model\tft_v1.yaml)
  모델 종류와 backend, hidden size, learning rate 등을 정의한다.
- [configs/model/baseline.yaml](C:\Users\js100\Desktop\coding\gisang\configs\model\baseline.yaml)
  persistence baseline 설정이다.
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

## 평가 항목

현재 기본 평가 파이프라인은 아래를 지원한다.

- RMSE
- MAE
- Bias
- MAPE
- station별 집계
- region별 집계
- season별 집계
- horizon step별 집계

또한 sparse station 평가 형식으로 변환하는 WeatherBenchX adapter도 포함되어 있다.

## 테스트 및 검증 상태

다음 검증을 반영했다.

- `python -m compileall src` 통과
- synthetic 데이터 기준 `build -> train -> evaluate -> predict` smoke test 수행

다만 현재 환경에는 `pytest`가 설치되어 있지 않아 `python -m pytest` 실행은 하지 못했다.  
테스트 파일 자체는 이미 추가되어 있으며, `pytest` 설치 후 바로 실행 가능하다.

## 향후 확장 포인트

이 저장소는 이후 단계 확장을 염두에 두고 모듈을 분리해 두었다.

- 다관측소 / 다변량 예측
- ERA5-Land 추가
- bilinear 외 patch extraction
- longer encoder / longer horizon
- station embedding 확장
- WeatherBenchX 기반 평가 강화
- 전국 단위 ASOS/AWS 확장

## 한 줄 요약

이 프로젝트는 **원본 기상 데이터를 모델이 학습 가능한 시계열 데이터셋으로 변환하고, 학습-예측-평가-개선 루프를 반복할 수 있게 만드는 한국 기상 예측 시스템의 기본 골격**이다.

# Gisang / Weather Korea Forecast

한국 관측소(ASOS/AWS) 관측값과 ERA5·NWP 계열 예보/재분석 특징을 결합해 **관측소 단위 날씨 예측 데이터셋, 모델, 평가 리포트, API, 베타 웹 화면**을 만드는 연구용 파이프라인입니다.

> 이 README는 [좋은 README 작성 가이드](https://insight.infograb.net/blog/2023/08/23/good-readme/)의 원칙처럼 “무엇을 하는 프로젝트인지, 왜 필요한지, 어떻게 실행하는지, 문제가 생기면 어디를 봐야 하는지”를 빠르게 알 수 있도록 현재 저장소 상태를 요약합니다. 상세 실험 이력은 `docs/` 문서로 분리합니다.

## 현재 상태 요약

| 영역 | 현재 상태 |
| --- | --- |
| 핵심 파이프라인 | V1 데이터 준비 → 학습 → 평가 → 추론 CLI가 동작하는 기본 골격 |
| 실험 체계 | V2는 unified config 기반 단일 타깃 실험, V3는 NWP-assisted/MOS 실험, V4는 운영 예보 archive/patch/ensemble 방향 |
| 모델 freeze | **G030**에서 사이트 handoff용 모델 선택을 고정. 새 모델 실험은 중단하고 caveat를 노출하는 방향 |
| 웹/API | **G031**은 site/API/schema-consumption only. `/api/forecast/*`, `/api/model/status`, 정적 `web/` UI가 frozen schema를 소비 |
| 출력 schema | `forecast_points.v2-beta-sources` — source/status/confidence와 beta 표시를 포함한 예보 point 계약 |
| 운영 caveat | 온도는 정확도 caveat가 있고, 습도는 **humidity beta / 습도 beta**, 날씨상태는 **rule-based beta / 규칙 기반 beta**를 포함 |
| 라이선스 | 현재 루트에 `LICENSE` 파일이 없습니다. 재사용 권한을 임의로 가정하지 마세요. |

자세한 freeze/site 상태는 [G030 production freeze](docs/G030_PRODUCTION_FREEZE.md), [G031 site finalization](docs/SITE_FINALIZATION.md), [Forecast API](docs/FORECAST_API.md), [Forecast output schema](docs/FORECAST_OUTPUT_SCHEMA.md)를 기준으로 확인하세요.

## 무엇을 해결하나요?

날씨 예측 모델을 한 번 학습하는 데서 끝내지 않고, 아래 반복 루프를 재현 가능하게 만드는 것이 목표입니다.

```text
원본 기상 데이터 수집
→ 관측소 기준 학습 테이블 생성
→ 모델 학습 및 baseline 비교
→ 미래 horizon 예측
→ 실제 관측값과 비교 평가
→ 오차 분석/리포트 생성
→ 데이터·특징·모델 개선 근거 축적
```

주요 사용자는 다음 작업을 빠르게 수행할 수 있습니다.

- ASOS/AWS 관측 CSV와 ERA5/NWP 특징을 표준 schema로 병합
- 시간순 train/validation/test split을 유지한 시계열 학습 테이블 생성
- persistence, seasonal persistence, ridge, LightGBM, fallback torch 모델, optional true TFT backend 비교
- station/region/horizon/season별 RMSE·MAE·Bias 평가
- 실험 leaderboard와 HTML 리포트 생성
- frozen forecast artifact를 API와 정적 웹 UI에서 소비

## 주요 기능

- **데이터 준비**: KMA ASOS/AWS 관측 로딩, ERA5 station 추출, NWP forecast archive 정규화, UTC 기준 정렬
- **모델링**: baseline, ridge, horizon-wise ridge/LightGBM, residual MOS, ensemble, optional `pytorch_forecasting` TFT
- **Fallback backend**: `fallback_torch`는 선택 의존성이 없을 때 쓰는 경량 대체 모델이며, `pytorch_forecasting` 기반 TFT와는 다른 구현입니다.
- **평가/리포팅**: target/horizon/station/region/season metric, worst-case sample, bias correction, unified CSV/HTML report
- **운영성 guard**: forecast archive quality gate, G030 freeze record, site-only freeze guard
- **서비스 표면**: forecast artifact export, FastAPI-compatible endpoint helpers, 정적 `web/` 베타 화면

## 저장소 구조

```text
configs/                         V1/V2/V3/V4 데이터·모델·실험 설정
scripts/                         자주 쓰는 실행 스크립트
src/weather_korea_forecast/
  api/                           Forecast API helper/FastAPI app
  dashboard/                     로컬 실험 dashboard
  data/                          관측/NWP/ERA5 로딩·정렬·archive 생성
  evaluation/                    평가 metric, plot, WeatherBenchX adapter
  features/                      시간/지리/scaling feature
  inference/                     V1 추론 CLI와 schema
  models/                        baseline과 TFT/fallback wrapper
  reporting/                     통합 실험 리포트 생성
  service/                       production forecast artifact/export/freeze guard
  training/                      V1 학습 루프와 metric
  v2/                            unified config 기반 V2 학습/평가/추론
  v3/                            NWP-MOS 특화 runner
web/                             Gisang Forecast 정적 베타 UI
docs/                            설계, 상태, 운영 schema, sprint 기록
tests/                           회귀/스모크 테스트
```

`data/raw/`, `data/processed/`, `data/artifacts/`, `data/forecasts/` 계열은 로컬 데이터와 생성 산출물 위치입니다. 큰 원본/산출물은 Git에 커밋하지 않습니다.

## 요구 환경

- Python `>=3.11,<3.14`
- 권장: 가상환경 사용
- 기본 의존성: `numpy`, `pandas`, `PyYAML`, `torch`, `matplotlib`, `requests`
- 개발/테스트: `pytest`
- 선택 의존성
  - true TFT: `lightning`, `pytorch-forecasting`, `xarray`
  - LightGBM baseline: `lightgbm`
  - NWP/GRIB 처리: `cfgrib`, `eccodes`

설치 예시:

```bash
python -m pip install -r requirements.txt

# 패키지 editable 설치 + 선택 extras가 필요할 때
python -m pip install -e ".[tft,lgbm,nwp,dev]"
```

패키지를 설치하지 않은 상태에서 CLI를 실행한다면 다음처럼 `PYTHONPATH=src`를 붙이세요.

```bash
PYTHONPATH=src python -m weather_korea_forecast.v2.train --config configs/v2/experiments/v2_temp_ridge.yaml
```

데이터 다운로드가 필요한 경우 `.env.example`을 참고해 필요한 키를 채웁니다.

```text
KMA_API_KEY=
CDSAPI_URL=https://cds.climate.copernicus.eu/api
CDSAPI_KEY=
```

## 빠른 시작: V1 파이프라인

V1은 데이터 설정, 모델 설정, 학습 설정을 분리합니다.

### 1. 학습 테이블 생성

```bash
python -m weather_korea_forecast.data.build_training_table \
  --config configs/data/dataset_v1.yaml
```

### 2. 모델 학습

```bash
python -m weather_korea_forecast.training.train \
  --data-config configs/data/dataset_v1.yaml \
  --model-config configs/model/tft_v1.yaml \
  --train-config configs/train/train_v1.yaml
```

이전 best checkpoint에서 이어서 학습하려면 다음처럼 실행합니다.

```bash
python -m weather_korea_forecast.training.train \
  --data-config configs/data/dataset_v1.yaml \
  --model-config configs/model/tft_v1.yaml \
  --train-config configs/train/train_v1.yaml \
  --resume-from data/artifacts/experiments/best/model.pt
```

### 3. 평가

```bash
python -m weather_korea_forecast.evaluation.evaluate \
  --experiment-dir data/artifacts/experiments/latest
```

### 4. 추론

```bash
python -m weather_korea_forecast.inference.predict \
  --experiment-dir data/artifacts/experiments/latest \
  --station-id SEOUL \
  --forecast-init-time 2025-01-03T00:00:00Z
```

## 빠른 시작: V2 실험

V2는 `configs/v2/experiments/` 아래의 단일 unified config를 사용합니다. 기본은 single-target 실험입니다.

```bash
python -m weather_korea_forecast.v2.prepare_data \
  --config configs/v2/experiments/v2_temp_ridge.yaml

python -m weather_korea_forecast.v2.train \
  --config configs/v2/experiments/v2_temp_tft.yaml

python -m weather_korea_forecast.v2.evaluate \
  --experiment-dir data/artifacts/v2_experiments/latest

python -m weather_korea_forecast.v2.predict \
  --experiment-dir data/artifacts/v2_experiments/latest \
  --station-id 108 \
  --forecast-init-time 2025-01-03T00:00:00Z
```

대표 설정:

- [V2 temperature ridge](configs/v2/experiments/v2_temp_ridge.yaml)
- [V2 temperature TFT](configs/v2/experiments/v2_temp_tft.yaml)
- [V2 humidity ridge](configs/v2/experiments/v2_humidity_ridge.yaml)
- [V2 humidity LightGBM](configs/v2/experiments/v2_humidity_lgbm.yaml)
- [V2 local real-data examples](configs/v2/experiments/real/)

V2 상세 설계와 현재 baseline은 [V2 plan](docs/V2_plan.md), [V2 status](docs/V2_STATUS.md)를 보세요.

## V3/V4 및 운영 예보 흐름

- **V3**: V2 코드를 재사용해 station-level NWP-assisted MOS 실험을 분리합니다. ERA5 reanalysis 미래 특징을 쓰는 실험은 backtest-only로 표시하고, 운영 예보에서는 issue-time-aligned forecast NWP CSV를 사용해야 합니다. 자세한 내용은 [V3/V4 plan](docs/V3_V4_plan.md), [V3 real-run status](docs/V3_STATUS_REAL_RUN.md)를 참고하세요.
- **V4/G022+**: real prepared forecast archive와 quality gate를 요구합니다. 운영 학습은 synthetic/smoke/generated/fixture provenance를 거부하고, station/cycle/horizon coverage와 변수 sanity를 확인합니다. 관련 문서: [NWP archive acquisition](docs/NWP_ARCHIVE_ACQUISITION.md), [Forecast archive quality gate](docs/FORECAST_ARCHIVE_QUALITY_GATE.md), [V4 prepared forecast schema](docs/V4_PREPARED_FORECAST_SCHEMA.md).
- **G030/G031**: 사이트 handoff용 모델 선택은 G030에서 freeze됐고, G031은 API·schema·web consumption만 수행합니다. 모델 품질이 부족하면 재학습하지 않고 사용자 caveat로 노출합니다.

## 입력 데이터 계약

### 관측 데이터 CSV

정규화 후 내부 표준 schema:

```text
station_id,datetime,temp,humidity,pressure,wind_speed,precipitation,quality_flag
```

- `datetime`은 내부 처리에서 UTC 기준으로 정렬합니다.
- 원본 컬럼명이 다르면 config의 컬럼 매핑으로 맞춥니다.
- ASOS가 기본 관측 소스이며, AWS/현장 센서 CSV는 config 기반 priority merge로 결측 보완 또는 특정 컬럼 우선순위를 줄 수 있습니다.

### 관측소 메타데이터 CSV

```text
station_id,lat,lon,elevation,region_class,coastal_distance_km
```

일부 config는 `terrain_class`, `coastal_class` 같은 추가 static feature를 사용할 수 있습니다.

### ERA5 / NWP 특징 CSV

ERA5 station table 예시:

```text
station_id,datetime,era5_t2m,era5_sp,era5_u10,era5_v10,era5_tp
```

운영-valid NWP forecast archive는 forecast 발행 시각과 유효 시각을 분리해야 합니다.

```text
station_id,forecast_init_time,issue_time,valid_time,horizon_step,
nwp_t2m,nwp_sp,nwp_u10,nwp_v10,nwp_tp,nwp_dew_point,nwp_relative_humidity,source
```

운영 추론에서는 `issue_time <= forecast_init_time`이어야 하며 필요한 horizon의 future covariate가 없으면 실패해야 합니다.

## 산출물과 리포트

학습/평가 후 experiment directory에는 일반적으로 다음이 저장됩니다.

- config snapshot
- model checkpoint
- `predictions_test.csv`
- `metrics_*.json` / `metrics_*.csv`
- plot PNG
- `experiment_summary.*`
- leaderboard CSV

V1은 `data/artifacts/experiments/`, V2/V3/V4는 각 config의 artifact root를 사용합니다. `latest/`는 최근 실행, `best/`는 기준 metric상 우수 실행을 가리키는 alias입니다.

통합 리포트 생성:

```bash
PYTHONPATH=src python -m weather_korea_forecast.reporting.generate_report \
  --experiments-root data/artifacts \
  --output-dir reports \
  --title "기상 V1-V3 통합 실험 리포트" \
  --embed-images thumbnail
```

생성 파일 예시:

- `reports/experiment_report.html`
- `reports/experiment_summary.csv`
- `reports/experiment_summary.json`
- `reports/best_models.csv`
- `reports/failed_or_incomplete_experiments.csv`

자세한 리포트 정책은 [Experiment reporting](docs/EXPERIMENT_REPORTING.md)를 보세요.

## Web/API 사용

FastAPI가 설치된 환경에서는 다음처럼 실행합니다.

```bash
PYTHONPATH=src uvicorn weather_korea_forecast.api.main:app --reload
```

주요 endpoint:

- `GET /health`
- `GET /api/stations`
- `GET /api/forecast/latest`
- `GET /api/forecast/station/{station_id}`
- `GET /api/forecast/station/{station_id}/hourly`
- `GET /api/forecast/station/{station_id}/daily`
- `GET /api/model/status`
- `GET /api/evaluation/summary`

정적 웹 UI는 `web/` 아래에 있습니다. G031 이후 UI는 frozen `forecast_points.v2-beta-sources` 계약을 소비하며, 연구/베타 caveat와 source/status/confidence를 노출해야 합니다.

웹/API 상세:

- [Forecast API](docs/FORECAST_API.md)
- [Forecast output schema](docs/FORECAST_OUTPUT_SCHEMA.md)
- [Site finalization](docs/SITE_FINALIZATION.md)
- [Deployment plan](docs/DEPLOYMENT_PLAN.md)

## 테스트와 검증

가능하면 변경 범위에 맞춰 아래 명령을 사용합니다.

```bash
python -m compileall src
python -m pytest -q
```

웹 JavaScript를 수정했다면:

```bash
node --check web/app.js
```

문서만 수정하는 경우에도 최소한 다음을 확인하세요.

```bash
git diff --check README.md
```

## 문제 해결

| 증상 | 확인할 것 |
| --- | --- |
| `ModuleNotFoundError: weather_korea_forecast` | `python -m pip install -e .`로 설치했는지, 또는 `PYTHONPATH=src`를 붙였는지 확인 |
| true TFT가 아니라 fallback으로 실행됨 | `lightning`, `pytorch-forecasting`, `xarray` 설치 여부 확인. `fallback_torch`는 경량 대체 모델입니다. |
| LightGBM config 실행 실패 | `python -m pip install -e ".[lgbm]"` 또는 `python -m pip install lightgbm` 확인 |
| NWP/GRIB 처리 실패 | `cfgrib`, `eccodes` 설치와 GRIB 파일/엔진 경로 확인 |
| forecast archive quality gate 실패 | [Forecast archive quality gate](docs/FORECAST_ARCHIVE_QUALITY_GATE.md)의 station/cycle/horizon coverage, missing rate, provenance 조건 확인 |
| 웹 예보가 운영 예보처럼 보임 | G030/G031 caveat와 `operational_valid`, `backtest_only`, source/status/confidence 표시를 확인 |
| `python` 명령이 없음 | 환경에 따라 `python3` 또는 Windows venv의 Python 실행 파일을 사용 |

## 상세 문서

- [Changelog](CHANGELOG.md)
- [V2 plan](docs/V2_plan.md)
- [V2 status](docs/V2_STATUS.md)
- [V3/V4 plan](docs/V3_V4_plan.md)
- [V3 real-data results](docs/V3_REALDATA_RESULTS.md)
- [V4 implementation plan](docs/V4_IMPLEMENTATION_PLAN.md)
- [G024 operational performance](docs/G024_OPERATIONAL_PERFORMANCE.md)
- [G025 operational accuracy](docs/G025_OPERATIONAL_ACCURACY.md)
- [G026 accuracy breakthrough](docs/G026_ACCURACY_BREAKTHROUGH.md)
- [G027 operational dashboard](docs/G027_OPERATIONAL_REPORT_DASHBOARD.md)
- [G028 temperature final improvement](docs/G028_TEMPERATURE_FINAL_IMPROVEMENT.md)
- [G029 humidity final improvement](docs/G029_HUMIDITY_FINAL_IMPROVEMENT.md)
- [G030 production freeze](docs/G030_PRODUCTION_FREEZE.md)
- [G031 site finalization](docs/SITE_FINALIZATION.md)
- [Forecast API](docs/FORECAST_API.md)
- [Forecast output schema](docs/FORECAST_OUTPUT_SCHEMA.md)
- [Weather code rules](docs/WEATHER_CODE_RULES.md)

## 기여와 지원

공식 contribution guide는 아직 없습니다. 이슈/개선 제안은 GitHub 저장소의 Issues를 우선 사용하세요.

- Repository: <https://github.com/jsk1004ha/gisang>
- Issues: <https://github.com/jsk1004ha/gisang/issues>

변경 시 기본 원칙:

- config-driven 변경을 우선하고 경로/관측소/feature를 코드에 하드코딩하지 않습니다.
- V1/V2 config 흐름을 섞지 않습니다.
- V2는 기본적으로 single-target 실험입니다.
- 데이터/산출물은 Git에 커밋하지 않습니다.
- training/inference/data schema를 바꾸면 README, 관련 config, tests를 함께 갱신합니다.

## 라이선스

현재 저장소 루트에 `LICENSE` 파일이 없습니다. 별도 라이선스가 추가되기 전까지는 이 코드를 자유롭게 재사용·배포할 수 있다고 가정하지 마세요.

## 한 줄 요약

**Gisang은 한국 관측소 기상 데이터를 모델 학습 가능한 시계열 데이터셋으로 바꾸고, 예측·평가·리포트·베타 서비스까지 이어지는 연구용 날씨 예측 파이프라인입니다.**

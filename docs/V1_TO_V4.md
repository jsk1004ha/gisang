# Gisang V1~V4 설명

이 문서는 Gisang 프로젝트의 버전별 연구 흐름을 간단히 정리합니다.

## 한눈에 보기

| 버전 | 핵심 목적 | 대표 특징 | 현재 의미 |
| --- | --- | --- | --- |
| V1 | 기본 파이프라인 구축 | 데이터 준비, 학습, 평가, 추론 CLI | 프로젝트의 초기 골격 |
| V2 | 실험 체계 정리 | unified config, single-target, leaderboard | 온도/습도 baseline 연구의 중심 |
| V3 | NWP-assisted/MOS 연구 | forecast NWP 또는 ERA5 기반 residual correction | 관측소 단위 운영형 예보로 가는 중간 단계 |
| V4 | 운영 예보와 공간 특징 확장 | real forecast archive, quality gate, grid patch, beta service | 운영 검증과 서비스 산출물 방향 |

## V1: 초기 station-level 예측 파이프라인

V1은 프로젝트의 기본 동작을 만든 단계입니다.

### 목표

- ASOS/AWS 관측 데이터를 읽고 내부 표준 schema로 정리합니다.
- ERA5 기반 특징을 관측소 시간축에 붙입니다.
- 시간순 train/validation/test split을 유지합니다.
- 학습, 평가, 추론을 CLI로 실행할 수 있게 합니다.

### 주요 구성

- 데이터 설정: `configs/data/*.yaml`
- 모델 설정: `configs/model/*.yaml`
- 학습 설정: `configs/train/*.yaml`
- 학습 코드: `src/weather_korea_forecast/training/`
- 추론 코드: `src/weather_korea_forecast/inference/`

### 대표 실행 흐름

```bash
python -m weather_korea_forecast.data.build_training_table \
  --config configs/data/dataset_v1.yaml

python -m weather_korea_forecast.training.train \
  --data-config configs/data/dataset_v1.yaml \
  --model-config configs/model/tft_v1.yaml \
  --train-config configs/train/train_v1.yaml

python -m weather_korea_forecast.evaluation.evaluate \
  --experiment-dir data/artifacts/experiments/latest
```

### V1의 의미

V1은 실험 품질을 완성한 단계라기보다, 이후 V2~V4가 공유하는 기본 데이터/학습/평가 구조를 만든 단계입니다.

## V2: unified config 기반 단일 타깃 실험

V2는 V1의 구조를 보존하면서 실험을 더 재현 가능하게 정리한 단계입니다.

### 목표

- `configs/v2/experiments/` 아래 단일 config로 데이터, 모델, 학습, 평가를 제어합니다.
- 온도와 습도를 분리한 single-target 실험을 기본으로 합니다.
- 72시간 또는 168시간 encoder로 24시간 direct forecast를 수행합니다.
- station/region/horizon/season별 평가 artifact와 leaderboard를 생성합니다.

### 주요 실험

- 온도 ridge baseline: `v2_temp_ridge.yaml`
- 온도 TFT/fallback torch 실험: `v2_temp_tft.yaml`
- 온도 horizon-wise ridge/LightGBM 실험
- 습도 ridge/LightGBM 실험
- 습도 dew point, dew-point depression, logit RH 변환 실험
- local real-data bootstrap 예시: `configs/v2/experiments/real/`

### 대표 실행 흐름

```bash
python -m weather_korea_forecast.v2.prepare_data \
  --config configs/v2/experiments/v2_temp_ridge.yaml

python -m weather_korea_forecast.v2.train \
  --config configs/v2/experiments/v2_temp_tft.yaml

python -m weather_korea_forecast.v2.evaluate \
  --experiment-dir data/artifacts/v2_experiments/latest
```

### V2의 의미

V2는 현재 저장소에서 baseline 비교와 실험 리포팅의 기준이 되는 단계입니다. 다만 ERA5 reanalysis 미래값을 decoder feature로 쓰는 실험은 운영 예보가 아니라 backtest-only로 구분해야 합니다.

## V3: NWP-assisted/MOS 관측소 예보

V3는 V2 파이프라인을 바탕으로 실제 예보 입력을 고려한 station-level MOS(Model Output Statistics) 연구를 분리한 단계입니다.

### 목표

- observation-only track과 NWP-assisted/MOS track을 분리합니다.
- ERA5 reanalysis 기반 미래 feature는 backtest-only로 표시합니다.
- 운영 예보에서는 issue-time-aligned forecast NWP CSV를 사용합니다.
- 온도는 NWP baseline의 residual을 학습하는 방식으로 접근합니다.
- 습도는 direct RH, dew point, dew-point depression 등 별도 target strategy를 비교합니다.

### 대표 구성

- V3 config: `configs/v3/experiments/`
- V3 humidity NWP-MOS runner: `src/weather_korea_forecast/v3/nwp_mos.py`
- artifact-level ensemble: `weather_korea_forecast.v2.ensemble`
- dashboard/reporting: `weather_korea_forecast.dashboard.app`, `weather_korea_forecast.reporting.generate_report`

### 대표 실험

- `v3_temp_mos_residual_ridge_72to24.yaml`
- `v3_temp_mos_residual_lgbm_72to24.yaml`
- `v3_temp_mos_ensemble_72to24.yaml`
- `v3_humidity_direct_lgbm_72to24.yaml`
- `v3_humidity_dewpoint_lgbm_72to24.yaml`
- `v3_humidity_nwp_mos_lgbm_72to24.yaml`

### V3의 의미

V3는 “학습용으로만 좋은 모델”과 “운영 시간에 실제로 쓸 수 있는 모델”을 분리하려는 단계입니다. `operational_valid`, `backtest_only`, `future_feature_source` 같은 metadata가 중요합니다.

## V4: 운영 예보 archive와 공간/확률 확장

V4는 V3의 station-level MOS를 운영 검증과 공간 특징으로 확장하는 단계입니다.

### 목표

- 실제 forecast archive를 사용합니다.
- archive quality gate를 통과하지 못하면 운영 학습을 막습니다.
- GFS/KMA/ECMWF 등 forecast source의 `issue_time`, `valid_time`, `horizon_step`을 명확히 분리합니다.
- station-nearest feature를 넘어 grid patch feature를 추가합니다.
- 온도/습도/날씨상태 결과를 서비스 schema로 내보냅니다.

### 주요 구성

- V4 config: `configs/v4/experiments/`
- forecast archive 품질 gate: `docs/FORECAST_ARCHIVE_QUALITY_GATE.md`
- NWP archive 획득/정규화: `docs/NWP_ARCHIVE_ACQUISITION.md`
- operational runner: `src/weather_korea_forecast/v4/`
- forecast/service schema: `src/weather_korea_forecast/service/`, `src/weather_korea_forecast/api/`

### G024~G031 운영 연구 흐름

- G024: real forecast NWP + real ASOS 기반 operational performance runner
- G025: 90-cycle medium archive와 true GFS grid patch evidence
- G026: 182-cycle strong archive와 target-specific production model manifest
- G027: 사람이 읽기 쉬운 operational dashboard report
- G028: temperature final improvement attempt, PASS gate 미달
- G029: humidity final improvement, NEAR_PASS beta
- G030: production freeze, 모델 선택 고정
- G031: site/API/schema-consumption only, 모델 개선 재개 금지

### V4의 의미

V4는 연구 결과를 운영 조건에 더 가깝게 검증하고, 베타 서비스가 사용할 수 있는 forecast artifact와 schema를 만드는 단계입니다. 단, G030 이후 사이트 handoff용 모델 선택은 freeze되어 있으며, 부족한 품질은 재학습으로 숨기지 않고 caveat/beta로 표시합니다.

## 버전 간 관계

```text
V1: 기본 CLI와 artifact 구조
  -> V2: unified config와 single-target 실험 체계
    -> V3: NWP-assisted/MOS와 운영-valid metadata 분리
      -> V4: real forecast archive, quality gate, patch, service schema
```

V1~V4는 완전히 분리된 제품 버전이라기보다, 같은 연구 목표를 더 엄격한 운영 조건으로 밀어 올린 단계적 연구 흐름입니다.

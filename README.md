# Gisang Weather Korea Forecast

**Gisang**은 한국 관측소의 기상 관측값과 예보/재분석 데이터를 이용해 관측소 단위 날씨 예측을 연구하는 프로젝트입니다.

이 저장소는 단순히 모델 하나를 학습하는 코드가 아니라, 데이터를 모으고 정리한 뒤 예측 모델을 만들고, 실제 관측값으로 평가하고, 결과를 리포트와 베타 웹/API 형태로 확인하는 전체 연구 흐름을 담고 있습니다.

## 프로젝트가 하는 일

Gisang의 기본 흐름은 아래와 같습니다.

```text
ASOS/AWS 관측 데이터
+ ERA5, GFS, KMA 등 예보/재분석 특징
-> 관측소 기준 학습 테이블 생성
-> 모델 학습 및 baseline 비교
-> 1~24시간 예측
-> 실제 관측값과 비교 평가
-> 실험 리포트와 베타 서비스 산출물 생성
```

## 왜 만들었나

한국 지역의 날씨 예측을 연구하려면 다음 문제가 반복됩니다.

- 관측 데이터와 예보 데이터를 같은 시간축으로 맞춰야 합니다.
- 미래 예보에 사용할 수 없는 값을 학습에 섞지 않아야 합니다.
- 관측소, 지역, 예보 horizon별로 오차를 따로 봐야 합니다.
- 온도와 습도는 서로 다른 모델링 전략과 평가 기준이 필요합니다.
- 연구 결과를 사람이 읽을 수 있는 리포트와 서비스용 schema로 정리해야 합니다.

Gisang은 이 과정을 재현 가능한 파이프라인으로 만들기 위한 연구 코드베이스입니다.

## 현재 프로젝트 상태

현재 저장소에는 V1부터 V4까지의 실험 흐름이 공존합니다.

- **V1**: 기본 데이터 준비, 학습, 평가, 추론 CLI를 갖춘 초기 파이프라인
- **V2**: unified config 기반의 단일 타깃 온도/습도 실험 체계
- **V3**: NWP-assisted/MOS 방식의 관측소 단위 운영형 예보 실험
- **V4**: real forecast archive, grid patch, 운영 품질 gate, 베타 서비스 산출물 방향

상세 설명은 [V1~V4 설명](docs/V1_TO_V4.md)을 보세요.

## 주요 구성 요소

```text
configs/                         데이터, 모델, 실험 설정
src/weather_korea_forecast/       파이프라인 소스 코드
  data/                           관측/NWP/ERA5 데이터 로딩과 학습 테이블 생성
  models/                         baseline, ridge, LightGBM, fallback torch, TFT wrapper
  training/                       V1 학습 루프
  evaluation/                     metric, plot, 평가 리포트
  inference/                      추론 CLI와 forecast schema
  reporting/                      통합 실험 리포트 생성
  service/                        운영 artifact, freeze guard, service helper
  api/                            Forecast API helper
  v2/                             V2 unified config 파이프라인
  v3/                             V3 NWP-MOS runner
  v4/                             V4 forecast archive, patch, operational runner
web/                              정적 베타 웹 화면
docs/                             설계, 상태, 연구 기록, 운영 문서
tests/                            회귀/스모크 테스트
```

## 데이터 원칙

- 관측 데이터는 내부적으로 아래 표준 schema로 정규화합니다.

```text
station_id, datetime, temp, humidity, pressure, wind_speed, precipitation, quality_flag
```

- 시간 처리는 UTC 기준으로 명시합니다.
- train/validation/test split은 시간 순서를 지킵니다.
- ASOS를 기본 관측 소스로 보고, AWS 등 보조 데이터는 config 기반으로 결측 보완 또는 일부 feature 우선순위에 사용합니다.
- 운영 예보 실험에서는 `issue_time <= forecast_init_time` 조건을 만족하는 실제 예보 archive를 사용해야 합니다.
- ERA5 reanalysis 미래값을 쓰는 실험은 운영 예보가 아니라 backtest/diagnostic으로 구분합니다.

## 모델링 원칙

- `fallback_torch`는 선택 의존성이 없을 때 쓰는 경량 대체 모델입니다. true TFT가 아닙니다.
- true TFT 경로는 `pytorch_forecasting` 등 선택 의존성이 설치된 경우에만 사용합니다.
- V2 이후 기본 실험은 온도와 습도를 분리한 single-target 흐름입니다.
- 실험 결과는 baseline과 비교하고, horizon/station/region/season별로 해석합니다.
- 운영 또는 사이트 표시용 결과에는 caveat, beta, source, confidence를 명확히 노출합니다.

## 연구 기록

프로젝트의 연구 진행 과정은 별도 문서로 정리되어 있습니다.

- [V1~V4 설명](docs/V1_TO_V4.md)
- [연구 타임라인](docs/RESEARCH_TIMELINE.md)
- [V2 plan](docs/V2_plan.md)
- [V2 status](docs/V2_STATUS.md)
- [V3/V4 plan](docs/V3_V4_plan.md)
- [V3 real-data results](docs/V3_REALDATA_RESULTS.md)
- [V4 implementation plan](docs/V4_IMPLEMENTATION_PLAN.md)
- [G030 production freeze](docs/G030_PRODUCTION_FREEZE.md)
- [G031 site finalization](docs/SITE_FINALIZATION.md)

## 현재 서비스 표면

G030에서 사이트 handoff용 모델 선택이 freeze되었고, G031은 모델 개선이 아니라 API와 웹 화면이 frozen schema를 소비하는 단계입니다.

- Forecast schema: `forecast_points.v2-beta-sources`
- 온도: 정확도 caveat 포함
- 습도: beta 표시 필요
- 날씨 상태: rule-based beta 표시 필요
- 웹/API는 `source`, `status`, `confidence`, `caveat` 정보를 숨기지 않는 방향입니다.

## 실행과 세부 사용법

README는 프로젝트 설명만 담고, 실행 명령과 세부 사용법은 목적별 문서로 분리합니다.

- 버전별 실행 흐름: [V1~V4 설명](docs/V1_TO_V4.md)
- 운영 예보 archive와 품질 gate: `docs/FORECAST_ARCHIVE_QUALITY_GATE.md`, `docs/NWP_ARCHIVE_ACQUISITION.md`
- API와 웹 schema: `docs/FORECAST_API.md`, `docs/FORECAST_OUTPUT_SCHEMA.md`

## 라이선스

현재 저장소 루트에 `LICENSE` 파일이 없습니다. 별도 라이선스가 추가되기 전까지는 자유로운 재사용·배포 권한을 가정하지 마세요.

## 한 줄 요약

**Gisang은 한국 관측소 날씨 데이터를 예측 가능한 시계열 데이터셋으로 만들고, 모델 학습·평가·리포트·베타 서비스까지 연결하는 날씨 예측 연구 파이프라인입니다.**

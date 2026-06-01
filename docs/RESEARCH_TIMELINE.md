# Gisang 연구 타임라인

이 문서는 저장소의 주요 연구 흐름을 시간순으로 요약합니다. 세부 변경 내역은 `CHANGELOG.md`와 각 G0xx 문서를 기준으로 확인하세요.

## 요약

```text
V1 기본 파이프라인
-> V2 unified config와 baseline 체계
-> V2 온도/습도 개선 실험
-> V3 NWP-assisted/MOS 분리
-> V4 real forecast archive와 운영 gate
-> G030 production freeze
-> G031 site/API handoff
```

## 2026-03-26: V2 파이프라인 도입

- `src/weather_korea_forecast/v2/` 아래 V2 단일 타깃 실험 파이프라인을 추가했습니다.
- direct multi-horizon dataset 흐름을 만들었습니다.
- station metadata와 geographic feature를 붙였습니다.
- persistence, seasonal persistence, ridge, optional LightGBM baseline을 정리했습니다.
- V2 artifact, leaderboard, alias 관리가 추가되었습니다.
- local real-data bootstrap config가 추가되었습니다.

## 2026-05-10: V2 온도 baseline 강화

- horizon-wise ridge와 horizon-wise LightGBM 실험 표면을 추가했습니다.
- 168h encoder, residual LightGBM-on-ridge, station-wise/region-wise scaling config를 추가했습니다.
- horizon/station/region/daily max-min artifact를 확장했습니다.
- V2 온도 ridge를 공식 72->24 baseline으로 문서화했습니다.

## 2026-05-11: V2 습도 연구 확장

- 습도 LightGBM 실험 config를 확장했습니다.
- logit RH, dew point, dew-point depression target transform을 추가했습니다.
- dry/humid event, low/high RH MAE 등 습도 진단 metric을 추가했습니다.
- 습도 feature engineering에 daytime, vapor pressure, absolute humidity, ERA5 wind feature 등을 추가했습니다.

## 2026-05-12: NWP-assisted metadata와 운영 추론 guard

- V2/V3 흐름에 `future_feature_metadata.json`와 per-track leaderboard metadata를 추가했습니다.
- `residual_from_feature` target transform으로 `observed_temp - future_era5_t2m_c` MOS 실험을 가능하게 했습니다.
- `--future-weather-csv` 기반 운영 NWP-assisted inference를 추가했습니다.
- 미래 weather covariate가 부족한 운영 추론은 실패하도록 guard를 강화했습니다.

## 2026-05-14: V3/V4 계획과 V3 MOS 표면 분리

- `docs/V3_V4_plan.md`로 V3 station-level MOS와 V4 national/spatial/probabilistic 방향을 분리했습니다.
- V3 temperature MOS residual ridge config를 추가했습니다.
- V3 humidity direct/dew-point/dew-point-depression starter config를 추가했습니다.
- issue-time-aligned V3 humidity NWP-MOS runner를 추가했습니다.
- diagnostic oracle config를 operational claim에서 제외하도록 문서화했습니다.
- artifact-level ensemble CLI와 local dashboard를 추가했습니다.

## 2026-05-25: V3 real-data run과 unified reporting 정리

- V3 temperature MOS, humidity, all-core runner script를 추가했습니다.
- unified reporting package를 추가해 CSV/JSON/HTML 리포트를 생성하게 했습니다.
- stale cached V2/V3 training table을 자동 재생성하도록 개선했습니다.
- V3 real-data result 문서에 현재 best non-oracle temperature MOS evidence를 기록했습니다.
- 현재 기록상 `v3_temp_mos_residual_ridge_72to24`는 RMSE 약 `1.064°C`로 1.0°C 목표에는 조금 못 미칩니다.

## 2026-05-28: G022~G024 운영 archive/gate 흐름

- G022 operational training gate를 추가해 forecast archive adequacy와 SHA-256 binding을 요구했습니다.
- G023 real NWP forecast archive acquisition adapter를 추가했습니다.
- G024 operational performance runner를 추가했습니다.
- real forecast NWP와 real ASOS를 결합해 raw GFS와 residual LightGBM baseline을 비교했습니다.
- forecast archive quality report와 NWP archive status가 HTML report에 통합되었습니다.

## G025: operational accuracy sprint

- 30-cycle short archive에서 90-cycle medium archive로 확장했습니다.
- true GFS GRIB-grid 3x3/5x5 patch summary 추출을 추가했습니다.
- operational runner에 grid patch feature, optional CatBoost, ensemble artifact를 추가했습니다.
- medium benchmark에서 temperature best는 약 `1.683°C`, humidity best는 약 `10.406%p`로 기록되었습니다.
- V4-C gate는 temperature/humidity 기준 미달로 FAIL, site readiness는 WARN으로 해석되었습니다.

## G026: accuracy breakthrough sprint

- 182-cycle strong archive 기반 운영 benchmark를 수행했습니다.
- full-variable GFS schema와 variable coverage report를 추가했습니다.
- target-specific production model manifest를 추가했습니다.
- strong run evidence에서 temperature best는 `1.694°C`, humidity best는 `10.396%p`로 기록되었습니다.
- 두 target 모두 PASS gate에는 미달했으므로 V4-C와 site operational beta는 계속 차단되었습니다.

## G027: operational report dashboard

- `operational_performance_report.html`을 사람이 읽기 쉬운 dashboard 형태로 재구성했습니다.
- KPI card, PASS/WARN/FAIL badge, gate/readiness card, target별 분석 section을 추가했습니다.
- forecast-vs-actual, scatter/residual, horizon, station/region, heatmap, patch ablation, calibration, model comparison plot을 생성했습니다.

## G028: temperature final improvement

- issue-time/horizon calibration candidate를 추가했습니다.
- strong archive에서 bounded temperature improvement allowlist를 실행했습니다.
- best temperature RMSE가 `1.694°C`로 남아 PASS 기준 `1.5°C`와 NEAR_PASS 기준 `1.6°C`를 넘었습니다.
- 결과는 FAIL로 기록되었고, G030에서는 temperature accuracy caveat를 표시하는 방향으로 결정되었습니다.

## G029: humidity final improvement

- no-patch LGBM calibration, logit-RH residual LightGBM, quantile/isotonic calibration 후보를 비교했습니다.
- validation holdout 기준으로 no-patch residual LGBM을 official humidity baseline으로 유지했습니다.
- humidity final RMSE는 `10.396%p`, bias는 `0.596%p`로 NEAR_PASS/beta로 기록되었습니다.

## G030: production freeze

- 사이트 handoff용 모델 선택을 고정했습니다.
- temperature: `ensemble_stationwise_inverse_rmse`, accuracy caveat 포함
- humidity: no-patch `operational_residual_lgbm_humidity`, beta 포함
- weather code: `rule_based_beta`
- freeze artifact와 manifest checksum, schema version `forecast_points.v2-beta-sources`를 기록했습니다.
- 이후 단계에서는 새 모델 후보를 추가하지 않는 규칙을 세웠습니다.

## G031: website finalization

- G030 freeze manifest와 frozen schema를 소비하는 site/API 작업만 수행했습니다.
- `/api/forecast/*`, `/api/model/status`와 정적 `web/` UI가 caveat, beta, reliability, confidence를 노출하도록 정리했습니다.
- 모델 품질 부족은 재학습으로 숨기지 않고 temperature caveat, humidity beta, rule-based weather-code beta로 표시합니다.

## 앞으로의 연구 방향

- 운영 archive를 더 긴 기간과 다양한 계절로 확장합니다.
- 실제 forecast source의 issue-time alignment를 유지합니다.
- temperature RMSE를 1.5°C 이하로 낮출 수 있는지 검증합니다.
- humidity RMSE를 10%p 이하로 안정화합니다.
- grid patch, spatial encoder, probabilistic forecast를 V4 후속 연구로 검토합니다.
- site/API는 frozen model selection과 caveat 표시 원칙을 유지합니다.

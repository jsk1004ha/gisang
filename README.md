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

`fallback_torch`는 실제 TFT가 아니라 경량 대체 모델이다. 이 경로에서는 `residual_baseline`을 켜면 마지막 encoder 관측값을 기본 예측선으로 두고 신경망이 horizon별 보정량을 학습한다. V1 기본 설정은 `target_source_features: [obs_temp]`를 사용한다.

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
- `ASOS + local AWS CSV`: [configs/data/dataset_asos_aws_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\data\dataset_asos_aws_v1.yaml)처럼 `paths.aws_observation_csv`, `aws.resample_rule`, `aws.prefer_columns`를 지정해 분단위 CSV를 시간 단위로 집계 후 컬럼별 우선순위 병합

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
- [configs/model/ridge_v1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\model\ridge_v1.yaml): ridge regression baseline. `alpha_grid`를 설정하면 기본적으로 validation loss가 가장 낮은 ridge alpha를 closed-form 후보 중에서 선택한다. V2처럼 보정 후 성능을 기준으로 고르고 싶으면 `model.alpha_selection.metric: bias_corrected_holdout_mse`를 사용할 수 있다.

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

V2 `evaluation.bias_correction`은 validation 예측을 시간 순서대로 calibration/holdout 구간으로 나누는 guard를 지원한다.

```yaml
evaluation:
  bias_correction:
    enabled: true
    mode: per_station_horizon
    method: mean_bias
    calibration_fraction: 0.7
    apply_when: improves_on_holdout
    selection_metric: rmse
```

이 설정에서는 calibration 구간으로 관측소·horizon별 평균 bias를 계산하고 holdout 구간에서 RMSE가 개선될 때만 보정을 적용한다. 채택되면 전체 validation으로 bias를 다시 산출하며, `bias_correction.json`에 raw/corrected holdout metric과 채택 여부가 저장된다. 더 강한 보정이 필요하면 `method: affine`으로 horizon별 `slope * prediction + intercept`를 쓰거나, `mode: auto`, `method: auto`와 `candidate_modes`/`candidate_methods`를 지정해 holdout에서 후보를 비교할 수 있다.

V2 ridge 계열은 raw validation MSE 대신 보정 후 holdout MSE로 `alpha_grid`를 선택할 수도 있다. 현재 `v2_temp_ridge.yaml`은 V2 temperature의 공식 72→24 기준선이며, 최신 로컬 benchmark에서는 보정 후 RMSE `2.511`, MAE `1.844`, Bias `-0.224` 수준을 기록했다. 이후 temperature 실험은 “TFT가 좋아졌나?”보다 “이 ridge 기준선을 이겼나?”를 우선 판단 기준으로 삼는다.

```yaml
model:
  type: ridge
  alpha: 1.0
  alpha_grid: [0.01, 0.1, 1.0, 10.0, 100.0]
  alpha_selection:
    metric: bias_corrected_holdout_mse
    calibration_fraction: 0.7
    correction_mode: per_horizon
    apply_when_improves: true
```

이 선택기는 validation 예측의 앞부분으로 평균 bias를 추정하고 뒤쪽 holdout에서 raw/보정 중 더 나은 MSE를 후보 alpha별로 비교한다. 보정 후 평가 흐름과 모델 선택 기준을 맞춰 과소 정규화된 ridge 후보가 선택되는 문제를 줄인다.

후반 horizon bias와 극값 압축을 줄이기 위한 V2 추가 모델 타입도 지원한다.

- `horizon_wise_ridge`: horizon 1~24 각각 별도 ridge head와 alpha를 학습하고 `horizon_model_metrics.csv`에 horizon별 alpha/RMSE/MAE/Bias를 저장한다.
- `horizon_wise_lightgbm`: horizon별 direct LightGBM estimator를 명시적으로 쓰고 feature importance와 horizon별 validation metric을 저장한다.
- `residual`: baseline 예측을 먼저 학습한 뒤 `actual - baseline` residual을 별도 모델이 학습하고 `baseline + residual`을 최종 예측으로 쓴다.
- `decoder_feature_baseline`: `model.target_source_features`에 지정한 future-known decoder feature를 target으로 그대로 복사한다. forecast-model baseline이나 명시적 oracle/backtest ceiling 점검용이며, target leakage feature를 넣은 config는 운영 가능 실험으로 해석하면 안 된다.

스케일링은 `data.scaling.mode`로 `global`, `stationwise`/`station_wise`, `regionwise`/`region_wise`, `none`을 선택할 수 있다. `scaler.json`에는 train split에서 fit한 global 및 group별 평균/표준편차가 저장된다.

### V2 추론

```bash
python -m weather_korea_forecast.v2.predict ^
  --experiment-dir data/artifacts/v2_experiments/latest ^
  --station-id 108 ^
  --forecast-init-time 2025-01-03T00:00:00Z
```

NWP-assisted/MOS 실전 추론은 future-valid forecast feature CSV를 함께 넘긴다. CSV는 `station_id`, `valid_time` 또는 `datetime`, 선택적 `issue_time`, 그리고 모델의 decoder weather feature에 대응되는 forecast columns를 가져야 한다. 예시는 [future_weather_forecast_template.csv](C:\Users\js100\Desktop\coding\gisang\configs\v2\templates\future_weather_forecast_template.csv)에 있다.

```bash
python -m weather_korea_forecast.v2.predict ^
  --experiment-dir data/artifacts/v2_experiments/latest ^
  --station-id 108 ^
  --forecast-init-time 2025-01-03T00:00:00Z ^
  --future-weather-csv data/raw/nwp/latest_station_forecast.csv ^
  --output-csv data/artifacts/v2_experiments/latest/forecast_operational.csv
```


### G023 real NWP forecast archive workflow

Operational-valid G022 training requires a real prepared forecast archive, not ERA5 reanalysis, smoke, synthetic, generated, or fixture rows. Build source-specific prepared files first, then merge and gate them:

```bash
python -m weather_korea_forecast.data.kma_forecast_archive \
  --input data/raw/nwp/kma_forecast_raw.csv \
  --station-metadata data/raw/metadata/stations.csv \
  --output data/raw/nwp/archive/kma_prepared_forecast.csv

python -m weather_korea_forecast.data.gfs_forecast_archive \
  --input data/raw/nwp/gfs_forecast_raw.csv \
  --station-metadata data/raw/metadata/stations.csv \
  --output data/raw/nwp/archive/gfs_prepared_forecast.csv

python -m weather_korea_forecast.data.nwp_archive \
  --inputs data/raw/nwp/archive/kma_prepared_forecast.csv data/raw/nwp/archive/gfs_prepared_forecast.csv \
  --output data/raw/nwp/archive/prepared_forecast_archive.csv \
  --quality-report data/raw/nwp/archive/archive_quality_report.json \
  --expected-columns nwp_t2m,nwp_sp,nwp_u10,nwp_v10,nwp_tp,nwp_dew_point
```

Adequacy defaults are `station_count >= 20`, `forecast_cycle_count >= 30`, horizon 1–24 coverage `>= 0.95`, missing rate `<= 0.05`, train/val/test split possible, and no synthetic/smoke/generated/fixture provenance. The quality report also checks humidity/dew-point units/ranges, precipitation probability, sky code, precipitation type, per-station cycle/horizon coverage, and configured expected forecast columns. It writes `archive_content_sha256`; G022 training rejects stale or mismatched quality reports.

Run operational G022 training only through the gated command:

```bash
python -m weather_korea_forecast.v2.train \
  --config configs/g022/experiments/g022_temp_operational_residual_ridge_72to24.yaml \
  --future-weather-csv data/raw/nwp/archive/prepared_forecast_archive.csv \
  --archive-quality-report data/raw/nwp/archive/archive_quality_report.json
```

If `forecast_archive_adequate=false`, training is blocked and the quality report's `blocking_reasons` explain which data-collection target is still short. See `docs/NWP_ARCHIVE_ACQUISITION.md`, `docs/KMA_FORECAST_ADAPTER.md`, `docs/GFS_FORECAST_ADAPTER.md`, and `docs/FORECAST_ARCHIVE_QUALITY_GATE.md`.

### G024 operational performance sprint

After the G023 archive gate passes, run the operational-valid LightGBM MOS sprint against real forecast and real ASOS observations. The runner always writes a visual HTML summary alongside JSON/CSV artifacts:

```bash
PYTHONPATH=src .venv312/Scripts/python.exe -m weather_korea_forecast.v4.operational_performance \
  --nwp-archive data/raw/nwp/archive/prepared_forecast_archive.csv \
  --archive-quality-report data/raw/nwp/archive/archive_quality_report.json \
  --observations data/raw/asos/asos_hourly.csv \
  --station-metadata data/raw/metadata/stations.csv \
  --output-dir data/artifacts/g024_operational_performance
```

Official operational baselines are `raw_gfs_t2m` and `operational_residual_lgbm_temp` for temperature, plus `raw_gfs_rh` and `operational_residual_lgbm_humidity` for humidity. Ridge residual models are reported under debug until they beat raw GFS on operational data. The report includes calibration selection, LGBM grid results, patch ablation, ridge residual sign/formula diagnostics, V4-C gate status, site-readiness status, and benchmark reliability (`smoke`/`short`/`medium`/`strong`/`seasonal`). See `docs/G024_OPERATIONAL_PERFORMANCE.md`.

### G025 operational accuracy sprint

G025 extends the operational benchmark to a 90-cycle medium archive and adds true GFS GRIB-grid patch features. Use `weather_korea_forecast.v4.gfs_grid_patch` to build ignored patch feature CSVs, then pass them to the operational runner with `--grid-patch-features`. The runner labels `patch_feature_mode` as `station_neighborhood_proxy` or `true_gfs_grid_patch`, uses all cycles for medium-or-larger archives with a time-ordered split, writes ensemble artifacts, and always emits `operational_performance_report.html`. See `docs/G025_OPERATIONAL_ACCURACY.md`.

`data.future_features.column_mapping`은 forecast CSV 컬럼을 학습 feature 이름으로 매핑한다. 예: `era5_t2m: gfs_t2m`, `era5_sp: gfs_sp`, `era5_u10: gfs_u10`, `era5_v10: gfs_v10`, `era5_tp: gfs_tp`. `issue_time`이 있으면 `forecast_init_time` 이하의 최신 run을 선택한다. 운영 추론에서는 미래 weather covariate가 모든 horizon에 없으면 실행을 중단한다.

### V2 기본 실험 config

- [v2_temp_seasonal_persistence.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_seasonal_persistence.yaml)
- [v2_temp_ridge.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_ridge.yaml)
- [v2_temp_ridge_168to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_ridge_168to24.yaml)
- [v2_temp_horizonwise_ridge_72to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_horizonwise_ridge_72to24.yaml)
- [v2_temp_horizonwise_ridge_168to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_horizonwise_ridge_168to24.yaml)
- [v2_temp_horizonwise_lgbm_168to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_horizonwise_lgbm_168to24.yaml)
- [v2_temp_residual_lgbm_on_ridge_168to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_residual_lgbm_on_ridge_168to24.yaml)
- [v2_temp_future_era5_ridge_72to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_future_era5_ridge_72to24.yaml)
- [v2_temp_future_era5_residual_ridge_72to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_future_era5_residual_ridge_72to24.yaml)
- [v2_temp_future_era5_residual_ridge_168to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_future_era5_residual_ridge_168to24.yaml)
- [v2_temp_future_era5_residual_horizonwise_ridge_72to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_future_era5_residual_horizonwise_ridge_72to24.yaml)
- [v2_temp_ridge_168to24_stationwise.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_ridge_168to24_stationwise.yaml)
- [v2_temp_ridge_168to24_regionwise.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_ridge_168to24_regionwise.yaml)
- [v2_temp_lgbm.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_lgbm.yaml)
- [v2_temp_tft.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_tft.yaml)
- [v2_temp_tft_168h.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_temp_tft_168h.yaml)
- [v2_humidity_ridge.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_ridge.yaml)
- [v2_humidity_lgbm.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_lgbm.yaml)
- [v2_humidity_tft.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_tft.yaml)
- [v2_humidity_lgbm_fixed_features_72to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_lgbm_fixed_features_72to24.yaml)
- [v2_humidity_lgbm_168to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_lgbm_168to24.yaml)
- [v2_humidity_horizonwise_lgbm_72to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_horizonwise_lgbm_72to24.yaml)
- [v2_humidity_horizonwise_lgbm_168to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_horizonwise_lgbm_168to24.yaml)
- [v2_humidity_logit_rh_lgbm_168to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_logit_rh_lgbm_168to24.yaml)
- [v2_humidity_dewpoint_lgbm_168to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_dewpoint_lgbm_168to24.yaml)
- [v2_humidity_dewpoint_depression_lgbm_168to24.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_dewpoint_depression_lgbm_168to24.yaml)
- [v2_humidity_lgbm_extreme_weighted.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_lgbm_extreme_weighted.yaml)
- [v2_humidity_lgbm_quantile_calibrated.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\v2_humidity_lgbm_quantile_calibrated.yaml)

습도 연구용 config는 `data.target_transform`으로 `logit_rh`, `dew_point`, `dew_point_depression`을 선택할 수 있다.
`logit_rh`는 예측 후 RH(0~100)로 역변환하고, dew-point 계열은 `_target_context_temp_c` 온도 context로 RH를 복원한다.
`data.sample_weighting.mode: humidity_extremes`는 `RH < 40`, `RH > 80` 같은 건조/고습 구간에 LightGBM sample weight를 줄 때 사용한다.

미래 ERA5/NWP 계열 decoder covariate를 쓰는 온도 config는 observation-only 리더보드와 직접 비교하지 않는다. `data.future_features`로 `track`, `source`, `operational_valid`를 명시하고, artifact에는 `future_feature_metadata.json`이 저장된다. ERA5 reanalysis를 `decoder_known`에 넣은 config는 backtest/MOS upper-bound이며 실전 예보에서는 같은 변수 구조의 forecast NWP source로 교체해야 한다. `data.target_transform.type: residual_from_feature`는 예를 들어 `baseline_column: era5_t2m_c`를 기준으로 `target_value = observed_temp - era5_t2m_c`를 학습하고 예측 후 baseline을 다시 더해 절대기온으로 복원한다.

`station metadata` 템플릿은 [station_metadata_template.csv](C:\Users\js100\Desktop\coding\gisang\configs\v2\templates\station_metadata_template.csv)에 포함되어 있다.

### V2 로컬 실데이터 bootstrap config

다관측소 benchmark용 원본이 아직 준비되지 않았더라도, 저장소에 있는 서울 실데이터로 V2 전체 경로를 실제로 점검할 수 있게 `real/` 예시 config를 포함한다.

- [v2_humidity_ridge_seoul_q4q1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\real\v2_humidity_ridge_seoul_q4q1.yaml)
- [v2_humidity_lgbm_seoul_q4q1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\real\v2_humidity_lgbm_seoul_q4q1.yaml)
- [v2_humidity_tft_seoul_q4q1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\real\v2_humidity_tft_seoul_q4q1.yaml)
- [v2_humidity_lgbm_fixed_features_seoul_q4q1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\real\v2_humidity_lgbm_fixed_features_seoul_q4q1.yaml)
- [v2_humidity_era5_dewpoint_oracle_decoder_feature_seoul_q4q1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\v2\experiments\real\v2_humidity_era5_dewpoint_oracle_decoder_feature_seoul_q4q1.yaml): 로컬 backtest-only oracle 예시다. 현재 서울 raw ERA5 파일에는 dew point가 없어 전처리된 `era5_dew_point_c`가 관측 습도에서 역산되므로, 이 config는 RMSE ceiling/누수 진단용이며 canonical benchmark나 운영 예보용이 아니다.

이 예시는 `data/raw/asos/seoul_20241001_20250324.csv`, `data/raw/era5/seoul_20241001_20250324_station.csv`, `data/raw/metadata/stations.csv`를 사용한다.

## V3 방향: station-level NWP-assisted MOS

V3는 V2 코드를 재사용하되 실험 의미를 운영 구조에 맞게 분리하는 단계다. 자세한 개발 계획은 [docs/V3_V4_plan.md](C:\Users\js100\Desktop\coding\gisang\docs\V3_V4_plan.md)에 정리했다.

첫 V3 config는 temperature MOS residual ridge다.

```bash
python -m weather_korea_forecast.v2.train ^
  --config configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml
```

이 config는 `version: v3`와 `artifacts.root_dir: data/artifacts/v3_experiments`를 사용하지만, 안정화 전까지는 기존 V2 CLI로 실행한다. 핵심 target은 다음과 같다.

```text
baseline = future NWP/ERA5-style t2m
target = observed_temp - baseline
prediction = baseline + predicted_residual
```

V3 leaderboard에는 최소한 `forecast_track`, `uses_future_weather_features`, `future_feature_source`, `operational_valid`, `backtest_only`, `target_name`, `model_family`가 남아야 한다. 또한 umbrella 파일로 `leaderboard_observation_only.csv`, `leaderboard_nwp_assisted.csv`, target별 `leaderboard_temp.csv` / `leaderboard_humidity.csv`를 유지한다.

ERA5 reanalysis decoder covariate를 쓰는 V3 starter config는 `forecast_track: nwp_assisted_mos`, `future_feature_source: era5_reanalysis`, `operational_valid: false`, `backtest_only: true`로 기록된다. 운영 예보에서는 같은 schema의 forecast NWP CSV를 `--future-weather-csv`로 넣어야 한다.

정직한 습도 `RMSE < 1`을 시도하려면 valid-time reanalysis가 아니라 예보 발행시각이 있는 forecast archive가 필요하다. V3 NWP-MOS 경로는 `station_id`, `issue_time`, `valid_time`, `lead_hour`, `nwp_relative_humidity_2m` 등을 가진 CSV를 관측과 조인해 `actual - nwp_relative_humidity_2m` residual을 학습한다.

```bash
# 선택: NOAA GFS GRIB 추출에는 optional nwp extra가 필요하다.
pip install -e ".[nwp,lgbm]"

python -m weather_korea_forecast.data.gfs_surface_forecast ^
  --station-metadata-csv data/raw/metadata/stations_v2.csv ^
  --output-csv data/raw/nwp/gfs_humidity_surface_20241001_20250228.csv ^
  --start-date 2024-10-01 ^
  --end-date 2025-02-28 ^
  --cycles 00,06,12,18 ^
  --lead-hours 1-24 ^
  --variables rh2m,t2m,d2m,sp,u10,v10,tp

python -m weather_korea_forecast.v3.nwp_mos ^
  --config configs/v3/experiments/v3_humidity_nwp_mos_lgbm_72to24.yaml
```

이 경로는 forecast archive의 `issue_time`과 `lead_hour`를 샘플 키로 사용하므로, 나중에 발행된 예보나 미래 관측 습도를 학습 feature로 쓰지 않는다. 단, 실제 `RMSE < 1` 달성 여부는 제공된 NWP 습도 예보 자체의 정확도에 좌우된다.

RMSE ceiling 또는 누수 진단이 필요할 때는 `configs/v3/experiments/diagnostic/v3_temp_observed_oracle_decoder_feature_72to24.yaml`과 `configs/v3/experiments/diagnostic/v3_humidity_observed_oracle_decoder_feature_72to24.yaml`을 사용할 수 있다. 이 config들은 decoder의 미래 `target_value`를 그대로 복사하므로 실제 미래 관측값을 사용한 backtest-only oracle이며, `forecast_track: observed_target_oracle`, `operational_valid: false`, `backtest_only: true`로 기록된다. 운영 예보 성능으로 해석하면 안 된다.

V3 습도 track은 RH 직접 예측과 별개로 물리 target을 비교한다.

```bash
python -m weather_korea_forecast.v2.train ^
  --config configs/v3/experiments/v3_humidity_direct_lgbm_72to24.yaml

python -m weather_korea_forecast.v2.train ^
  --config configs/v3/experiments/v3_humidity_dewpoint_lgbm_72to24.yaml

python -m weather_korea_forecast.v2.train ^
  --config configs/v3/experiments/v3_humidity_dewpoint_depression_lgbm_72to24.yaml
```

세 config 모두 `version: v3`, `artifacts.root_dir: data/artifacts/v3_experiments`를 사용한다. Direct RH는 바로 `0..100`으로 clip하고, dew point/depression track은 temperature context로 RH를 복원한 뒤 `0..100`으로 clip한다.

여러 습도 track을 결합해 더 강한 예측 artifact를 만들 때는 prediction ensemble CLI를 사용한다.

```bash
python -m weather_korea_forecast.v2.ensemble ^
  --experiment-dir data/artifacts/v3_experiments/v3_humidity_direct_lgbm_72to24_YYYYMMDDTHHMMSSZ ^
  --experiment-dir data/artifacts/v3_experiments/v3_humidity_dewpoint_lgbm_72to24_YYYYMMDDTHHMMSSZ ^
  --experiment-dir data/artifacts/v3_experiments/v3_humidity_dewpoint_depression_lgbm_72to24_YYYYMMDDTHHMMSSZ ^
  --output-root data/artifacts/v3_experiments ^
  --name v3_humidity_mean_ensemble_72to24 ^
  --method mean ^
  --clip-min 0 --clip-max 100 ^
  --leaderboard-path data/artifacts/v3_experiments/leaderboard.csv
```

이 CLI는 component `predictions_test.csv`를 key 기준으로 정렬/검증한 뒤 평균 또는 median ensemble을 만들고, 동일한 V2 evaluator로 `metrics_test.json`, station/region/horizon report, `leaderboard_humidity.csv`를 갱신한다.

로컬 웹에서 예측/실험 현황을 보려면 dependency 없는 dashboard 서버를 실행한다.

```bash
python -m weather_korea_forecast.dashboard.app ^
  --artifact-root data/artifacts/v3_experiments ^
  --host 127.0.0.1 ^
  --port 8765
```

브라우저에서 `http://127.0.0.1:8765/`를 열면 target/track별 best, latest/best alias, leaderboard, comparison report 링크를 볼 수 있다. 정적 HTML snapshot만 만들려면 `--write-html data/artifacts/v3_experiments/dashboard.html`을 사용한다.

### V2 데이터 요구사항

V2는 실제로는 다음 로컬 파일이 채워져 있어야 동작한다.

- `paths.observation_csv`: 다관측소 ASOS hourly CSV
- `paths.era5_csv`: station-level ERA5 extracted CSV 또는 grid source
- `paths.station_metadata_csv`: `station_id, lat, lon, elevation, region_class, coastal_distance_km` plus optional `terrain_class`, `coastal_class`

즉 저장소에는 V2 파이프라인과 config가 포함되어 있고, 사용자는 로컬 데이터 경로만 맞추면 된다.

### V2 다관측소 원본 준비

다관측소 V2 benchmark를 준비할 때는 아래 순서를 권장한다.

1. KMA station metadata 다운로드
2. ASOS hourly 다관측소 CSV 다운로드
3. ERA5 raw netCDF 다운로드
4. station metadata 기준으로 ERA5를 station CSV로 추출
5. V2 `prepare_data -> train -> evaluate -> predict`

관련 파일:

- [asos_station_metadata_v2.yaml](C:\Users\js100\Desktop\coding\gisang\configs\data\asos_station_metadata_v2.yaml)
- [asos_download_multistation_v2.yaml](C:\Users\js100\Desktop\coding\gisang\configs\data\asos_download_multistation_v2.yaml)
- [era5_download_multistation_v2_2024q4.yaml](C:\Users\js100\Desktop\coding\gisang\configs\data\era5_download_multistation_v2_2024q4.yaml)
- [era5_download_multistation_v2_2025q1.yaml](C:\Users\js100\Desktop\coding\gisang\configs\data\era5_download_multistation_v2_2025q1.yaml)
- [run_prepare_multistation_v2_data.sh](C:\Users\js100\Desktop\coding\gisang\scripts\run_prepare_multistation_v2_data.sh)

메타데이터 다운로드:

```bash
python -m weather_korea_forecast.data.download_kma_station_metadata \
  --config configs/data/asos_station_metadata_v2.yaml
```

기본 메타데이터 경로는 공식 KMA DataWiki 표를 사용한다. `KMA API Hub` auth key가 따로 있으면 downloader source를 바꿔 확장할 수 있다.

ASOS 다운로드:

```bash
python -m weather_korea_forecast.data.download_kma_obs \
  --config configs/data/asos_download_multistation_v2.yaml
```

ERA5 다운로드는 비용 제한 때문에 `2024Q4`, `2025Q1`로 나눠 받는 기본 경로를 권장한다.

```bash
python -m weather_korea_forecast.data.download_era5 \
  --config configs/data/era5_download_multistation_v2_2024q4.yaml

python -m weather_korea_forecast.data.download_era5 \
  --config configs/data/era5_download_multistation_v2_2025q1.yaml
```

ERA5 station 추출:

```bash
python -m weather_korea_forecast.data.extract_era5_at_station \
  --era5-path data/raw/era5/asos_multistation_v2_2024q4.nc \
  --station-metadata-path data/raw/metadata/stations_v2.csv \
  --output-path data/raw/era5/asos_multistation_v2_2024q4_station.csv \
  --mode bilinear

python -m weather_korea_forecast.data.extract_era5_at_station \
  --era5-path data/raw/era5/asos_multistation_v2_2025q1.nc \
  --station-metadata-path data/raw/metadata/stations_v2.csv \
  --output-path data/raw/era5/asos_multistation_v2_2025q1_station.csv \
  --mode bilinear

python -m weather_korea_forecast.data.concat_tables \
  --input-path data/raw/era5/asos_multistation_v2_2024q4_station.csv \
  --input-path data/raw/era5/asos_multistation_v2_2025q1_station.csv \
  --output-path data/raw/era5/asos_multistation_v2_station.csv \
  --sort-column station_id \
  --sort-column datetime
```

기본 V2 다관측소 추천 관측소는 다음 12개다.

- `90`, `93`, `108`, `112`, `133`, `138`, `143`, `152`, `156`, `159`, `184`, `192`

`region_class`와 `coastal_distance_km`는 config 기반 수동 매핑을 사용한다. 필요하면 로컬 운영 기준에 맞춰 바꿔도 된다.

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
  prefer_columns:
    - humidity
    - pressure
    - wind_speed
    - precipitation
    - quality_flag
  resample_rule: 1h
  aggregation:
    temp: mean
    humidity: mean
    pressure: mean
    wind_speed: mean
    precipitation: sum
    quality_flag: last
```

병합 규칙은 `priority` 기반이며, `prefer_columns`에 들어간 컬럼은 해당 소스의 값을 우선 사용한다. 위 기본 예시는 `temp`는 ASOS를 유지하고 습도/기압/풍속/강수/품질 플래그는 시간 집계된 AWS 값을 우선 사용한다. `prefer_columns`에 없는 컬럼은 ASOS 우선으로 결측만 보완한다.

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
  ridge baseline 설정이다. `alpha_grid`가 있으면 후보 alpha를 validation loss로 선택한다. V2는 필요 시 `model.alpha_selection`으로 보정 후 holdout MSE 선택을 사용할 수 있다.
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
- `horizon_station_heatmap.png`
- `station_rmse_bar.png`
- `region_rmse_bar.png`
- `daily_max_min_error.png`
- `extreme_target_scatter.png`
- `bias_correction.json`
- `future_feature_metadata.json`
- `worst_case_samples.csv`
- `worst_case_summary.json`
- `feature_importance.csv`
- `horizon_model_metrics.csv`
- `predictions_test_components.csv` for residual experiments
- `daily_target_errors.csv`
- `metrics_daily_target.csv`
- `metrics_humidity_extremes.csv` for humidity-only dry/humid event metrics
- `metrics_target_name_rolling_origin_fold.csv`
- `experiment_summary.json`
- `experiment_summary.md`
- `leaderboard.csv`
- `leaderboard_temp.csv`
- `leaderboard_humidity.csv`
- `leaderboard_observation_only.csv`
- `leaderboard_nwp_assisted.csv`

`leaderboard.csv`에는 최소한 아래 컬럼이 저장된다.

- `experiment_name`
- `version`
- `target_name`
- `model_name`
- `model_type`
- `encoder_length`
- `prediction_length`
- `scaling_mode`
- `forecast_track`
- `uses_future_nwp_features`
- `future_feature_source`
- `operational_valid`
- `num_stations`
- `train_period`
- `val_period`
- `test_period`
- `best_horizon`
- `worst_horizon`
- `rmse_raw`
- `rmse_corrected`
- `mae_raw`
- `mae_corrected`
- `bias_raw`
- `bias_corrected`
- `daily_max_temp_mae`
- `daily_min_temp_mae`
- `diurnal_range_mae`
- `diurnal_range_bias`
- `daily_score`
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

온도 실험에서는 0도 근처/음수 때문에 MAPE가 왜곡될 수 있으므로, 주요 비교 지표는 RMSE, MAE, Bias, horizon별 RMSE/MAE/Bias로 둔다.

V2 평가에서는 추가로 아래를 기본 저장한다.

- raw vs corrected comparison
- bias correction calibration/holdout selection artifact
- horizon별 RMSE / MAE / Bias
- station별 / region별 / season별 breakdown
- station×horizon RMSE heatmap
- daily max/min target error
- daily target range error
- extreme-target scatter
- humidity dry/humid event hit rates and low/high RH MAE when `target_name: humidity`
- worst-case summary JSON
- rolling-origin slice report
- worst-case sample table
- ridge / lightgbm feature importance

## 테스트 및 검증 상태

다음 검증을 반영했다.

- `python -m compileall src` 통과
- synthetic leakage guard: lag/rolling/delta feature가 미래값을 쓰지 않는지, scaler가 train split group stats만 쓰는지 검증
- fallback_torch residual shortcut / gradient clipping contract test 수행
- synthetic 데이터 기준 `build -> train -> evaluate -> predict` smoke test 수행
- synthetic horizon-wise ridge와 residual framework artifact smoke test 수행
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
- V2 true-TFT residual forecasting variant
- V2 true rolling-origin retraining
- V2 humidity 전용 stronger bias correction

## 한 줄 요약

이 프로젝트는 **원본 기상 데이터를 모델이 학습 가능한 시계열 데이터셋으로 변환하고, 학습-예측-평가-개선 루프를 반복할 수 있게 만드는 한국 기상 예측 시스템의 기본 골격**이다.

### V3 temperature MOS experiment order

Run the temperature NWP-assisted/MOS track separately from observation-only baselines:

```bash
python -m weather_korea_forecast.v2.train --config configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml
python -m weather_korea_forecast.v2.train --config configs/v3/experiments/v3_temp_mos_residual_ridge_168to24.yaml
python -m weather_korea_forecast.v2.train --config configs/v3/experiments/v3_temp_mos_horizonwise_residual_ridge_72to24.yaml
python -m weather_korea_forecast.v2.train --config configs/v3/experiments/v3_temp_mos_residual_lgbm_72to24.yaml
```

Then ensemble aligned artifacts with:

```bash
python -m weather_korea_forecast.v2.ensemble \
  --experiment-dir <ridge-72-experiment-dir> \
  --experiment-dir <lgbm-72-experiment-dir> \
  --output-root data/artifacts/v3_experiments \
  --name v3_temp_mos_ensemble_ridge_lgbm_72to24 \
  --method mean \
  --leaderboard-path data/artifacts/v3_experiments/leaderboard.csv
```

V3 residual MOS `predictions_test.csv` includes `baseline_prediction`, `predicted_residual`, `actual_residual`, `prediction_raw`, `prediction_corrected`, `error`, and `abs_error`.  The leaderboard adds `track`, `leakage_risk_note`, and `worst_station_rmse`.  ERA5 reanalysis future features remain backtest-only; use `--operational` during inference to fail fast if a saved experiment still depends on backtest-only future covariates.

## 통합 실험 리포트 생성

V1/V2/V3 실험 결과가 각 experiment directory에 흩어지는 문제를 줄이기 위해
단일 CSV/HTML 리포트를 생성할 수 있다. HTML은 외부 CDN 없이 CSS/JS와 PNG 이미지를
내장하므로 파일 하나만 열어도 주요 plot과 leaderboard를 볼 수 있다.

```bash
PYTHONPATH=src .venv312/Scripts/python.exe -m weather_korea_forecast.reporting.generate_report \
  --experiments-root data/artifacts \
  --output-dir reports \
  --title "기상 V1-V3 통합 실험 리포트" \
  --embed-images thumbnail
```

생성 파일:

- `reports/experiment_report.html`
- `reports/experiment_summary.csv`
- `reports/experiment_summary.json`
- `reports/best_models.csv`
- `reports/failed_or_incomplete_experiments.csv`

HTML 리포트는 `version`, `target_name`, `track`, `model_type`,
`future_feature_source`, `operational_valid`, `rmse_goal_met` 필터를 제공한다.
V3 목표 기준은 temp NWP-assisted MOS `RMSE <= 1.0°C`, temp observation-only
`RMSE <= 2.0°C`, humidity `RMSE <= 10%p`로 자동 계산된다.
단, `oracle`, `decoder_feature_baseline`, `observed_target_oracle`, `rmse=0`인
backtest-only sanity check는 `is_diagnostic=true`로 분리되어 main leaderboard,
KPI, `best_models.csv`, RMSE 목표 달성 수에서 제외된다. `best`/`latest` alias
artifact도 상세 섹션에는 남기지만 main leaderboard 대표 run 산정에서는 제외된다.
CSV/JSON에는 `included_in_main_leaderboard` 컬럼이 함께 저장되며, HTML main
leaderboard와 chart는 이 값이 true인 대표 non-diagnostic run만 사용한다.
이미지는 `--embed-images full|thumbnail|external-assets`로 제어할 수 있으며,
기본값은 standalone HTML을 위한 `full` base64 embed다.
가벼운 공유/검토용 리포트는 `--no-images`로 생성해 CSV와 HTML 중심으로 묶는다.

V3.5 이후 실험에서도 파일 수를 최소화하려면 config에 다음을 둔다.

```yaml
artifacts:
  root_dir: data/artifacts/v3_experiments
  leaderboard_path: data/artifacts/v3_experiments/leaderboard.csv
  profile: minimal
```

`artifacts.profile: minimal`은 report/leaderboard에 필요한 핵심 CSV/JSON
(`predictions_test.csv`, `metrics_summary.json`, target/horizon/station metrics,
`experiment_summary.json`, `leaderboard*.csv`)을 유지하고 plot, worst-case sample,
rolling-origin 상세 등 무거운 보조 산출물 생성을 생략한다. HTML collector는 이
profile을 인식해 missing plot 경고를 내지 않는다. V4 config도 같은 profile 키를
계속 사용한다.

학습/평가 CLI는 기본적으로 이 통합 CSV/HTML 리포트를 자동 갱신한다. 빠른
실험이나 CI에서 리포트 생성을 건너뛰고 싶을 때만 `--no-update-report`를 사용한다.

```bash
PYTHONPATH=src .venv312/Scripts/python.exe -m weather_korea_forecast.v2.train \
  --config configs/v3/experiments/v3_temp_mos_residual_ridge_72to24.yaml
```

V3 NWP-assisted/MOS 실험은 observation-only와 별도 track으로 비교해야 한다.
`future_feature_source: era5_reanalysis` 또는 `era5_reanalysis_backtest`는
backtest-only로 표시되며, 운영 추론에서는 `--operational`이 이러한 모델을 차단한다.
prepared forecast CSV를 쓰는 경우 `prepared_forecast_csv`, `gfs_forecast`,
`ecmwf_forecast`, `kma_forecast` 중 하나로 source를 명시한다.

## Forecast Web Service MVP

G021 adds a service-layer forecast artifact format and API skeleton. Research outputs such as `predictions_test.csv` remain separate from production forecast artifacts under `data/forecasts/`.

Export example:

```bash
PYTHONPATH=src python -m weather_korea_forecast.service.export_forecast \
  --predictions path/to/predictions_inference.csv \
  --station-metadata data/raw/metadata/stations.csv \
  --output-dir data/forecasts \
  --forecast-run-id auto \
  --operational-valid false \
  --backtest-only true
```

API helpers live in `weather_korea_forecast.api.main`. If FastAPI is installed, run with `uvicorn weather_korea_forecast.api.main:app`. Without FastAPI, tests use the same pure-Python endpoint helpers.

Operational warning: until an operational-valid real forecast NWP archive is available, website forecasts must be shown as research/backtest outputs.

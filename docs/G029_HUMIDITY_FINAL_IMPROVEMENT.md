# G029 Humidity Final Improvement

## Result

- Status: **NEAR_PASS**
- Selected candidate: `g029_humidity_final_selection` (source `g029_humidity_no_patch_lgbm_calibrated`)
- Selected humidity RMSE: **10.396%p**
- Selected humidity bias: **0.596%p**
- PASS threshold: `RMSE <= 10.000%p` and `|Bias| <= 2.000%p`
- NEAR_PASS threshold: `RMSE <= 10.500%p`
- Gap to PASS RMSE: `0.396%p`

G029 remains validation-first: the logit-RH candidate reached `9.921%p` on the test split, but its validation holdout RMSE was worse than the official no-patch baseline. It is therefore **not adopted** to avoid test-metric selection.

## Candidate audit

| Candidate | Model | RMSE %p | MAE %p | Bias %p | Validation RMSE %p | Selection |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `g029_humidity_no_patch_lgbm_calibrated` | `operational_residual_lgbm_humidity` | 10.396 | 8.087 | 0.596 | 10.098 | source candidate selected by validation |
| `g029_humidity_logit_rh_lgbm` | `logit_rh_residual_lgbm` | 9.921 | 7.710 | 0.976 | 10.414 | test PASS but validation worse; not adopted |
| `g029_humidity_quantile_or_isotonic` | `no_patch_quantile_or_isotonic_calibration_candidates` | 11.238 | 8.492 | 2.244 | 11.238 | calibration candidates only; not eligible as final model |
| `g029_humidity_final_selection` | `final_selection:operational_residual_lgbm_humidity` | 10.396 | 8.087 | 0.596 | 10.098 | selected final; source no_patch baseline |

## No-new-experiment check

- No patch model was introduced for humidity.
- No hidden humidity ensemble was introduced.
- Allowlist: `g029_humidity_no_patch_lgbm_calibrated`, `g029_humidity_logit_rh_lgbm`, `g029_humidity_quantile_or_isotonic`, `g029_humidity_final_selection`
- Out-of-allowlist candidate ids: none.

## Artifacts

- Final selection: `data/artifacts/g029_humidity_final_selection/g029_humidity_final_selection.json`
- Candidate audit: `data/artifacts/g029_humidity_final_selection/g029_humidity_candidate_audit.csv`
- Evidence JSON: `.omx/reports/g029-humidity-final-evidence.json`

# V4 NWP Patch Feature Foundation

V4 patch features summarize a small grid neighborhood around each station before modeling. They are a spatial extension over station nearest-point MOS, not a replacement for the V3.5 baselines.

## Initial patch sets

| Feature set | Patch sizes | Intended use |
| --- | --- | --- |
| `summary_v1` | `3x3`, `5x5` | Tree-model summaries for first V4 comparison. |
| `coastal_gradient_v1` | `5x5` | Adds upwind/downwind and coast-normal gradients for coastal stations. |

## Summary columns

For each configured variable, emit stable feature names such as:

```text
patch_<variable>_center
patch_<variable>_mean
patch_<variable>_std
patch_<variable>_min
patch_<variable>_max
patch_<variable>_range
patch_<variable>_gradient_x
patch_<variable>_gradient_y
```

Patch-capable experiment summaries should set:

```yaml
patch_features:
  enabled: true
  patch_size: 5
  feature_set: summary_v1
```

## Guardrails

- Patch extraction must preserve issue-time alignment from the prepared forecast archive.
- Missing patch cells are a validation failure for operational mode unless a config explicitly opts into imputation for research runs.
- Patch features should be compared against the V3/V3.5 station-level MOS baselines before adding CNN/ConvLSTM models.

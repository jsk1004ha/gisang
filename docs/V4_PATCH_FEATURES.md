# V4 NWP Patch Feature Foundation

V4 patch features summarize a small grid neighborhood around each station before modeling. They are a spatial extension over station nearest-point MOS, not a replacement for the V3.5 baselines.

## Initial patch sets

| Feature set | Patch sizes | Intended use |
| --- | --- | --- |
| `summary_v1` | `3x3`, `5x5` | Tree-model summaries for first V4 comparison. |
| `coastal_gradient_v1` | `5x5` | Adds upwind/downwind and coast-normal gradients for coastal stations. |

## Summary columns

For each configured variable, emit stable feature names in both legacy and V4
contract forms. The canonical V4 form is `patch_<variable>_<stat>`; the
current tree-model compatibility alias is `<variable>_patch_<stat>`.

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

Compatibility aliases:

```text
<variable>_patch_center
<variable>_patch_mean
<variable>_patch_std
<variable>_patch_min
<variable>_patch_max
<variable>_patch_range
<variable>_patch_gradient_x
<variable>_patch_gradient_y
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
- The first V4-B implementation pads edge cells with `NaN` to preserve fixed
  tensor shape. Operational datasets should report and monitor the NaN rate;
  production inference may later tighten this to a hard failure or explicit
  imputation policy.
- Patch features should be compared against the V3/V3.5 station-level MOS baselines before adding CNN/ConvLSTM models.

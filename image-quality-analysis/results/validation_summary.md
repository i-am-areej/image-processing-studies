# Metric validation results

Spearman rank correlation between degradation severity and metric value.
A value near +1 or -1 means the metric tracks that degradation monotonically.
A value near 0 means the metric is blind to it, which is desirable for every
degradation except the one the metric targets.

## Sensitivity: does each metric detect the damage it targets?

| Degradation | Target metric | Spearman rho | Monotonic? |
|---|---|---|---|
| gaussian_blur | var_laplacian | -0.943 | yes |
| motion_blur | var_laplacian | -0.961 | yes |
| gaussian_noise | noise_sigma | +1.000 | yes |
| jpeg_compression | blockiness | +0.794 | partial |
| underexposure | mean_luminance | -1.000 | yes |
| overexposure | clipped_fraction | +0.986 | yes |
| resolution_loss | high_freq_ratio | -0.394 | NO |

### Per-image breakdown of the target metric

An averaged correlation can hide a metric that works on some scenes and fails
on others. This table shows rho per image, which is where that shows up.

| Degradation | Target metric | checkerboard | gradient | texture |
|---|---|---|---|---|
| gaussian_blur | var_laplacian | -1.00 | -0.83 | -1.00 |
| motion_blur | var_laplacian | -1.00 | -0.88 | -1.00 |
| gaussian_noise | noise_sigma | +1.00 | +1.00 | +1.00 |
| jpeg_compression | blockiness | +0.78 | +0.60 | +1.00 |
| underexposure | mean_luminance | -1.00 | -1.00 | -1.00 |
| overexposure | clipped_fraction | +1.00 | +0.96 | +1.00 |
| resolution_loss | high_freq_ratio | -1.00 | +0.82 | -1.00 |

## Full matrix

| Degradation | var_laplacian | tenengrad | high_freq_ratio | noise_sigma | mean_luminance | clipped_fraction | rms_contrast | michelson_contrast | colourfulness | blockiness |
|---|---|---|---|---|---|---|---|---|---|---|
| gaussian_blur | -0.94 | -0.90 | -0.20 | -0.85 | -0.27 | -0.28 | -0.99 | -0.65 | -1.00 | +0.71 |
| motion_blur | -0.96 | -0.84 | -0.37 | -0.98 | -0.66 | -0.33 | -0.99 | -0.67 | -1.00 | +0.32 |
| gaussian_noise | +1.00 | +1.00 | +1.00 | +1.00 | -0.07 | +0.24 | +0.39 | +1.00 | +1.00 | +0.27 |
| jpeg_compression | +0.66 | +0.63 | +0.71 | -0.06 | +0.26 | +0.31 | +0.33 | +0.56 | -0.37 | +0.79 |
| underexposure | -1.00 | -1.00 | +0.94 | -0.99 | -1.00 | -0.18 | -1.00 | +0.28 | -1.00 | -0.40 |
| overexposure | +0.12 | -0.18 | -0.32 | -0.66 | +1.00 | +0.99 | -0.22 | -0.47 | -0.59 | -0.79 |
| resolution_loss | -0.62 | -0.72 | -0.39 | -0.96 | -0.90 | -0.33 | -0.87 | -0.67 | -0.84 | +0.67 |

## Selectivity

Count of degradations each metric responds strongly to (|rho| >= 0.9).
A count of 1 means the metric is specific. A high count means it detects
that something is wrong but cannot say what.

| Metric | Strong responses | Responds to |
|---|---|---|
| var_laplacian | 4 | gaussian_blur, motion_blur, gaussian_noise, underexposure |
| tenengrad | 2 | gaussian_noise, underexposure |
| high_freq_ratio | 2 | gaussian_noise, underexposure |
| noise_sigma | 4 | motion_blur, gaussian_noise, underexposure, resolution_loss |
| mean_luminance | 3 | underexposure, overexposure, resolution_loss |
| clipped_fraction | 1 | overexposure |
| rms_contrast | 3 | gaussian_blur, motion_blur, underexposure |
| michelson_contrast | 1 | gaussian_noise |
| colourfulness | 4 | gaussian_blur, motion_blur, gaussian_noise, underexposure |
| blockiness | 0 | none |

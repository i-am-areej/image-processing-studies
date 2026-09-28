# Restoration benchmark results

PSNR in dB and SSIM, averaged over images. The `none` row is the
do-nothing baseline: a filter that does not beat it is making things worse.

## denoise

### Mean PSNR (dB) by noise sigma

| Filter | 5 | 10 | 20 | 30 | 40 | mean |
|---|---|---|---|---|---|---|
| none | 34.11 | 28.18 | 22.47 | 19.21 | 16.92 | 24.18 |
| gaussian_s1.0 | 31.08 | 29.15 | 26.82 | 25.12 | 23.69 | 27.17 |
| gaussian_s2.0 | 29.60 | 28.05 | 26.01 | 24.61 | 23.51 | 26.36 |
| median_k3 | 35.56 | 32.19 | 27.90 | 25.10 | 23.01 | 28.75 |
| median_k5 | 32.69 | 30.66 | 27.94 | 25.97 | 24.37 | 28.33 |
| bilateral | 43.05 | 35.89 | 25.76 | 20.70 | 17.68 | 28.62 |
| nlm_h8 | 37.40 | 35.00 | 28.63 | 21.60 | 17.51 | 28.03 |
| nlm_h15 | 33.97 | 33.29 | 30.42 | 27.95 | 25.01 | 30.13 |

### Mean SSIM by noise sigma

| Filter | 5 | 10 | 20 | 30 | 40 | mean |
|---|---|---|---|---|---|---|
| none | 0.8678 | 0.6892 | 0.4797 | 0.3668 | 0.2963 | 0.5400 |
| gaussian_s1.0 | 0.8783 | 0.8513 | 0.7724 | 0.6899 | 0.6164 | 0.7617 |
| gaussian_s2.0 | 0.6866 | 0.6829 | 0.6686 | 0.6478 | 0.6235 | 0.6619 |
| median_k3 | 0.9215 | 0.8600 | 0.7147 | 0.5954 | 0.5064 | 0.7196 |
| median_k5 | 0.8185 | 0.7968 | 0.7288 | 0.6514 | 0.5787 | 0.7148 |
| bilateral | 0.9883 | 0.9279 | 0.6007 | 0.4113 | 0.3159 | 0.6488 |
| nlm_h8 | 0.8994 | 0.8978 | 0.8251 | 0.5332 | 0.3370 | 0.6985 |
| nlm_h15 | 0.7823 | 0.7823 | 0.7638 | 0.7524 | 0.6804 | 0.7523 |

### Best filter at each severity

| noise sigma | best by PSNR | gain over baseline (dB) | best by SSIM |
|---|---|---|---|
| 5 | bilateral | +8.94 | bilateral |
| 10 | bilateral | +7.70 | bilateral |
| 20 | nlm_h15 | +7.95 | nlm_h8 |
| 30 | nlm_h15 | +8.74 | nlm_h15 |
| 40 | nlm_h15 | +8.09 | nlm_h15 |

## deblur

### Mean PSNR (dB) by blur sigma

| Filter | 1.0 | 1.5 | 2.0 | 3.0 | mean |
|---|---|---|---|---|---|
| none | 48.02 | 37.57 | 35.55 | 32.97 | 38.53 |
| wiener_oracle_nsr0.001 | 25.53 | 23.29 | 21.81 | 19.46 | 22.52 |
| wiener_oracle_nsr0.01 | 29.47 | 26.22 | 24.25 | 21.78 | 25.43 |
| wiener_oracle_nsr0.1 | 23.63 | 22.29 | 21.34 | 20.23 | 21.87 |
| wiener_assumed_nsr0.01 | 22.52 | 24.40 | 24.25 | 21.93 | 23.27 |
| rl_oracle_10iter | 36.71 | 33.79 | 32.30 | 30.81 | 33.40 |
| rl_oracle_30iter | 38.13 | 34.65 | 32.92 | 31.07 | 34.19 |
| rl_assumed_30iter | 31.81 | 33.30 | 32.92 | 30.83 | 32.22 |
| unsharp | 43.14 | 37.70 | 36.45 | 33.24 | 37.63 |

### Mean SSIM by blur sigma

| Filter | 1.0 | 1.5 | 2.0 | 3.0 | mean |
|---|---|---|---|---|---|
| none | 0.8886 | 0.7761 | 0.6883 | 0.5938 | 0.7367 |
| wiener_oracle_nsr0.001 | 0.9653 | 0.9106 | 0.8368 | 0.6880 | 0.8502 |
| wiener_oracle_nsr0.01 | 0.9700 | 0.8956 | 0.7933 | 0.6302 | 0.8223 |
| wiener_oracle_nsr0.1 | 0.9230 | 0.8117 | 0.7173 | 0.5912 | 0.7608 |
| wiener_assumed_nsr0.01 | 0.7573 | 0.8033 | 0.7933 | 0.6236 | 0.7444 |
| rl_oracle_10iter | 0.9441 | 0.8349 | 0.7460 | 0.6349 | 0.7900 |
| rl_oracle_30iter | 0.9661 | 0.8811 | 0.7923 | 0.6676 | 0.8268 |
| rl_assumed_30iter | 0.7974 | 0.8215 | 0.7923 | 0.6447 | 0.7640 |
| unsharp | 0.9425 | 0.8467 | 0.7379 | 0.6088 | 0.7840 |

### Best filter at each severity

| blur sigma | best by PSNR | gain over baseline (dB) | best by SSIM |
|---|---|---|---|
| 1.0 | unsharp | -4.88 | wiener_oracle_nsr0.01 |
| 1.5 | unsharp | +0.13 | wiener_oracle_nsr0.001 |
| 2.0 | unsharp | +0.90 | wiener_oracle_nsr0.001 |
| 3.0 | unsharp | +0.26 | wiener_oracle_nsr0.001 |


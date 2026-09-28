# Low-light enhancement results

Images darkened by a gain, then sensor noise at sigma = 6.0 added,
then enhanced. SSIM and PSNR are measured against the undarkened original.
`noise_gain` is the factor by which the operator multiplied the noise it was given:
1.0 means it left noise untouched, 3.0 means it tripled it.

## Structural recovery (SSIM, higher is better)

| Operator | gain 0.5 | gain 0.3 | gain 0.2 | gain 0.1 | mean |
|---|---|---|---|---|---|
| none | 0.5689 | 0.3254 | 0.2001 | 0.0927 | 0.2968 |
| gamma_0.45 | 0.6594 | 0.5333 | 0.4197 | 0.2327 | 0.4613 |
| hist_eq | 0.3961 | 0.3348 | 0.2904 | 0.2160 | 0.3093 |
| clahe_c2 | 0.5774 | 0.4655 | 0.3443 | 0.2067 | 0.3985 |
| clahe_c4 | 0.4560 | 0.4151 | 0.3552 | 0.2427 | 0.3672 |
| ssr | 0.5006 | 0.4047 | 0.3279 | 0.2074 | 0.3602 |
| msr | 0.5215 | 0.4194 | 0.3356 | 0.2076 | 0.3710 |
| gamma_clahe | 0.5686 | 0.4901 | 0.4128 | 0.2809 | 0.4381 |

## Noise amplification (lower is better)

| Operator | gain 0.5 | gain 0.3 | gain 0.2 | gain 0.1 | mean |
|---|---|---|---|---|---|
| none | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| gamma_0.45 | 1.24x | 1.70x | 2.14x | 3.16x | 2.06x |
| hist_eq | 4.69x | 6.95x | 9.05x | 13.12x | 8.45x |
| clahe_c2 | 2.47x | 2.70x | 2.66x | 2.19x | 2.50x |
| clahe_c4 | 3.90x | 4.35x | 4.34x | 3.53x | 4.03x |
| ssr | 2.87x | 4.38x | 5.77x | 8.74x | 5.44x |
| msr | 2.64x | 4.16x | 5.58x | 8.77x | 5.29x |
| gamma_clahe | 2.73x | 3.62x | 4.45x | 6.15x | 4.24x |

## Brightness achieved (mean luma; the clean originals average 128)

| Operator | gain 0.5 | gain 0.3 | gain 0.2 | gain 0.1 |
|---|---|---|---|---|
| none | 63 | 38 | 25 | 13 |
| gamma_0.45 | 128 | 101 | 83 | 59 |
| hist_eq | 124 | 125 | 125 | 127 |
| clahe_c2 | 84 | 63 | 46 | 28 |
| clahe_c4 | 97 | 78 | 63 | 41 |
| ssr | 192 | 185 | 180 | 169 |
| msr | 190 | 186 | 182 | 170 |
| gamma_clahe | 118 | 103 | 91 | 76 |

## Brightness is not recovery

| Operator | mean luma | SSIM | noise amplification |
|---|---|---|---|
| none | 35 | 0.2968 | 1.00x |
| gamma_0.45 | 93 | 0.4613 | 2.06x |
| hist_eq | 125 | 0.3093 | 8.45x |
| clahe_c2 | 55 | 0.3985 | 2.50x |
| clahe_c4 | 70 | 0.3672 | 4.03x |
| ssr | 181 | 0.3602 | 5.44x |
| msr | 182 | 0.3710 | 5.29x |
| gamma_clahe | 97 | 0.4381 | 4.24x |

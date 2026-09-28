# Classical image restoration: what actually recovers an image, and what just looks like it does

A controlled benchmark of classical denoising and deblurring filters. The
degradation is applied by the benchmark, so the clean original is available as
ground truth and recovery can be *measured* rather than eyeballed.

PSNR and SSIM are both implemented from scratch in `restore/quality.py`, as are
Wiener and Richardson-Lucy deconvolution in `restore/filters.py`. The details
those hide are the interesting part.

## The experiment

**Denoise track.** Clean image, add Gaussian noise at sigma in {5, 10, 20, 30, 40},
restore with eight filters, compare to the clean original.

**Deblur track.** Clean image, Gaussian blur at sigma in {1.0, 1.5, 2.0, 3.0},
restore with nine configurations, compare to the clean original.

Deconvolution needs the blur kernel. In a lab you can measure it; in the field
you estimate it and the estimate is wrong. So every deconvolution runs twice:
**oracle**, given the true sigma, and **assumed**, given a fixed guess of 2.0.
That split is what separates "deconvolution works" from "deconvolution works when
you already know the answer".

Sanity check: at blur sigma 2.0 the oracle and assumed configurations coincide by
construction, and they produce identical numbers (SSIM 0.7933 both). That
confirms the plumbing is right.

## Results

### Denoising: the filters work, and the winner changes with severity

| Noise sigma | Best filter | PSNR gain over doing nothing |
|---|---|---|
| 5 | bilateral | +8.94 dB |
| 10 | bilateral | +7.70 dB |
| 20 | non-local means (h=15) | +7.95 dB |
| 30 | non-local means (h=15) | +8.74 dB |
| 40 | non-local means (h=15) | +8.09 dB |

Bilateral filtering wins at low noise because it preserves edges while smoothing
flat regions. Non-local means takes over above sigma 20, because once noise
swamps local structure the only reliable signal left is redundancy across the
whole image. Every denoiser beats the baseline, which is the reassuring case.

### Deblurring: getting the PSF wrong is worse than not deblurring at all

Mean SSIM, blur sigma 1.0:

| Configuration | SSIM |
|---|---|
| do nothing | 0.8886 |
| Wiener, **oracle** PSF | **0.9700** |
| Wiener, **assumed** PSF (off by 1.0) | 0.7573 |
| Richardson-Lucy, **oracle** PSF | 0.9661 |
| Richardson-Lucy, **assumed** PSF | 0.7974 |

With the correct kernel, Wiener deconvolution lifts SSIM from 0.889 to 0.970.
With the kernel wrong by one sigma, it drops to 0.757, **below the untouched
image**. The same holds for Richardson-Lucy. Deconvolution is not a little worse
when misinformed; it is actively harmful, because it amplifies frequencies the
true blur never attenuated and turns them into ringing.

The practical reading: PSF estimation is not a preprocessing detail, it is the
whole problem. A restoration system that cannot estimate its own blur kernel
should not deblur.

### PSNR and SSIM disagree, and PSNR is the one that is wrong

| Blur sigma | Best by PSNR | Best by SSIM |
|---|---|---|
| 1.0 | unsharp mask | Wiener oracle |
| 1.5 | unsharp mask | Wiener oracle |
| 2.0 | unsharp mask | Wiener oracle |
| 3.0 | unsharp mask | Wiener oracle |

By PSNR, unsharp masking beats every deconvolution and Wiener looks catastrophic
(25.5 dB against a 48.0 dB baseline at sigma 1.0). By SSIM, Wiener wins
everywhere and unsharp masking is mid-table.

Unsharp masking is in this benchmark as a control. It adds back a scaled
high-pass residual: it recovers no information at all, it only raises local
contrast at edges. That it wins on PSNR is the finding. PSNR is a per-pixel error
measure, so it punishes the mild ringing that successful deconvolution produces
far more than it punishes the genuine loss of detail that unsharp masking leaves
untouched. Reporting PSNR alone would have ranked the method that recovers
nothing above the method that recovers the image.

## Running it

```bash
pip install -r requirements.txt
python run_benchmark.py                          # synthetic scenes, ~10 s
python run_benchmark.py --images ~/photos --limit 6
```

Outputs land in `results/`: two CSVs, three figures and `benchmark_summary.md`.

## Layout

```
restore/quality.py    PSNR and SSIM, implemented from the 2004 SSIM paper
restore/filters.py    denoisers, Wiener and Richardson-Lucy deconvolution, unsharp control
run_benchmark.py      the experiment and its report
```

## Limitations

- The synthetic scenes skew PSNR. One of the three is a smooth gradient that blur
  barely damages, which inflates the do-nothing baseline to 48 dB at sigma 1.0.
  Re-run on real photographs before quoting any absolute number.
- Only Gaussian blur is tested. Motion blur and out-of-focus discs have very
  different spectra, and a Gaussian PSF assumption will fail on both.
- Noise is additive, white and Gaussian. Real sensor noise is signal-dependent
  and partly Poisson, which is the regime Richardson-Lucy was designed for and
  where it should do relatively better than shown here.
- The "assumed" PSF is always 2.0. A more honest experiment would sweep the
  assumed sigma across a range and plot degradation against estimation error.
- SSIM is computed per channel and averaged. The original paper works on
  luminance.

## Open questions

1. **Sweep the PSF error.** Fix the true blur, vary the assumed sigma from 0.5x
   to 2x, and plot SSIM against the ratio. That converts the headline finding
   from one number into a curve, and tells you how accurate kernel estimation
   has to be before deconvolution pays off.
2. **Add blind deconvolution** and see how close it gets to the oracle.
3. **Test on real degradations**, not synthetic ones: genuinely out-of-focus or
   motion-blurred photographs, where the true kernel is unknown.
4. **Add a learned baseline** such as DnCNN or a small U-Net and report where the
   classical filters remain competitive on cost.
5. **Validate the metric disagreement against human judgement.** Run a small
   forced-choice study: show pairs of restorations and ask which looks better.
   If people prefer the Wiener result, that is direct evidence PSNR is
   misleading here.

## References

- Wang, Bovik, Sheikh and Simoncelli (2004). Image quality assessment: from error visibility to structural similarity. *IEEE TIP*, 13(4).
- Tomasi and Manduchi (1998). Bilateral filtering for gray and color images. *ICCV*.
- Buades, Coll and Morel (2005). A non-local algorithm for image denoising. *CVPR*.
- Richardson (1972). Bayesian-based iterative method of image restoration. *JOSA*, 62(1).
- Lucy (1974). An iterative technique for the rectification of observed distributions. *Astronomical Journal*, 79.

## Acknowledgements

The initial implementation of this repository was developed with AI assistance (Claude). Every method is
implemented from the primary source cited in the references of each study, not adapted from an existing
library, and the findings reported are the output of the code in this repository rather than quoted from
elsewhere. Results are reproducible by running the commands in each study's README.

# Which no-reference image quality metrics can you actually trust?

A small empirical study of classical no-reference image quality metrics. It asks
a narrow question and answers it with data: **when an image is damaged in a known
way, which metrics detect it, and which metrics are fooled?**

No trained models, no GPU, no reference image. Everything here is a classical
image-processing operator with a published source, applied to images degraded by
a controlled amount so that the ground-truth ordering is known.

---

## Running it

```bash
pip install -r requirements.txt

# validation experiment on built-in synthetic scenes
python run_validation.py

# the same experiment on your own photographs (much more interesting)
python run_validation.py --images ~/path/to/photos --limit 12

# score a real folder and flag the damaged images
python analyze_folder.py --images ~/path/to/photos --panels 10
```

`analyze_folder.py` flags images relative to the set rather than against absolute
thresholds, because an absolute blur cutoff is meaningless across different
cameras and subjects. A uniformly excellent set will still have a bottom 10%, so
read the metric values and not only the flags.

## Layout

```
iqa/metrics.py      ten no-reference metrics, each with its citation
iqa/degrade.py      seven degradations with controllable severity
iqa/samples.py      synthetic scenes so it runs with no downloads
iqa/visualize.py    heatmap, sensitivity curves, diagnostic panels
run_validation.py   the experiment: do the metrics track known damage?
analyze_folder.py   the application: score and flag a real image folder
```

## Where the metrics fail

Stated plainly, because a methods section that claims everything worked is not
believable.

- `high_freq_energy_ratio` is only meaningful on images that have high-frequency
  content to lose. On near-flat scenes it is dominated by resampling artefacts
  and can move in the wrong direction. It should be gated on a baseline detail
  measure; it currently is not.
- `noise_sigma` (Immerkaer) assumes locally-linear image content, so strong edges
  inflate the estimate. It reads texture as noise.
- `blockiness` assumes an 8x8 DCT grid aligned to the image origin. Any crop that
  is not a multiple of 8, or any resize after compression, breaks the alignment
  and the metric collapses. This is the likely reason its JPEG correlation is
  only +0.79.
- The synthetic scenes are not photographs. They have no depth of field, no
  natural noise and no lens aberration. Every number here should be re-measured
  on real images before it is believed.
- Severity is linear in the parameter, not in perceived damage. Spearman only
  needs monotonicity so this is defensible, but it means the curve shapes in
  `sensitivity_curves.png` are not perceptual.

## Open questions

Any of these turns this from an exercise into a contribution.

1. **Correct for the noise confound.** Estimate noise first, then compensate the
   Laplacian variance for it, and test whether the corrected measure separates
   blur from noise. Finding 1 is the motivation.
2. **Gate the frequency metric on baseline detail** and re-measure whether the
   resolution-loss correlation recovers to near -1.
3. **Validate against human judgement.** Correlate these metrics with subjective
   scores on a public database such as LIVE, TID2013 or KonIQ-10k. That is the
   test that matters, and it makes the study comparable to published work.
4. **Compare against a learned metric** such as BRISQUE, NIQE or a small CNN, on
   equal footing, and report where the classical metrics are competitive. They
   often are, at a fraction of the cost.
5. **Multi-metric diagnosis.** Finding 3 says no single metric identifies the
   damage but the pattern might. Train a simple classifier on the ten-metric
   vector to predict which degradation was applied, and report the confusion
   matrix.

## References

- Pech-Pacheco et al. (2000). Diatom autofocusing in brightfield microscopy: a comparative study. *ICPR*.
- Krotkov (1987). Focusing. *International Journal of Computer Vision*, 1(3).
- Immerkaer (1996). Fast noise variance estimation. *Computer Vision and Image Understanding*, 64(2).
- Hasler and Suesstrunk (2003). Measuring colourfulness in natural images. *SPIE Human Vision and Electronic Imaging*.
- Wang, Bovik and Evan (2000). Blind measurement of blocking artifacts in images. *ICIP*.

## Acknowledgements

The initial implementation of this repository was developed with AI assistance (Claude). Every method is
implemented from the primary source cited in the references of each study, not adapted from an existing
library, and the findings reported are the output of the code in this repository rather than quoted from
elsewhere. Results are reproducible by running the commands in each study's README.

# Low-light enhancement: brightness is not recovery

Six classical enhancement operators, measured on whether they actually recover
image structure or merely make a dark photograph look brighter. All six are
implemented in `enhance.py` rather than imported: gamma correction, global
histogram equalisation, CLAHE at two clip limits, single-scale Retinex and
multi-scale Retinex.

## The experiment

A clean image is darkened by a known exposure gain in {0.5, 0.3, 0.2, 0.1}, then
sensor noise at sigma 6 is added, in that order, because that is the order a
camera does it. Each operator is then applied.

Because the undarkened original is retained, three things are measured
separately that are usually conflated:

| Measure | Question it answers |
|---|---|
| SSIM against the clean original | did it recover structure? |
| mean luma | does it look correctly exposed? |
| noise amplification | what did that cost? |

Noise amplification is the ratio of Immerkaer noise sigma after enhancement to
before, so 1.0 means untouched and 8.0 means the operator multiplied the noise
eightfold. The quality metrics come from the sibling `image-quality-analysis`
project; PSNR and SSIM come from `image-restoration-benchmark`.

## Results

Averaged over all exposure gains. The clean originals average luma 128.

| Operator | SSIM | mean luma | noise amplification |
|---|---|---|---|
| none (darkened input) | 0.2968 | 35 | 1.00x |
| **gamma 0.45** | **0.4613** | 93 | **2.06x** |
| gamma then CLAHE | 0.4381 | 97 | 4.24x |
| CLAHE (clip 2) | 0.3985 | 55 | 2.50x |
| multi-scale Retinex | 0.3710 | 182 | 5.29x |
| CLAHE (clip 4) | 0.3672 | 70 | 4.03x |
| single-scale Retinex | 0.3602 | 181 | 5.44x |
| histogram equalisation | 0.3093 | 125 | 8.45x |

### The finding

**Rank these operators by brightness and by recovery, and you get two different
orderings.**

Global histogram equalisation produces the most natural-looking exposure of any
operator: mean luma 125 against the clean originals' 128. It is also the worst
performer in the table. It recovers almost nothing, lifting SSIM from 0.2968 to
0.3093, a gain of 0.012, while amplifying noise **8.45 times**. It looks correct
and is nearly worthless.

Gamma correction is the opposite. It leaves the image visibly dim at luma 93,
well short of the 128 target, and yet it recovers more than five times as much
structure as histogram equalisation, +0.165 SSIM, at a quarter of the noise cost.

Retinex sits at the far end: single- and multi-scale Retinex overshoot to luma
181 and 182, brighter than the original ever was, for mid-table recovery and more
than five times the noise.

The reason is that histogram equalisation stretches the histogram to fill the
available range, and in a dark image most of that histogram *is* noise, so it
stretches the noise along with everything else. CLAHE's clip limit exists
precisely to bound this, and the data shows it working: clip 2 amplifies noise
2.50x against equalisation's 8.45x. Raising the clip to 4 moves it back toward
equalisation's behaviour, 4.03x noise for *lower* SSIM, which is the clip limit
doing exactly what it is documented to do.

The practical reading: any enhancement pipeline evaluated by eye, or by a
brightness or entropy statistic, will select histogram equalisation. Evaluated
against ground truth, it selects gamma. If you cannot measure against a clean
reference, you cannot tell these two apart, and every operator here will look
like an improvement.

## Running it

```bash
pip install -r requirements.txt
python enhance.py
python enhance.py --images ~/photos --limit 6
```

Outputs: `results/enhancement_summary.md`, `enhancement_results.csv`,
`enhancement_tradeoff.png` (recovery against noise cost) and
`enhancement_examples.png`.

## Limitations

- The darkening model is a linear gain plus additive Gaussian noise. Real short
  exposures are Poisson-limited, and real cameras apply a non-linear tone curve
  before you ever see the pixels.
- No colour-constancy evaluation. Retinex is a colour theory as much as a
  brightness one, and measuring only luma understates it.
- Operator parameters are fixed. Gamma at 0.45 happens to suit this darkening
  range; a fair comparison would tune each operator per image.
- SSIM against the clean original assumes the goal is to reproduce the
  well-exposed photograph. A photographer might reasonably want something else.
- Synthetic scenes only, by default. Run it on real low-light photographs before
  quoting any number.

## Open questions

1. **Tune each operator per image** rather than fixing parameters, and check
   whether gamma still wins when histogram equalisation and Retinex are given
   the same courtesy.
2. **Denoise first, then enhance.** Every operator here amplifies existing
   noise. Chaining the non-local means denoiser from the sibling restoration
   project ahead of enhancement should change the ranking, and by how much is
   the interesting part.
3. **Add a learned baseline** such as Zero-DCE or RetinexNet and report where
   the classical operators remain competitive.
4. **Test on a real low-light dataset** with paired short and long exposures,
   such as SID or LOL, which gives genuine ground truth instead of simulation.
5. **Check whether the brightness-recovery divergence holds for human viewers.**
   If people prefer the histogram-equalised result despite its lower SSIM, that
   is a finding about the metric, not about the operator.

## References

- Pizer et al. (1987). Adaptive histogram equalization and its variations. *Computer Vision, Graphics, and Image Processing*, 39(3).
- Zuiderveld (1994). Contrast limited adaptive histogram equalization. *Graphics Gems IV*.
- Jobson, Rahman and Woodell (1997). A multiscale Retinex for bridging the gap between color images and the human observation of scenes. *IEEE TIP*, 6(7).
- Land (1977). The Retinex theory of color vision. *Scientific American*, 237(6).

## Acknowledgements

The initial implementation of this repository was developed with AI assistance (Claude). Every method is
implemented from the primary source cited in the references of each study, not adapted from an existing
library, and the findings reported are the output of the code in this repository rather than quoted from
elsewhere. Results are reproducible by running the commands in each study's README.

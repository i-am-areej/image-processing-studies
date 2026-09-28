# Image Processing Studies

Three small empirical studies on classical image processing. Each asks a narrow question and answers it with
a controlled experiment: apply a known degradation, measure what happens, report what the data says including
where it contradicts the intuition the study started with.

No trained models, no GPU, no external datasets required. Everything runs on CPU in seconds to a couple of
minutes using NumPy, OpenCV, SciPy and Matplotlib. Every metric, filter and quality measure is implemented
from its primary source rather than imported from a toolbox, because the details a library hides are usually
the interesting ones.

| Study | Question | Headline finding |
|---|---|---|
| [image-quality-analysis](image-quality-analysis/) | Which no-reference quality metrics can be trusted? | Gaussian noise inflates **every** high-frequency sharpness metric. Laplacian variance rises roughly 16-fold under σ = 30 noise and correlates +1.00 with noise severity, so a noisy image reads as *sharper* than a clean one. |
| [image-restoration-benchmark](image-restoration-benchmark/) | What actually recovers a damaged image, and what only looks like it does? | Deconvolution with a point spread function wrong by one σ scores **below the untouched image** (SSIM 0.757 vs a 0.889 baseline). Separately, PSNR and SSIM rank the methods in opposite orders, with PSNR preferring an unsharp-mask control that recovers no information at all. |
| [low-light-enhancement](low-light-enhancement/) | Does enhancement recover structure, or just brightness? | Histogram equalisation produces the most natural-looking exposure (mean luma 125 against a reference of 128) while recovering almost nothing (+0.012 SSIM) and amplifying noise 8.45-fold. Gamma correction looks dimmer and recovers five times more structure at a quarter of the noise cost. |

## Why these three

They form a chain. The first asks whether you can even measure image damage without a reference. The second
asks whether you can undo it. The third asks whether the methods that make an image *look* better actually
make it better. Each one kept producing the same underlying lesson from a different direction: apparent
quality and measured quality come apart, and which metric you choose decides which method wins.

The studies also compose in code. The restoration benchmark imports the quality metrics from the first study;
the enhancement study imports from both. They ship as one repository for that reason.

## Running them

```bash
pip install -r requirements.txt

cd image-quality-analysis      && python run_validation.py
cd image-restoration-benchmark && python run_benchmark.py
cd low-light-enhancement       && python enhance.py
```

Each writes figures, CSVs and a markdown summary into its own `results/` directory, which is committed so the
reported numbers can be checked without running anything. All three accept `--images path/to/folder` to run on
real photographs instead of the built-in synthetic scenes, and every number in this README should be
re-measured on real images before being relied on.

## A note on method

Two things were done deliberately and are worth stating.

**Correlations are computed within each image, then averaged.** Pooling all images into a single correlation
would measure scene-to-scene variation rather than the severity response being studied. Making that mistake
during development dropped the blur correlation from −0.94 to −0.60, which is how the error was caught.

**Every study reports where it fails.** The frequency-domain metric inverts on smooth scenes. Immerkær noise
estimation reads strong edges as noise. The blockiness measure breaks on any crop that is not a multiple of 8.
Each README has a limitations section, and each lists open questions that would extend the work.

## Acknowledgements

The initial implementation of this repository was developed with AI assistance (Claude). Every method is
implemented from the primary source cited in the references of each study, not adapted from an existing
library, and the findings reported are the output of the code in this repository rather than quoted from
elsewhere. Results are reproducible by running the commands above.

## References

Each study lists its own. Across the repository: Pech-Pacheco et al. (ICPR 2000), Krotkov (IJCV 1987),
Immerkær (CVIU 1996), Hasler & Süsstrunk (SPIE 2003), Wang, Bovik & Evan (ICIP 2000), Wang, Bovik, Sheikh &
Simoncelli (IEEE TIP 2004), Tomasi & Manduchi (ICCV 1998), Buades, Coll & Morel (CVPR 2005), Richardson
(JOSA 1972), Lucy (AJ 1974), Pizer et al. (CVGIP 1987), Zuiderveld (Graphics Gems IV 1994), Jobson, Rahman &
Woodell (IEEE TIP 1997), Land (Scientific American 1977).

## Licence

MIT. See [LICENSE](LICENSE).

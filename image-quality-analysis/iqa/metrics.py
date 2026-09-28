"""No-reference image quality metrics.

Every metric here is a classical image-processing operator with a published
source. None of them needs a trained model, a GPU, or a reference image, which
is what makes them usable on an arbitrary folder of real photographs.

References
----------
Pech-Pacheco et al. (2000), "Diatom autofocusing in brightfield microscopy:
    a comparative study", ICPR.                         -> variance_of_laplacian
Krotkov (1987), "Focusing", Int. J. Computer Vision.    -> tenengrad
Immerkaer (1996), "Fast noise variance estimation",
    Computer Vision and Image Understanding 64(2).      -> noise_sigma
Hasler & Suesstrunk (2003), "Measuring colourfulness in
    natural images", SPIE Human Vision and Electronic
    Imaging.                                            -> colourfulness
Wang, Bovik & Evan (2000), "Blind measurement of blocking
    artifacts in images", ICIP.                         -> blockiness
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "variance_of_laplacian",
    "tenengrad",
    "high_freq_energy_ratio",
    "noise_sigma",
    "mean_luminance",
    "clipped_fraction",
    "rms_contrast",
    "michelson_contrast",
    "colourfulness",
    "blockiness",
    "compute_all",
    "METRIC_NAMES",
]


def _as_gray(image: np.ndarray) -> np.ndarray:
    """Return a float32 single-channel view of `image` in the range [0, 255]."""
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"expected HxW or HxWx3, got shape {image.shape}")
    return gray.astype(np.float32)


# --------------------------------------------------------------------------
# Sharpness / blur
# --------------------------------------------------------------------------

def variance_of_laplacian(image: np.ndarray) -> float:
    """Focus measure: variance of the Laplacian response.

    High on sharp images, low on blurred ones. The single most widely used
    no-reference blur indicator.
    """
    gray = _as_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def tenengrad(image: np.ndarray) -> float:
    """Focus measure: mean squared Sobel gradient magnitude.

    Independent of `variance_of_laplacian` in how it weights edges, so the two
    disagreeing is itself informative.
    """
    gray = _as_gray(image)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def high_freq_energy_ratio(image: np.ndarray, cutoff: float = 0.25,
                           window: bool = True) -> float:
    """Fraction of Fourier magnitude lying outside a radius of `cutoff`.

    `cutoff` is expressed as a fraction of the smaller spectrum half-dimension.
    Blurring is a low-pass operation, so this falls as blur increases.

    A separable Hann window is applied first. The DFT treats the image as
    periodic, so the discontinuity between the left and right edges (and top
    and bottom) injects broadband high-frequency energy that has nothing to do
    with image content. On detailed scenes that artefact is swamped by real
    detail, but on smooth scenes it dominates the measurement and the metric
    stops being meaningful. Pass window=False to reproduce the unwindowed
    behaviour.

    Known limitation: even with windowing, this metric is only informative on
    images that have high-frequency content to lose. See README, "Where the
    metrics fail".
    """
    gray = _as_gray(image)
    if window:
        h_win = np.hanning(gray.shape[0]).astype(np.float32)
        w_win = np.hanning(gray.shape[1]).astype(np.float32)
        gray = gray * np.outer(h_win, w_win)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = spectrum.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    limit = cutoff * min(cy, cx)
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    return float(spectrum[radius > limit].sum() / total)


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------

_IMMERKAER_MASK = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32
)


def noise_sigma(image: np.ndarray) -> float:
    """Immerkaer's fast estimate of additive Gaussian noise sigma.

    Convolves with a mask that annihilates locally-linear image content, so
    what survives is dominated by noise. Edges inflate the estimate slightly,
    which is the known limitation of the method.
    """
    gray = _as_gray(image)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    response = cv2.filter2D(gray, cv2.CV_32F, _IMMERKAER_MASK)
    # Trim the border, where filter2D's padding fabricates content.
    response = response[1:-1, 1:-1]
    scale = np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2))
    return float(scale * np.abs(response).sum())


# --------------------------------------------------------------------------
# Exposure and contrast
# --------------------------------------------------------------------------

def mean_luminance(image: np.ndarray) -> float:
    """Mean of the luma channel, in [0, 255]."""
    return float(_as_gray(image).mean())


def clipped_fraction(image: np.ndarray, low: int = 2, high: int = 253) -> float:
    """Fraction of pixels crushed to black or blown to white.

    Clipped pixels have lost their information irrecoverably, so this is a
    measure of damage rather than of appearance.
    """
    gray = _as_gray(image)
    clipped = (gray <= low) | (gray >= high)
    return float(clipped.mean())


def rms_contrast(image: np.ndarray) -> float:
    """Standard deviation of luma. The standard global contrast measure."""
    return float(_as_gray(image).std())


def michelson_contrast(image: np.ndarray, percentile: float = 1.0) -> float:
    """(max - min) / (max + min) on percentile-trimmed luma.

    Trimming stops a handful of outlier pixels pinning the result at 1.0.
    """
    gray = _as_gray(image)
    lo = float(np.percentile(gray, percentile))
    hi = float(np.percentile(gray, 100.0 - percentile))
    if hi + lo <= 1e-6:
        return 0.0
    return float((hi - lo) / (hi + lo))


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------

def colourfulness(image: np.ndarray) -> float:
    """Hasler and Suesstrunk's colourfulness metric.

    Returns 0.0 for a greyscale input. Useful for spotting washed-out or
    heavily colour-cast photographs.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return 0.0
    b, g, r = (c.astype(np.float32) for c in cv2.split(image))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


# --------------------------------------------------------------------------
# Compression artefacts
# --------------------------------------------------------------------------

def blockiness(image: np.ndarray, block: int = 8) -> float:
    """Ratio of gradient energy on the JPEG block grid to gradient energy off it.

    JPEG quantises 8x8 DCT blocks independently, which leaves discontinuities
    exactly on the block boundaries. A clean image scores near 1.0; a heavily
    compressed one scores above it.
    """
    gray = _as_gray(image)
    h, w = gray.shape
    if h < 2 * block or w < 2 * block:
        return 1.0

    d_horizontal = np.abs(np.diff(gray, axis=1))
    columns = np.arange(d_horizontal.shape[1])
    on_grid_h = d_horizontal[:, (columns + 1) % block == 0]
    off_grid_h = d_horizontal[:, (columns + 1) % block != 0]

    d_vertical = np.abs(np.diff(gray, axis=0))
    rows = np.arange(d_vertical.shape[0])
    on_grid_v = d_vertical[(rows + 1) % block == 0, :]
    off_grid_v = d_vertical[(rows + 1) % block != 0, :]

    on_grid = np.concatenate([on_grid_h.ravel(), on_grid_v.ravel()])
    off_grid = np.concatenate([off_grid_h.ravel(), off_grid_v.ravel()])
    if off_grid.size == 0 or off_grid.mean() < 1e-6:
        return 1.0
    return float(on_grid.mean() / off_grid.mean())


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------

METRIC_NAMES = (
    "var_laplacian",
    "tenengrad",
    "high_freq_ratio",
    "noise_sigma",
    "mean_luminance",
    "clipped_fraction",
    "rms_contrast",
    "michelson_contrast",
    "colourfulness",
    "blockiness",
)


def compute_all(image: np.ndarray) -> dict[str, float]:
    """Compute every metric for one BGR or greyscale image."""
    return {
        "var_laplacian": variance_of_laplacian(image),
        "tenengrad": tenengrad(image),
        "high_freq_ratio": high_freq_energy_ratio(image),
        "noise_sigma": noise_sigma(image),
        "mean_luminance": mean_luminance(image),
        "clipped_fraction": clipped_fraction(image),
        "rms_contrast": rms_contrast(image),
        "michelson_contrast": michelson_contrast(image),
        "colourfulness": colourfulness(image),
        "blockiness": blockiness(image),
    }

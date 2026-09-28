"""Classical restoration filters.

Denoisers and deblurrers, each with its published source. Richardson-Lucy and
Wiener deconvolution are implemented here rather than imported, because the
regularisation choices are the interesting part.

References
----------
Tomasi and Manduchi (1998), "Bilateral filtering for gray and color images",
    ICCV.                                               -> bilateral
Buades, Coll and Morel (2005), "A non-local algorithm for image denoising",
    CVPR.                                               -> non_local_means
Richardson (1972), JOSA 62(1); Lucy (1974), Astronomical
    Journal 79.                                         -> richardson_lucy
Wiener (1949), "Extrapolation, Interpolation and Smoothing
    of Stationary Time Series".                         -> wiener_deconvolution
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["DENOISERS", "DEBLURRERS", "gaussian", "median", "bilateral",
           "non_local_means", "wiener_deconvolution", "richardson_lucy",
           "unsharp_mask", "identity", "gaussian_psf"]


def identity(image: np.ndarray, **_) -> np.ndarray:
    """Do nothing. The baseline every filter must beat to be worth running."""
    return image.copy()


# --------------------------------------------------------------------------
# Denoising
# --------------------------------------------------------------------------

def gaussian(image: np.ndarray, sigma: float = 1.0, **_) -> np.ndarray:
    """Linear smoothing. Removes noise and detail without distinguishing them."""
    if sigma <= 0:
        return image.copy()
    return cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def median(image: np.ndarray, ksize: int = 3, **_) -> np.ndarray:
    """Rank filter. Strong against impulse noise, weaker against Gaussian."""
    ksize = int(ksize) | 1  # must be odd
    return cv2.medianBlur(image, max(3, ksize))


def bilateral(image: np.ndarray, sigma_color: float = 25.0,
              sigma_space: float = 5.0, **_) -> np.ndarray:
    """Edge-preserving smoothing: weights by spatial and intensity distance."""
    return cv2.bilateralFilter(image, d=-1, sigmaColor=sigma_color,
                               sigmaSpace=sigma_space)


def non_local_means(image: np.ndarray, strength: float = 8.0, **_) -> np.ndarray:
    """Averages similar patches from across the image, not just nearby pixels."""
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(
            image, None, float(strength), float(strength), 7, 21)
    return cv2.fastNlMeansDenoising(image, None, float(strength), 7, 21)


# --------------------------------------------------------------------------
# Deblurring
# --------------------------------------------------------------------------

def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """Normalised 2-D Gaussian point spread function."""
    size = int(size) | 1
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    psf = kernel_1d @ kernel_1d.T
    return (psf / psf.sum()).astype(np.float64)


def _pad_psf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Place the PSF into an image-sized array, centred on the origin."""
    padded = np.zeros(shape, dtype=np.float64)
    h, w = psf.shape
    padded[:h, :w] = psf
    return np.roll(padded, (-(h // 2), -(w // 2)), axis=(0, 1))


def _per_channel(fn, image: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Apply a single-channel restoration to each channel of a colour image."""
    if image.ndim == 2:
        return fn(image, *args, **kwargs)
    channels = [fn(image[:, :, c], *args, **kwargs) for c in range(image.shape[2])]
    return np.stack(channels, axis=2)


def _wiener_channel(channel: np.ndarray, psf: np.ndarray, nsr: float) -> np.ndarray:
    data = channel.astype(np.float64)
    otf = np.fft.fft2(_pad_psf(psf, data.shape))
    # H* / (|H|^2 + NSR). NSR standing in for the noise-to-signal power ratio.
    restored = np.fft.ifft2(
        np.fft.fft2(data) * np.conj(otf) / (np.abs(otf) ** 2 + nsr)
    ).real
    return np.clip(restored, 0, 255)


def wiener_deconvolution(image: np.ndarray, psf_sigma: float = 2.0,
                         nsr: float = 0.01, **_) -> np.ndarray:
    """Linear inverse filtering, regularised by an assumed noise level.

    `nsr` is the knob: too small and noise explodes, too large and the result
    stays blurred. That trade-off is the point of the experiment.
    """
    psf = gaussian_psf(int(6 * psf_sigma) | 1, psf_sigma)
    return _per_channel(_wiener_channel, image, psf, nsr).astype(np.uint8)


def _rl_channel(channel: np.ndarray, psf: np.ndarray, iterations: int) -> np.ndarray:
    data = channel.astype(np.float64) + 1e-6
    estimate = np.full_like(data, data.mean())
    psf_mirror = psf[::-1, ::-1]
    for _ in range(iterations):
        convolved = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT)
        ratio = data / np.maximum(convolved, 1e-6)
        correction = cv2.filter2D(ratio, -1, psf_mirror, borderType=cv2.BORDER_REFLECT)
        estimate = estimate * correction
        estimate = np.clip(estimate, 0, 255)
    return estimate


def richardson_lucy(image: np.ndarray, psf_sigma: float = 2.0,
                    iterations: int = 15, **_) -> np.ndarray:
    """Iterative maximum-likelihood deconvolution for Poisson noise.

    Converges towards the sharp image but amplifies noise as iterations grow,
    so the iteration count is itself a regularisation parameter.
    """
    psf = gaussian_psf(int(6 * psf_sigma) | 1, psf_sigma)
    return _per_channel(_rl_channel, image, psf, int(iterations)).astype(np.uint8)


def unsharp_mask(image: np.ndarray, sigma: float = 1.5, amount: float = 1.0,
                 **_) -> np.ndarray:
    """Add back a scaled high-pass residual. Sharpens apparent detail only.

    Included as a control: it raises sharpness metrics without recovering any
    information, which is exactly the behaviour the experiment should expose.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    sharpened = cv2.addWeighted(image.astype(np.float32), 1.0 + amount,
                                blurred.astype(np.float32), -amount, 0.0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


DENOISERS = {
    "none": (identity, {}),
    "gaussian_s1.0": (gaussian, {"sigma": 1.0}),
    "gaussian_s2.0": (gaussian, {"sigma": 2.0}),
    "median_k3": (median, {"ksize": 3}),
    "median_k5": (median, {"ksize": 5}),
    "bilateral": (bilateral, {"sigma_color": 25.0, "sigma_space": 5.0}),
    "nlm_h8": (non_local_means, {"strength": 8.0}),
    "nlm_h15": (non_local_means, {"strength": 15.0}),
}

DEBLURRERS = {
    "none": (identity, {}),
    "wiener_nsr0.001": (wiener_deconvolution, {"psf_sigma": 2.0, "nsr": 0.001}),
    "wiener_nsr0.01": (wiener_deconvolution, {"psf_sigma": 2.0, "nsr": 0.01}),
    "wiener_nsr0.1": (wiener_deconvolution, {"psf_sigma": 2.0, "nsr": 0.1}),
    "rl_10iter": (richardson_lucy, {"psf_sigma": 2.0, "iterations": 10}),
    "rl_30iter": (richardson_lucy, {"psf_sigma": 2.0, "iterations": 30}),
    "unsharp": (unsharp_mask, {"sigma": 1.5, "amount": 1.0}),
}


def apply_filter(table: dict, name: str, image: np.ndarray) -> np.ndarray:
    """Apply a named filter from DENOISERS or DEBLURRERS."""
    if name not in table:
        raise KeyError(f"unknown filter {name!r}; choose from {sorted(table)}")
    fn, kwargs = table[name]
    return fn(image, **kwargs)

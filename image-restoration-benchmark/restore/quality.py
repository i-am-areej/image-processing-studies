"""Full-reference quality measures: PSNR and SSIM.

SSIM is implemented here from the original paper rather than imported, because
the details that matter (Gaussian window rather than uniform, the stabilising
constants, per-channel handling) are exactly the details a library hides.

Reference
---------
Wang, Bovik, Sheikh and Simoncelli (2004), "Image quality assessment: from
error visibility to structural similarity", IEEE Transactions on Image
Processing 13(4), 600-612.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["psnr", "ssim", "mse"]


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    """Mean squared error over all pixels and channels."""
    a = reference.astype(np.float64)
    b = test.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB.

    Returns infinity for identical images, which callers should expect and
    handle rather than averaging blindly.
    """
    error = mse(reference, test)
    if error <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / error))


def _ssim_single_channel(a: np.ndarray, b: np.ndarray, data_range: float,
                         sigma: float) -> float:
    """SSIM for one channel, using an 11x11 Gaussian window as in the paper."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    a = a.astype(np.float64)
    b = b.astype(np.float64)

    # Truncating the Gaussian at 3.5 sigma gives the paper's 11x11 window at
    # the standard sigma of 1.5.
    radius = int(3.5 * sigma + 0.5)
    ksize = 2 * radius + 1

    def blur(image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (ksize, ksize), sigma,
                                borderType=cv2.BORDER_REFLECT)

    mu_a = blur(a)
    mu_b = blur(b)
    mu_a_sq, mu_b_sq, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    # Var(X) = E[X^2] - E[X]^2, computed with the same weighting as the means.
    sigma_a_sq = blur(a * a) - mu_a_sq
    sigma_b_sq = blur(b * b) - mu_b_sq
    sigma_ab = blur(a * b) - mu_ab

    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)

    # Discard the border, where the window overlaps reflected content.
    if ssim_map.shape[0] > 2 * radius and ssim_map.shape[1] > 2 * radius:
        ssim_map = ssim_map[radius:-radius, radius:-radius]
    return float(ssim_map.mean())


def ssim(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0,
         sigma: float = 1.5) -> float:
    """Structural similarity index, averaged over channels.

    Ranges from -1 to 1, where 1 means the images are identical.
    """
    if reference.shape != test.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {test.shape}")
    if reference.ndim == 2:
        return _ssim_single_channel(reference, test, data_range, sigma)
    scores = [
        _ssim_single_channel(reference[:, :, c], test[:, :, c], data_range, sigma)
        for c in range(reference.shape[2])
    ]
    return float(np.mean(scores))

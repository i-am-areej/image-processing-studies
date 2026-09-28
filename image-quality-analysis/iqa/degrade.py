"""Synthetic degradations with known, controllable severity.

These exist so the metrics in `iqa.metrics` can be validated. If you apply a
degradation whose strength you chose yourself, you know the ground-truth
ordering, and you can ask whether each metric recovers it.

Each function takes a BGR uint8 image and a severity in [0, 1], where 0 means
"leave the image alone" and 1 means the strongest setting in the sweep.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["DEGRADATIONS", "apply", "gaussian_blur", "motion_blur",
           "gaussian_noise", "jpeg_compression", "underexposure",
           "overexposure", "resolution_loss"]


def _clip(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, severity: float) -> np.ndarray:
    """Defocus. sigma sweeps 0 -> 5 px."""
    sigma = 5.0 * severity
    if sigma <= 1e-3:
        return image.copy()
    ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def motion_blur(image: np.ndarray, severity: float, angle: float = 30.0) -> np.ndarray:
    """Camera shake along a line. Kernel length sweeps 1 -> 25 px."""
    length = int(round(1 + 24 * severity))
    if length <= 1:
        return image.copy()
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    total = kernel.sum()
    if total <= 0:
        return image.copy()
    return cv2.filter2D(image, -1, kernel / total)


def gaussian_noise(image: np.ndarray, severity: float, seed: int = 0) -> np.ndarray:
    """Sensor noise. sigma sweeps 0 -> 30 grey levels."""
    sigma = 30.0 * severity
    if sigma <= 1e-3:
        return image.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, image.shape)
    return _clip(image.astype(np.float32) + noise)


def jpeg_compression(image: np.ndarray, severity: float) -> np.ndarray:
    """Lossy compression. Quality sweeps 95 -> 5."""
    quality = int(round(95 - 90 * severity))
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image.copy()
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def underexposure(image: np.ndarray, severity: float) -> np.ndarray:
    """Too little light. Gain sweeps 1.0 -> 0.15."""
    gain = 1.0 - 0.85 * severity
    return _clip(image.astype(np.float32) * gain)


def overexposure(image: np.ndarray, severity: float) -> np.ndarray:
    """Too much light, with genuine highlight clipping. Gain sweeps 1.0 -> 3.0."""
    gain = 1.0 + 2.0 * severity
    return _clip(image.astype(np.float32) * gain)


def resolution_loss(image: np.ndarray, severity: float) -> np.ndarray:
    """Downscale then upscale back. Scale factor sweeps 1.0 -> 0.15."""
    factor = 1.0 - 0.85 * severity
    if factor >= 0.999:
        return image.copy()
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(w * factor)), max(1, int(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


DEGRADATIONS = {
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
    "jpeg_compression": jpeg_compression,
    "underexposure": underexposure,
    "overexposure": overexposure,
    "resolution_loss": resolution_loss,
}

# Which metric each degradation is expected to move. Used by the validation
# script to separate "the metric worked" from "the metric happened to move".
EXPECTED_SENSITIVE_METRIC = {
    "gaussian_blur": "var_laplacian",
    "motion_blur": "var_laplacian",
    "gaussian_noise": "noise_sigma",
    "jpeg_compression": "blockiness",
    "underexposure": "mean_luminance",
    "overexposure": "clipped_fraction",
    "resolution_loss": "high_freq_ratio",
}


def apply(name: str, image: np.ndarray, severity: float) -> np.ndarray:
    """Apply a named degradation at a given severity in [0, 1]."""
    if name not in DEGRADATIONS:
        raise KeyError(f"unknown degradation {name!r}; choose from {sorted(DEGRADATIONS)}")
    return DEGRADATIONS[name](image, float(np.clip(severity, 0.0, 1.0)))

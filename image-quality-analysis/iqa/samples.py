"""Synthetic test scenes, so the pipeline runs with no downloads.

These are not a substitute for real photographs. They exist so `make check`
works on a fresh clone and so the validation experiment has content with known
structure: edges at several orientations, smooth gradients, fine texture and
saturated colour, which is the range the metrics need to be exercised over.

Swap in real images as soon as you have them. `analyze_folder.py` takes any
directory of JPEG or PNG files.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["checkerboard_scene", "gradient_scene", "texture_scene", "make_all"]


def checkerboard_scene(size: int = 512, square: int = 32) -> np.ndarray:
    """Hard edges at two orientations, plus coloured discs."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    tile = (np.indices((size, size)).sum(axis=0) // square) % 2
    canvas[tile == 0] = (235, 235, 235)
    canvas[tile == 1] = (25, 25, 25)
    for cx, cy, radius, colour in [
        (140, 150, 60, (40, 60, 220)),
        (330, 200, 45, (60, 190, 70)),
        (250, 370, 70, (210, 140, 30)),
    ]:
        cv2.circle(canvas, (cx, cy), radius, colour, thickness=-1, lineType=cv2.LINE_AA)
    cv2.line(canvas, (0, size - 40), (size, 40), (255, 255, 255), 3, cv2.LINE_AA)
    return canvas


def gradient_scene(size: int = 512) -> np.ndarray:
    """Smooth ramps. Sensitive to banding, quantisation and exposure shifts."""
    x = np.linspace(0, 255, size, dtype=np.float32)
    y = np.linspace(0, 255, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    blue = xx
    green = yy
    red = 255.0 - 0.5 * (xx + yy)
    canvas = np.dstack([blue, green, red])
    return np.clip(canvas, 0, 255).astype(np.uint8)


def texture_scene(size: int = 512, seed: int = 7) -> np.ndarray:
    """Band-limited noise texture. Stands in for fine detail such as fabric or gravel."""
    rng = np.random.default_rng(seed)
    field = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    field = cv2.GaussianBlur(field, (0, 0), 1.6)
    field -= field.min()
    denominator = field.max() if field.max() > 0 else 1.0
    field = field / denominator
    base = (field * 200.0 + 30.0).astype(np.float32)
    canvas = np.dstack([base * 0.85, base * 0.95, base * 1.05])
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    cv2.rectangle(canvas, (60, 60), (200, 200), (20, 20, 200), thickness=4)
    cv2.rectangle(canvas, (300, 280), (450, 430), (200, 200, 20), thickness=4)
    return canvas


def make_all() -> dict[str, np.ndarray]:
    """Return every synthetic scene, keyed by name."""
    return {
        "checkerboard": checkerboard_scene(),
        "gradient": gradient_scene(),
        "texture": texture_scene(),
    }

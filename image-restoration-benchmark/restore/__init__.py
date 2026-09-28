"""Classical image restoration: denoising, deblurring and full-reference quality."""
from . import filters, quality  # noqa: F401

__all__ = ["filters", "quality"]

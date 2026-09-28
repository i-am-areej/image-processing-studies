"""Plotting for the quality-analysis pipeline.

Matplotlib only, no seaborn, so the dependency list stays short. Every figure
is written to disk rather than shown, so the scripts run headless.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plot_heatmap", "plot_sensitivity_curves", "plot_diagnostic_panel",
           "plot_distributions", "plot_degradation_strip"]

_DPI = 140


def plot_heatmap(matrix: dict[str, dict[str, float]], metric_names: list[str],
                 out_path: Path) -> None:
    """Degradation x metric correlation matrix."""
    degradations = list(matrix)
    data = np.array([[matrix[d][m] for m in metric_names] for d in degradations])

    fig, ax = plt.subplots(figsize=(1.05 * len(metric_names) + 3,
                                    0.55 * len(degradations) + 2.5))
    image = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(degradations)))
    ax.set_yticklabels(degradations, fontsize=9)

    for i in range(len(degradations)):
        for j in range(len(metric_names)):
            value = data[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.55 else "black")

    ax.set_title("Spearman correlation: degradation severity vs metric value",
                 fontsize=11, pad=12)
    fig.colorbar(image, ax=ax, shrink=0.8, label="rho")
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)


def plot_sensitivity_curves(records, expected: dict[str, str], out_path: Path) -> None:
    """Target metric value against severity, one panel per degradation."""
    degradations = list(expected)
    cols = 4
    rows = int(np.ceil(len(degradations) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.9 * rows), squeeze=False)

    for index, degradation in enumerate(degradations):
        ax = axes[index // cols][index % cols]
        metric_name = expected[degradation]
        subset = [r for r in records if r["degradation"] == degradation]
        for image_name in sorted({r["image"] for r in subset}):
            points = sorted(
                ((r["severity"], r[metric_name]) for r in subset if r["image"] == image_name),
                key=lambda t: t[0],
            )
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.3, label=image_name)
        ax.set_title(f"{degradation}\n-> {metric_name}", fontsize=9)
        ax.set_xlabel("severity", fontsize=8)
        ax.set_ylabel(metric_name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if index == 0:
            ax.legend(fontsize=6, loc="best")

    for index in range(len(degradations), rows * cols):
        axes[index // cols][index % cols].axis("off")

    fig.suptitle("Does each metric move monotonically with the damage it targets?",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)


def plot_diagnostic_panel(image: np.ndarray, values: dict[str, float],
                          title: str, out_path: Path) -> None:
    """Per-image panel: original, Laplacian response, histogram, FFT spectrum."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    laplacian = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3))
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else gray,
                   cmap=None if image.ndim == 3 else "gray")
    axes[0].set_title("input", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(laplacian, cmap="inferno")
    axes[1].set_title(f"|Laplacian|  (var = {values['var_laplacian']:.0f})", fontsize=9)
    axes[1].axis("off")

    axes[2].hist(gray.ravel(), bins=64, range=(0, 255), color="#1f3864")
    axes[2].set_title(
        f"luma histogram\nmean {values['mean_luminance']:.0f}, "
        f"clipped {values['clipped_fraction'] * 100:.1f}%", fontsize=9)
    axes[2].tick_params(labelsize=7)

    axes[3].imshow(spectrum, cmap="viridis")
    axes[3].set_title(f"log spectrum  (HF ratio = {values['high_freq_ratio']:.3f})",
                      fontsize=9)
    axes[3].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)


def plot_degradation_strip(image: np.ndarray, degradation_name: str,
                           apply_fn, severities, out_path: Path) -> None:
    """One image shown across a severity sweep, for the report."""
    fig, axes = plt.subplots(1, len(severities), figsize=(2.5 * len(severities), 2.9))
    for ax, severity in zip(np.atleast_1d(axes), severities):
        damaged = apply_fn(degradation_name, image, float(severity))
        ax.imshow(cv2.cvtColor(damaged, cv2.COLOR_BGR2RGB))
        ax.set_title(f"s = {severity:.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(degradation_name, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)


def plot_distributions(records, metric_names, out_path: Path) -> None:
    """Distribution of each metric across a real image set."""
    cols = 4
    rows = int(np.ceil(len(metric_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.6 * rows), squeeze=False)

    for index, metric_name in enumerate(metric_names):
        ax = axes[index // cols][index % cols]
        values = [r[metric_name] for r in records]
        ax.hist(values, bins=min(30, max(5, len(values) // 3)), color="#1f3864")
        median = float(np.median(values))
        ax.axvline(median, color="#c23b22", linewidth=1.4,
                   label=f"median {median:.3g}")
        ax.set_title(metric_name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

    for index in range(len(metric_names), rows * cols):
        axes[index // cols][index % cols].axis("off")

    fig.suptitle("Metric distributions across the image set", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)

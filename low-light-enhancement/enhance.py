#!/usr/bin/env python3
"""Does low-light image enhancement actually help, or does it just look brighter?

The question
------------
Histogram equalisation, CLAHE, gamma correction and Retinex all make a dark
photograph look better. But enhancement cannot create information that the
sensor never captured, and every one of these operators amplifies whatever noise
is present along with the signal. So: which enhancement recovers the most
structure, and what does each one cost in noise?

Method
------
Take a clean image, darken it by a known gain, optionally add sensor noise, then
enhance. Because the clean original is retained, recovery is measured with SSIM
and PSNR rather than judged by eye. Noise amplification is measured separately
with Immerkaer's estimator, so "looks brighter" and "is better" can be
distinguished.

Operators (all implemented here, none imported from a toolbox):
  gamma           per-pixel power law
  hist_eq         global histogram equalisation on the luma channel
  clahe           contrast-limited adaptive equalisation, tile-local
  ssr             single-scale Retinex
  msr             multi-scale Retinex
  gamma_clahe     gamma then CLAHE, to test whether the two compose

Usage
-----
  python enhance.py
  python enhance.py --images ~/photos --limit 6
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "image-quality-analysis"))
sys.path.insert(0, str(_HERE.parent / "image-restoration-benchmark"))

from iqa import metrics as iqa_metrics, samples as iqa_samples  # noqa: E402
from restore import quality  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DARKEN_GAINS = [0.5, 0.3, 0.2, 0.1]
NOISE_SIGMA = 6.0  # added after darkening, as a sensor would


# --------------------------------------------------------------------------
# Enhancement operators
# --------------------------------------------------------------------------

def identity(image: np.ndarray) -> np.ndarray:
    return image.copy()


def gamma(image: np.ndarray, g: float = 0.45) -> np.ndarray:
    """Power law on normalised intensity. g < 1 brightens."""
    table = (np.linspace(0, 1, 256) ** g * 255.0).astype(np.uint8)
    return cv2.LUT(image, table)


def _on_luma(image: np.ndarray, fn) -> np.ndarray:
    """Apply a single-channel operator to L in LAB, preserving colour."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = fn(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def hist_eq(image: np.ndarray) -> np.ndarray:
    """Global histogram equalisation. Maximises entropy, ignores locality."""
    return _on_luma(image, cv2.equalizeHist)


def clahe(image: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Contrast-limited adaptive equalisation.

    The clip limit is what stops it amplifying noise without bound in flat
    regions, which is exactly what plain histogram equalisation does.
    """
    operator = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return _on_luma(image, operator.apply)


def _retinex_single(channel: np.ndarray, sigma: float) -> np.ndarray:
    """log(I) - log(I * G_sigma): the reflectance estimate at one scale."""
    data = channel.astype(np.float32) + 1.0
    blurred = cv2.GaussianBlur(data, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return np.log(data) - np.log(blurred + 1.0)


def _normalise(array: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(array, 1.0), np.percentile(array, 99.0)
    if hi - lo < 1e-6:
        return np.zeros_like(array, dtype=np.uint8)
    scaled = (array - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def ssr(image: np.ndarray, sigma: float = 40.0) -> np.ndarray:
    """Single-scale Retinex. Removes illumination, keeps reflectance."""
    channels = [_normalise(_retinex_single(image[:, :, c], sigma))
                for c in range(image.shape[2])]
    return np.stack(channels, axis=2)


def msr(image: np.ndarray, sigmas=(15.0, 80.0, 250.0)) -> np.ndarray:
    """Multi-scale Retinex: average the log-reflectance across three scales."""
    channels = []
    for c in range(image.shape[2]):
        stacked = np.mean([_retinex_single(image[:, :, c], s) for s in sigmas], axis=0)
        channels.append(_normalise(stacked))
    return np.stack(channels, axis=2)


def gamma_clahe(image: np.ndarray) -> np.ndarray:
    """Gamma first to lift the midtones, then CLAHE for local contrast."""
    return clahe(gamma(image, 0.6), clip=2.0, tiles=8)


OPERATORS = {
    "none": identity,
    "gamma_0.45": lambda im: gamma(im, 0.45),
    "hist_eq": hist_eq,
    "clahe_c2": lambda im: clahe(im, 2.0, 8),
    "clahe_c4": lambda im: clahe(im, 4.0, 8),
    "ssr": ssr,
    "msr": msr,
    "gamma_clahe": gamma_clahe,
}


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def load_images(folder: Path | None, limit: int) -> dict[str, np.ndarray]:
    if folder is not None:
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        images = {}
        for path in paths[:limit]:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            images[path.stem] = image
        if images:
            print(f"Loaded {len(images)} image(s) from {folder}")
            return images
        print(f"No readable images under {folder}; using synthetic scenes.")
    print("Using synthetic scenes.")
    return {name: cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            for name, image in iqa_samples.make_all().items()}


def darken(image: np.ndarray, gain: float, sigma: float, seed: int = 0) -> np.ndarray:
    """Simulate a short exposure: scale down, then add sensor noise."""
    rng = np.random.default_rng(seed)
    dark = image.astype(np.float32) * gain
    dark = dark + rng.normal(0.0, sigma, dark.shape)
    return np.clip(dark, 0, 255).astype(np.uint8)


def run(images: dict[str, np.ndarray]) -> list[dict]:
    records = []
    for gain in DARKEN_GAINS:
        for name, clean in images.items():
            dark = darken(clean, gain, NOISE_SIGMA)
            noise_before = iqa_metrics.noise_sigma(dark)
            for op_name, op in OPERATORS.items():
                enhanced = op(dark)
                records.append({
                    "gain": gain,
                    "image": name,
                    "operator": op_name,
                    "psnr": quality.psnr(clean, enhanced),
                    "ssim": quality.ssim(clean, enhanced),
                    "mean_luma": iqa_metrics.mean_luminance(enhanced),
                    "noise_sigma": iqa_metrics.noise_sigma(enhanced),
                    "noise_gain": iqa_metrics.noise_sigma(enhanced) / max(noise_before, 1e-6),
                    "clipped_frac": iqa_metrics.clipped_fraction(enhanced),
                })
        print(f"  gain = {gain} done")
    return records


def mean_of(records, operator: str, field: str, gain=None) -> float:
    values = [r[field] for r in records if r["operator"] == operator
              and (gain is None or r["gain"] == gain) and np.isfinite(r[field])]
    return float(np.mean(values)) if values else float("nan")


def write_outputs(records, images, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "enhancement_results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = ["# Low-light enhancement results", "",
             f"Images darkened by a gain, then sensor noise at sigma = {NOISE_SIGMA} added,",
             "then enhanced. SSIM and PSNR are measured against the undarkened original.",
             "`noise_gain` is the factor by which the operator multiplied the noise it was given:",
             "1.0 means it left noise untouched, 3.0 means it tripled it.", "",
             "## Structural recovery (SSIM, higher is better)", "",
             "| Operator | " + " | ".join(f"gain {g}" for g in DARKEN_GAINS) + " | mean |",
             "|" + "---|" * (len(DARKEN_GAINS) + 2)]
    for op_name in OPERATORS:
        cells = [f"{mean_of(records, op_name, 'ssim', g):.4f}" for g in DARKEN_GAINS]
        lines.append(f"| {op_name} | " + " | ".join(cells)
                     + f" | {mean_of(records, op_name, 'ssim'):.4f} |")

    lines += ["", "## Noise amplification (lower is better)", "",
              "| Operator | " + " | ".join(f"gain {g}" for g in DARKEN_GAINS) + " | mean |",
              "|" + "---|" * (len(DARKEN_GAINS) + 2)]
    for op_name in OPERATORS:
        cells = [f"{mean_of(records, op_name, 'noise_gain', g):.2f}x" for g in DARKEN_GAINS]
        lines.append(f"| {op_name} | " + " | ".join(cells)
                     + f" | {mean_of(records, op_name, 'noise_gain'):.2f}x |")

    lines += ["", "## Brightness achieved (mean luma; the clean originals average "
              f"{np.mean([iqa_metrics.mean_luminance(i) for i in images.values()]):.0f})", "",
              "| Operator | " + " | ".join(f"gain {g}" for g in DARKEN_GAINS) + " |",
              "|" + "---|" * (len(DARKEN_GAINS) + 1)]
    for op_name in OPERATORS:
        cells = [f"{mean_of(records, op_name, 'mean_luma', g):.0f}" for g in DARKEN_GAINS]
        lines.append(f"| {op_name} | " + " | ".join(cells) + " |")

    lines += ["", "## Brightness is not recovery", "",
              "| Operator | mean luma | SSIM | noise amplification |", "|---|---|---|---|"]
    for op_name in OPERATORS:
        lines.append(f"| {op_name} | {mean_of(records, op_name, 'mean_luma'):.0f} | "
                     f"{mean_of(records, op_name, 'ssim'):.4f} | "
                     f"{mean_of(records, op_name, 'noise_gain'):.2f}x |")

    (out_dir / "enhancement_summary.md").write_text("\n".join(lines) + "\n")


def plot_tradeoff(records, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for op_name in OPERATORS:
        ys = [mean_of(records, op_name, "ssim", g) for g in DARKEN_GAINS]
        style = "--" if op_name == "none" else "-"
        axes[0].plot(DARKEN_GAINS, ys, style, marker="o", markersize=4, label=op_name)
    axes[0].set_xlabel("exposure gain (lower is darker)", fontsize=9)
    axes[0].set_ylabel("SSIM vs clean original", fontsize=9)
    axes[0].set_title("Structural recovery", fontsize=10)
    axes[0].invert_xaxis()
    axes[0].grid(alpha=0.25)

    for op_name in OPERATORS:
        x = mean_of(records, op_name, "noise_gain")
        y = mean_of(records, op_name, "ssim")
        axes[1].scatter(x, y, s=55)
        axes[1].annotate(op_name, (x, y), fontsize=7.5,
                         textcoords="offset points", xytext=(5, 4))
    axes[1].set_xlabel("noise amplification (x)", fontsize=9)
    axes[1].set_ylabel("SSIM vs clean original", fontsize=9)
    axes[1].set_title("The trade-off: recovery against noise cost", fontsize=10)
    axes[1].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    fig.suptitle("Low-light enhancement: does it recover structure or just amplify?",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_examples(images, out_path: Path) -> None:
    name = sorted(images)[0]
    clean = images[name]
    dark = darken(clean, 0.2, NOISE_SIGMA)
    shown = ["none", "gamma_0.45", "hist_eq", "clahe_c2", "ssr", "msr"]
    fig, axes = plt.subplots(1, len(shown) + 1, figsize=(2.7 * (len(shown) + 1), 3.2))
    axes[0].imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
    axes[0].set_title("clean reference", fontsize=8)
    axes[0].axis("off")
    for ax, op_name in zip(axes[1:], shown):
        enhanced = OPERATORS[op_name](dark)
        ax.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{op_name}\nSSIM {quality.ssim(clean, enhanced):.3f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Enhancement at exposure gain 0.2 ({name})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    images = load_images(args.images, args.limit)
    print("Running enhancement sweep...")
    records = run(images)
    write_outputs(records, images, args.out)
    plot_tradeoff(records, args.out / "enhancement_tradeoff.png")
    plot_examples(images, args.out / "enhancement_examples.png")

    print(f"\nWrote results to {args.out}/\n")
    print(f"{'operator':<14} {'SSIM':>7} {'luma':>6} {'noise':>8}   verdict")
    baseline = mean_of(records, "none", "ssim")
    for op_name in OPERATORS:
        ssim_value = mean_of(records, op_name, "ssim")
        verdict = "" if op_name == "none" else (
            "helps" if ssim_value > baseline else "HURTS vs no enhancement")
        print(f"{op_name:<14} {ssim_value:7.4f} "
              f"{mean_of(records, op_name, 'mean_luma'):6.0f} "
              f"{mean_of(records, op_name, 'noise_gain'):7.2f}x   {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

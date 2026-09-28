#!/usr/bin/env python3
"""Benchmark classical restoration filters against known degradations.

The question
-----------
Given an image damaged in a known way, which classical filter recovers the most,
and does the winner change with damage severity? Because the degradation is
applied by this script, the clean original is available as ground truth, so
recovery can be measured with full-reference metrics rather than guessed at.

Two tracks:
  denoise  clean -> add Gaussian noise -> restore -> compare to clean
  deblur   clean -> Gaussian blur -> restore -> compare to clean

Scored with PSNR and SSIM, both implemented in `restore/quality.py`.
A filter is only worth running if it beats the `none` baseline.

Outputs
-------
  results/denoise_results.csv, results/deblur_results.csv
  results/denoise_curves.png, results/deblur_curves.png
  results/restoration_examples.png
  results/benchmark_summary.md

Usage
-----
  python run_benchmark.py
  python run_benchmark.py --images ~/photos --limit 6
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "image-quality-analysis"))

from restore import filters, quality  # noqa: E402

try:
    from iqa import samples as iqa_samples  # reuse the synthetic scenes
except ImportError:  # pragma: no cover
    iqa_samples = None

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NOISE_SIGMAS = [5, 10, 20, 30, 40]
BLUR_SIGMAS = [1.0, 1.5, 2.0, 3.0]


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
        print(f"No readable images under {folder}; falling back to synthetic scenes.")

    if iqa_samples is None:
        raise SystemExit("Synthetic scenes unavailable: run with --images instead.")
    print("Using synthetic scenes from the image-quality-analysis project.")
    return {name: cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            for name, image in iqa_samples.make_all().items()}


def run_denoise(images: dict[str, np.ndarray], rng_seed: int = 0) -> list[dict]:
    records = []
    for sigma in NOISE_SIGMAS:
        for name, clean in images.items():
            rng = np.random.default_rng(rng_seed)
            noisy = np.clip(clean.astype(np.float32) +
                            rng.normal(0, sigma, clean.shape), 0, 255).astype(np.uint8)
            for filter_name in filters.DENOISERS:
                restored = filters.apply_filter(filters.DENOISERS, filter_name, noisy)
                records.append({
                    "track": "denoise",
                    "severity": sigma,
                    "image": name,
                    "filter": filter_name,
                    "psnr": quality.psnr(clean, restored),
                    "ssim": quality.ssim(clean, restored),
                })
        print(f"  denoise: sigma = {sigma} done")
    return records


# Deconvolution needs to know the blur kernel. In a lab you can measure it; in
# the field you must estimate it, and the estimate is wrong. Each configuration
# below is run twice: once with the true sigma ("oracle") and once with a fixed
# guess of 2.0 ("assumed"), which is what isolates PSF sensitivity from
# deconvolution quality.
DEBLUR_CONFIGS = {
    "none": (filters.identity, {}, False),
    "wiener_oracle_nsr0.001": (filters.wiener_deconvolution, {"nsr": 0.001}, True),
    "wiener_oracle_nsr0.01": (filters.wiener_deconvolution, {"nsr": 0.01}, True),
    "wiener_oracle_nsr0.1": (filters.wiener_deconvolution, {"nsr": 0.1}, True),
    "wiener_assumed_nsr0.01": (filters.wiener_deconvolution,
                               {"nsr": 0.01, "psf_sigma": 2.0}, False),
    "rl_oracle_10iter": (filters.richardson_lucy, {"iterations": 10}, True),
    "rl_oracle_30iter": (filters.richardson_lucy, {"iterations": 30}, True),
    "rl_assumed_30iter": (filters.richardson_lucy,
                          {"iterations": 30, "psf_sigma": 2.0}, False),
    "unsharp": (filters.unsharp_mask, {"sigma": 1.5, "amount": 1.0}, False),
}


def run_deblur(images: dict[str, np.ndarray]) -> list[dict]:
    records = []
    for sigma in BLUR_SIGMAS:
        for name, clean in images.items():
            blurred = cv2.GaussianBlur(clean, (0, 0), sigma,
                                       borderType=cv2.BORDER_REFLECT)
            for filter_name, (fn, kwargs, use_oracle) in DEBLUR_CONFIGS.items():
                call_kwargs = dict(kwargs)
                if use_oracle:
                    call_kwargs["psf_sigma"] = sigma
                restored = fn(blurred, **call_kwargs)
                records.append({
                    "track": "deblur",
                    "severity": sigma,
                    "image": name,
                    "filter": filter_name,
                    "psnr": quality.psnr(clean, restored),
                    "ssim": quality.ssim(clean, restored),
                })
        print(f"  deblur: sigma = {sigma} done")
    return records


def mean_by(records, track: str, filter_name: str, severity=None, metric="psnr") -> float:
    values = [r[metric] for r in records
              if r["track"] == track and r["filter"] == filter_name
              and (severity is None or r["severity"] == severity)
              and np.isfinite(r[metric])]
    return float(np.mean(values)) if values else float("nan")


def plot_curves(records, track: str, filter_names, severities, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for metric, ax in zip(("psnr", "ssim"), axes):
        for filter_name in filter_names:
            ys = [mean_by(records, track, filter_name, s, metric) for s in severities]
            style = "--" if filter_name == "none" else "-"
            width = 2.2 if filter_name == "none" else 1.4
            ax.plot(severities, ys, style, marker="o", markersize=4,
                    linewidth=width, label=filter_name)
        ax.set_xlabel("noise sigma" if track == "denoise" else "blur sigma", fontsize=9)
        ax.set_ylabel(metric.upper(), fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    axes[0].set_title(f"{track}: PSNR (higher is better)", fontsize=10)
    axes[1].set_title(f"{track}: SSIM (higher is better)", fontsize=10)
    axes[1].legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle(f"Restoration benchmark: {track}. Dashed line is the do-nothing baseline.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_examples(images: dict[str, np.ndarray], out_path: Path) -> None:
    name = sorted(images)[0]
    clean = images[name]
    rng = np.random.default_rng(0)
    noisy = np.clip(clean.astype(np.float32) + rng.normal(0, 25, clean.shape),
                    0, 255).astype(np.uint8)
    shown = ["none", "gaussian_s1.0", "median_k3", "bilateral", "nlm_h8"]
    fig, axes = plt.subplots(1, len(shown) + 1, figsize=(3.0 * (len(shown) + 1), 3.2))
    axes[0].imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
    axes[0].set_title("clean (reference)", fontsize=9)
    axes[0].axis("off")
    for ax, filter_name in zip(axes[1:], shown):
        restored = filters.apply_filter(filters.DENOISERS, filter_name, noisy)
        ax.imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{filter_name}\nPSNR {quality.psnr(clean, restored):.1f} dB  "
                     f"SSIM {quality.ssim(clean, restored):.3f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Denoising at sigma = 25 ({name})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_summary(records, out_dir: Path) -> None:
    lines = ["# Restoration benchmark results", "",
             "PSNR in dB and SSIM, averaged over images. The `none` row is the",
             "do-nothing baseline: a filter that does not beat it is making things worse.",
             ""]

    for track, table, severities, label in (
        ("denoise", filters.DENOISERS, NOISE_SIGMAS, "noise sigma"),
        ("deblur", DEBLUR_CONFIGS, BLUR_SIGMAS, "blur sigma"),
    ):
        lines += [f"## {track}", "",
                  "### Mean PSNR (dB) by " + label, "",
                  "| Filter | " + " | ".join(str(s) for s in severities) + " | mean |",
                  "|" + "---|" * (len(severities) + 2)]
        for filter_name in table:
            cells = [f"{mean_by(records, track, filter_name, s, 'psnr'):.2f}"
                     for s in severities]
            overall = mean_by(records, track, filter_name, None, "psnr")
            lines.append(f"| {filter_name} | " + " | ".join(cells) + f" | {overall:.2f} |")

        lines += ["", "### Mean SSIM by " + label, "",
                  "| Filter | " + " | ".join(str(s) for s in severities) + " | mean |",
                  "|" + "---|" * (len(severities) + 2)]
        for filter_name in table:
            cells = [f"{mean_by(records, track, filter_name, s, 'ssim'):.4f}"
                     for s in severities]
            overall = mean_by(records, track, filter_name, None, "ssim")
            lines.append(f"| {filter_name} | " + " | ".join(cells) + f" | {overall:.4f} |")

        lines += ["", "### Best filter at each severity", "",
                  f"| {label} | best by PSNR | gain over baseline (dB) | best by SSIM |",
                  "|---|---|---|---|"]
        for severity in severities:
            candidates = [f for f in table if f != "none"]
            best_psnr = max(candidates,
                            key=lambda f: mean_by(records, track, f, severity, "psnr"))
            best_ssim = max(candidates,
                            key=lambda f: mean_by(records, track, f, severity, "ssim"))
            gain = (mean_by(records, track, best_psnr, severity, "psnr")
                    - mean_by(records, track, "none", severity, "psnr"))
            lines.append(f"| {severity} | {best_psnr} | {gain:+.2f} | {best_ssim} |")
        lines.append("")

    (out_dir / "benchmark_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    images = load_images(args.images, args.limit)
    args.out.mkdir(parents=True, exist_ok=True)

    print("Running denoise track...")
    records = run_denoise(images)
    print("Running deblur track...")
    records += run_deblur(images)

    for track, filename in (("denoise", "denoise_results.csv"),
                            ("deblur", "deblur_results.csv")):
        subset = [r for r in records if r["track"] == track]
        with (args.out / filename).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(subset[0]))
            writer.writeheader()
            writer.writerows(subset)

    plot_curves(records, "denoise", list(filters.DENOISERS), NOISE_SIGMAS,
                args.out / "denoise_curves.png")
    plot_curves(records, "deblur", list(DEBLUR_CONFIGS), BLUR_SIGMAS,
                args.out / "deblur_curves.png")
    plot_examples(images, args.out / "restoration_examples.png")
    write_summary(records, args.out)

    print(f"\nWrote results to {args.out}/\n")
    for track, table, severities in (("denoise", filters.DENOISERS, NOISE_SIGMAS),
                                     ("deblur", DEBLUR_CONFIGS, BLUR_SIGMAS)):
        print(f"{track}: best filter by mean PSNR at each severity")
        for severity in severities:
            candidates = [f for f in table if f != "none"]
            best = max(candidates,
                       key=lambda f: mean_by(records, track, f, severity, "psnr"))
            gain = (mean_by(records, track, best, severity, "psnr")
                    - mean_by(records, track, "none", severity, "psnr"))
            verdict = "beats baseline" if gain > 0 else "WORSE than doing nothing"
            print(f"  severity {severity:>4}: {best:<16} {gain:+6.2f} dB  {verdict}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

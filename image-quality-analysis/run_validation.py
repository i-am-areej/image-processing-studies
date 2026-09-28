#!/usr/bin/env python3
"""Validate no-reference quality metrics against known degradations.

The experiment
--------------
For each source image, each degradation type and a sweep of severities, apply
the degradation and compute every metric. A metric is useful for a degradation
if its value moves monotonically with severity, so the score is the Spearman
rank correlation between severity and metric value.

Two numbers matter:

  sensitivity  |rho| for the metric the degradation is supposed to move.
               Near 1.0 means the metric tracks that damage reliably.
  selectivity  Whether the metric stays flat under unrelated degradations.
               A metric that moves for everything cannot tell you what is wrong.

Outputs
-------
  results/validation_correlations.csv   full metric x degradation matrix
  results/validation_heatmap.png        the same matrix, rendered
  results/sensitivity_curves.png        metric value vs severity, per degradation
  results/validation_summary.md         the table to paste into a report

Usage
-----
  python run_validation.py                      # synthetic scenes
  python run_validation.py --images path/to/dir # your own photographs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iqa import degrade, metrics, samples  # noqa: E402
from iqa.visualize import plot_heatmap, plot_sensitivity_curves  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_images(folder: Path | None, limit: int) -> dict[str, np.ndarray]:
    """Load real images from `folder`, or fall back to synthetic scenes."""
    if folder is None:
        print("No --images given; using synthetic scenes.")
        return samples.make_all()

    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        print(f"No images found under {folder}; using synthetic scenes instead.")
        return samples.make_all()

    images: dict[str, np.ndarray] = {}
    for path in paths[:limit]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  skipped unreadable file: {path.name}")
            continue
        # Cap the long edge so the FFT stays cheap and results stay comparable.
        h, w = image.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        images[path.stem] = image
    print(f"Loaded {len(images)} image(s) from {folder}")
    return images


def run_sweep(images: dict[str, np.ndarray], severities: np.ndarray):
    """Return records of (degradation, image, severity, metric values)."""
    records = []
    for degradation in degrade.DEGRADATIONS:
        for name, image in images.items():
            for severity in severities:
                damaged = degrade.apply(degradation, image, float(severity))
                values = metrics.compute_all(damaged)
                records.append({
                    "degradation": degradation,
                    "image": name,
                    "severity": float(severity),
                    **values,
                })
        print(f"  swept {degradation}")
    return records


def correlation_matrix(records) -> dict[str, dict[str, float]]:
    """Spearman rho between severity and each metric, per degradation.

    The correlation is computed within each image and then averaged across
    images. Pooling all images into a single correlation would be wrong: the
    absolute value of a metric varies far more between scenes than it does
    across a severity sweep, so a pooled correlation measures scene-to-scene
    variation and hides the severity response the experiment is asking about.
    """
    matrix: dict[str, dict[str, float]] = {}
    for degradation in degrade.DEGRADATIONS:
        subset = [r for r in records if r["degradation"] == degradation]
        image_names = sorted({r["image"] for r in subset})
        row: dict[str, float] = {}
        for metric_name in metrics.METRIC_NAMES:
            per_image = []
            for image_name in image_names:
                points = [(r["severity"], r[metric_name])
                          for r in subset if r["image"] == image_name]
                severities = [p[0] for p in points]
                values = [p[1] for p in points]
                if len(set(values)) <= 1 or len(set(severities)) <= 1:
                    per_image.append(0.0)
                    continue
                rho, _ = spearmanr(severities, values)
                per_image.append(0.0 if np.isnan(rho) else float(rho))
            row[metric_name] = float(np.mean(per_image)) if per_image else 0.0
        matrix[degradation] = row
    return matrix


def write_outputs(records, matrix, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "validation_correlations.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["degradation", *metrics.METRIC_NAMES])
        for degradation, row in matrix.items():
            writer.writerow([degradation, *[f"{row[m]:.4f}" for m in metrics.METRIC_NAMES]])

    with (out_dir / "validation_raw.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["degradation", "image", "severity", *metrics.METRIC_NAMES]
        )
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# Metric validation results",
        "",
        "Spearman rank correlation between degradation severity and metric value.",
        "A value near +1 or -1 means the metric tracks that degradation monotonically.",
        "A value near 0 means the metric is blind to it, which is desirable for every",
        "degradation except the one the metric targets.",
        "",
        "## Sensitivity: does each metric detect the damage it targets?",
        "",
        "| Degradation | Target metric | Spearman rho | Monotonic? |",
        "|---|---|---|---|",
    ]
    for degradation, target in degrade.EXPECTED_SENSITIVE_METRIC.items():
        rho = matrix[degradation][target]
        verdict = "yes" if abs(rho) >= 0.9 else ("partial" if abs(rho) >= 0.7 else "NO")
        lines.append(f"| {degradation} | {target} | {rho:+.3f} | {verdict} |")

    lines += [
        "",
        "### Per-image breakdown of the target metric",
        "",
        "An averaged correlation can hide a metric that works on some scenes and fails",
        "on others. This table shows rho per image, which is where that shows up.",
        "",
    ]
    image_names = sorted({r["image"] for r in records})
    lines.append("| Degradation | Target metric | " + " | ".join(image_names) + " |")
    lines.append("|" + "---|" * (len(image_names) + 2))
    for degradation, target in degrade.EXPECTED_SENSITIVE_METRIC.items():
        cells = []
        for image_name in image_names:
            points = [(r["severity"], r[target]) for r in records
                      if r["degradation"] == degradation and r["image"] == image_name]
            values = [p[1] for p in points]
            if len(set(values)) <= 1:
                cells.append("n/a")
                continue
            rho, _ = spearmanr([p[0] for p in points], values)
            cells.append("n/a" if np.isnan(rho) else f"{rho:+.2f}")
        lines.append(f"| {degradation} | {target} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Full matrix",
        "",
        "| Degradation | " + " | ".join(metrics.METRIC_NAMES) + " |",
        "|" + "---|" * (len(metrics.METRIC_NAMES) + 1),
    ]
    for degradation, row in matrix.items():
        cells = " | ".join(f"{row[m]:+.2f}" for m in metrics.METRIC_NAMES)
        lines.append(f"| {degradation} | {cells} |")

    lines += [
        "",
        "## Selectivity",
        "",
        "Count of degradations each metric responds strongly to (|rho| >= 0.9).",
        "A count of 1 means the metric is specific. A high count means it detects",
        "that something is wrong but cannot say what.",
        "",
        "| Metric | Strong responses | Responds to |",
        "|---|---|---|",
    ]
    for metric_name in metrics.METRIC_NAMES:
        hits = [d for d in matrix if abs(matrix[d][metric_name]) >= 0.9]
        listed = ", ".join(hits) if hits else "none"
        lines.append(f"| {metric_name} | {len(hits)} | {listed} |")

    (out_dir / "validation_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=None,
                        help="folder of real images; omit to use synthetic scenes")
    parser.add_argument("--limit", type=int, default=8,
                        help="max images to use (default 8)")
    parser.add_argument("--steps", type=int, default=9,
                        help="severity levels from 0 to 1 (default 9)")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    images = load_images(args.images, args.limit)
    if not images:
        print("No usable images.", file=sys.stderr)
        return 1

    severities = np.linspace(0.0, 1.0, args.steps)
    print(f"Sweeping {len(degrade.DEGRADATIONS)} degradations x "
          f"{len(images)} images x {args.steps} severities...")
    records = run_sweep(images, severities)
    matrix = correlation_matrix(records)

    write_outputs(records, matrix, args.out)
    plot_heatmap(matrix, list(metrics.METRIC_NAMES), args.out / "validation_heatmap.png")
    plot_sensitivity_curves(records, degrade.EXPECTED_SENSITIVE_METRIC,
                            args.out / "sensitivity_curves.png")

    print(f"\nWrote results to {args.out}/")
    print("\nSensitivity check:")
    failures = 0
    for degradation, target in degrade.EXPECTED_SENSITIVE_METRIC.items():
        rho = matrix[degradation][target]
        mark = "ok  " if abs(rho) >= 0.9 else ("weak" if abs(rho) >= 0.7 else "FAIL")
        if abs(rho) < 0.7:
            failures += 1
        print(f"  [{mark}] {degradation:<18} -> {target:<18} rho = {rho:+.3f}")
    if failures:
        print(f"\n{failures} metric(s) failed to track their target degradation. "
              f"That is a finding, not a bug: write it up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score a folder of real photographs and flag the damaged ones.

This is the applied half of the project. `run_validation.py` establishes which
metrics can be trusted for which damage; this script applies them to real
images and ranks the results.

Outputs
-------
  results/folder_metrics.csv        one row per image, every metric
  results/folder_distributions.png  where the set sits on each metric
  results/panels/<name>.png         diagnostic panel for the worst offenders
  results/folder_report.md          ranked list with the reason for each flag

Usage
-----
  python analyze_folder.py --images ~/photos
  python analyze_folder.py --images ~/photos --panels 12
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iqa import metrics  # noqa: E402
from iqa.visualize import plot_diagnostic_panel, plot_distributions  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Thresholds are deliberately relative to the set, not absolute. An absolute
# blur threshold is meaningless across different cameras and subjects; a
# percentile threshold adapts to whatever collection you point this at.
FLAG_RULES = {
    "blurred": ("var_laplacian", "low", 10.0),
    "noisy": ("noise_sigma", "high", 90.0),
    "underexposed": ("mean_luminance", "low", 10.0),
    "overexposed": ("clipped_fraction", "high", 90.0),
    "low_contrast": ("rms_contrast", "low", 10.0),
    "compressed": ("blockiness", "high", 90.0),
}


def load_paths(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def score_images(paths: list[Path], max_edge: int) -> list[dict]:
    records = []
    for index, path in enumerate(paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  [{index}/{len(paths)}] unreadable, skipped: {path.name}")
            continue
        h, w = image.shape[:2]
        if max(h, w) > max_edge:
            scale = max_edge / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        values = metrics.compute_all(image)
        records.append({"path": str(path), "name": path.stem,
                        "width": w, "height": h, **values})
        if index % 25 == 0 or index == len(paths):
            print(f"  scored {index}/{len(paths)}")
    return records


def apply_flags(records: list[dict]) -> None:
    """Attach a list of flags to each record, using set-relative thresholds."""
    if not records:
        return
    cutoffs = {}
    for flag, (metric_name, direction, percentile) in FLAG_RULES.items():
        values = [r[metric_name] for r in records]
        cutoffs[flag] = (metric_name, direction, float(np.percentile(values, percentile)))

    for record in records:
        flags = []
        for flag, (metric_name, direction, cutoff) in cutoffs.items():
            value = record[metric_name]
            if direction == "low" and value <= cutoff:
                flags.append(flag)
            elif direction == "high" and value >= cutoff:
                flags.append(flag)
        record["flags"] = ";".join(flags)
        record["flag_count"] = len(flags)


def write_report(records: list[dict], out_dir: Path, panels: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "name", "width", "height", *metrics.METRIC_NAMES,
                  "flags", "flag_count"]
    with (out_dir / "folder_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    worst = sorted(records, key=lambda r: (-r["flag_count"], r["var_laplacian"]))

    lines = [
        "# Image quality report",
        "",
        f"Scored {len(records)} image(s).",
        "",
        "Flags are assigned relative to this set, not against absolute thresholds:",
        "an image is flagged 'blurred' if it falls in the bottom 10% of this set's",
        "Laplacian variance, and so on. That makes the report meaningful on any",
        "collection, but it also means a uniformly excellent set will still have a",
        "bottom 10%. Read the metric values, not only the flags.",
        "",
        "## Most affected images",
        "",
        "| Image | Flags | var_laplacian | noise_sigma | mean_luma | clipped % | blockiness |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in worst[:30]:
        lines.append(
            f"| {record['name']} | {record['flags'] or '-'} | "
            f"{record['var_laplacian']:.0f} | {record['noise_sigma']:.2f} | "
            f"{record['mean_luminance']:.0f} | {record['clipped_fraction'] * 100:.1f} | "
            f"{record['blockiness']:.2f} |"
        )

    counts: dict[str, int] = {}
    for record in records:
        for flag in filter(None, record["flags"].split(";")):
            counts[flag] = counts.get(flag, 0) + 1
    lines += ["", "## Flag frequency", "", "| Flag | Images |", "|---|---|"]
    for flag, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {flag} | {count} |")

    (out_dir / "folder_report.md").write_text("\n".join(lines) + "\n")

    if panels > 0:
        panel_dir = out_dir / "panels"
        panel_dir.mkdir(exist_ok=True)
        for record in worst[:panels]:
            image = cv2.imread(record["path"], cv2.IMREAD_COLOR)
            if image is None:
                continue
            title = f"{record['name']}  [{record['flags'] or 'no flags'}]"
            plot_diagnostic_panel(image, record, title,
                                  panel_dir / f"{record['name']}.png")
        print(f"  wrote {min(panels, len(worst))} diagnostic panel(s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="folder of images")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--panels", type=int, default=8,
                        help="diagnostic panels for the worst N images (default 8)")
    parser.add_argument("--max-edge", type=int, default=1024,
                        help="downscale images whose long edge exceeds this (default 1024)")
    args = parser.parse_args()

    if not args.images.is_dir():
        print(f"Not a directory: {args.images}", file=sys.stderr)
        return 1

    paths = load_paths(args.images)
    if not paths:
        print(f"No images found under {args.images}", file=sys.stderr)
        return 1

    print(f"Scoring {len(paths)} image(s) from {args.images}...")
    records = score_images(paths, args.max_edge)
    if not records:
        print("No readable images.", file=sys.stderr)
        return 1

    apply_flags(records)
    write_report(records, args.out, args.panels)
    plot_distributions(records, list(metrics.METRIC_NAMES),
                       args.out / "folder_distributions.png")

    flagged = sum(1 for r in records if r["flag_count"] > 0)
    print(f"\nWrote results to {args.out}/")
    print(f"{flagged} of {len(records)} image(s) carry at least one flag.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

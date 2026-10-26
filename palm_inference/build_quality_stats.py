"""
Build the palm training-quality JSON used at inference.

Reads palm_all_frame_quality_features.csv, keeps split == train, and writes
the same attribute summary as the nail stats file: p05, p50, p95, mean, std.

Example (from repo root):

    python palm_inference/build_quality_stats.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
NAIL_BRANCH = REPO_ROOT / "src" / "nail_branch"
if str(NAIL_BRANCH) not in sys.path:
    sys.path.insert(0, str(NAIL_BRANCH))

from image_quality import PICTURE_ATTRS, summarize_train  # noqa: E402

# Frame-level CSV headers -> names expected by image_quality.method_d_q_values.
COLUMN_MAP = {
    "brightness": "brightness",
    "contrast": "contrast",
    "saturation": "saturation",
    "highlight_fraction": "highlight_frac",
    "shadow_fraction": "shadow_frac",
    "sharpness": "sharpness",
    "original_width": "orig_w",
    "original_height": "orig_h",
    "original_area": "orig_area",
    "aspect_ratio": "aspect_ratio",
}

# Patient-summary CSV: one row per ID, attributes averaged over frames.
# The *_std columns are within-patient frame spread, not the training distribution.
PATIENT_MEAN_MAP = {
    "brightness_mean": "brightness",
    "contrast_mean": "contrast",
    "saturation_mean": "saturation",
    "highlight_fraction_mean": "highlight_frac",
    "shadow_fraction_mean": "shadow_frac",
    "sharpness_mean": "sharpness",
    "original_width_mean": "orig_w",
    "original_height_mean": "orig_h",
    "original_area_mean": "orig_area",
    "aspect_ratio_mean": "aspect_ratio",
}

# Patient CSV stores brightness, contrast, and saturation on 0-1.
# Inference computes them on the 0-255 OpenCV scale.
PATIENT_MEAN_SCALE = {
    "brightness": 255.0,
    "contrast": 255.0,
    "saturation": 255.0,
}

DEFAULT_CSV = SCRIPT_DIR / "palm_all_frame_quality_features.csv"
DEFAULT_OUT = SCRIPT_DIR / "palm_quality_train_stats.json"


def train_rows(csv_path: Path) -> tuple[list[dict], str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "split" not in fields:
            raise ValueError(f"{csv_path} has no split column")
        if "brightness_mean" in fields:
            column_map = PATIENT_MEAN_MAP
            kind = "patient_mean"
        else:
            column_map = COLUMN_MAP
            kind = "frame"
        missing = [src for src in column_map if src not in fields]
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        rows = []
        for raw in reader:
            if raw["split"].strip().lower() != "train":
                continue
            row = {}
            for src, dst in column_map.items():
                value = float(raw[src])
                if kind == "patient_mean":
                    value *= PATIENT_MEAN_SCALE.get(dst, 1.0)
                row[dst] = value
            rows.append(row)
    if not rows:
        raise ValueError(f"no train rows in {csv_path}")
    return rows, kind


def build_stats(csv_path: Path) -> dict:
    rows, kind = train_rows(csv_path)
    summary = summarize_train(rows, PICTURE_ATTRS)
    summary["split"] = {
        "name": "train",
        "source_csv": csv_path.name,
        "rows": kind,
    }
    if kind == "patient_mean":
        summary["split"]["unit_scale"] = (
            "brightness, contrast, and saturation multiplied by 255"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summary = build_stats(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"wrote {args.out}")
    print(f"train rows {summary['n']} ({summary['split'].get('rows', 'frame')})")
    for name, stats in summary["attributes"].items():
        print(f"  {name:16s} mean {stats['mean']:.4f}  std {stats['std']:.4f}")


if __name__ == "__main__":
    main()

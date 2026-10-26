"""
ID wrapper for the synthetic nail and palm models.

Anemic nail crops come from data/nail_dataset/nail_anemia_synthetic when present.
Palm frames are the current video_frame_selection test, validation, then Normal dump.

Example:

    python orchestrator/run_ids.py --ids ID723,ID086 --output_csv outputs/fusion_ids.csv
    python orchestrator/run_ids.py --ids-file outputs/fusion_overlap_ids.csv --output_csv outputs/overlap26.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ORCH_DIR = Path(__file__).resolve().parent
RUN_PY = ORCH_DIR / "run.py"
DEFAULT_CONFIG = ORCH_DIR / "config.json"

NAIL_SYNTH_ROOT = REPO_ROOT / "data" / "nail_dataset" / "nail_anemia_synthetic"
NAIL_CROP_ROOT = REPO_ROOT / "data" / "nail_dataset" / "unas_cropped"
PALM_VFS = REPO_ROOT / "data" / "palm_dataset" / "video_frame_selection"
# Test, then validation, then the Normal dump. The dump also contains eval IDs,
# so it has to be last. Validation frames are nested validation/validation/frames.
PALM_SEARCH_ROOTS = [
    PALM_VFS / "test" / "frames",
    PALM_VFS / "validation" / "validation" / "frames",
    PALM_VFS / "validation" / "frames",
    REPO_ROOT / "data" / "palm_dataset" / "test" / "frames",
    PALM_VFS / "Normal" / "Normal",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANEMIA_FOLDERS = {"leve", "moderada", "moderado"}
CSV_FIELDS = [
    "id",
    "nail_p",
    "nail_q",
    "nail_sq",
    "palm_p",
    "palm_q",
    "palm_sq",
    "S",
    "prediction",
    "isUncertain",
    "true_label",
]

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from fusion.fuse import fuse_packets  # noqa: E402


def empty_row(patient_id: str) -> dict:
    row = {k: "" for k in CSV_FIELDS}
    row["id"] = patient_id
    return row


def folder_to_binary(name: str | None) -> int | None:
    if not name:
        return None
    key = name.strip().lower()
    if key == "normal":
        return 0
    if key in ANEMIA_FOLDERS:
        return 1
    return None


def find_nail_in_root(root: Path, patient_id: str) -> tuple[Path | None, str | None]:
    if not root.is_dir():
        return None, None
    matches: list[Path] = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ext in IMAGE_EXTS:
            candidate = cls_dir / f"{patient_id}{ext}"
            if candidate.is_file():
                matches.append(candidate)
    if not matches:
        return None, None
    path = sorted(matches)[0]
    return path.resolve(), path.parent.name


def find_nail_path(patient_id: str) -> tuple[Path | None, str | None]:
    """Boosted anemic crop when this wrapper has one; otherwise the original crop."""
    if NAIL_SYNTH_ROOT is not None:
        synth_path, synth_class = find_nail_in_root(NAIL_SYNTH_ROOT, patient_id)
        if synth_path is not None:
            return synth_path, synth_class
    return find_nail_in_root(NAIL_CROP_ROOT, patient_id)


def find_palm_dir(patient_id: str) -> tuple[Path | None, str | None]:
    hits: list[Path] = []
    for root in PALM_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for cls_dir in root.iterdir():
            if not cls_dir.is_dir():
                continue
            patient_dir = cls_dir / patient_id
            if patient_dir.is_dir() and any(
                f.is_file() and f.suffix.lower() in IMAGE_EXTS
                for f in patient_dir.iterdir()
            ):
                hits.append(patient_dir)
        direct = root / patient_id
        if direct.is_dir() and any(
            f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in direct.iterdir()
        ):
            hits.append(direct)
    if not hits:
        return None, None
    path = hits[0].resolve()
    parent = path.parent.name
    label_folder = parent if folder_to_binary(parent) is not None else None
    return path, label_folder


def parse_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.ids:
        raw = args.ids.replace(",", " ").split()
        ids.extend(x.strip() for x in raw if x.strip())
    if args.ids_file:
        path = Path(args.ids_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ids file not found: {path}")
        if path.suffix.lower() == ".csv":
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                col = args.id_column
                if not reader.fieldnames or col not in reader.fieldnames:
                    raise ValueError(
                        f"CSV must have column '{col}'. Found: {reader.fieldnames}"
                    )
                for row in reader:
                    value = str(row[col]).strip()
                    if value:
                        ids.append(value)
        else:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    value = line.strip()
                    if value and not value.startswith("#"):
                        ids.append(value)
    if not ids:
        raise ValueError("provide --ids and/or --ids-file")
    seen = set()
    ordered = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def true_label(nail_class: str | None, palm_class: str | None) -> str:
    nail_bin = folder_to_binary(nail_class)
    palm_bin = folder_to_binary(palm_class)
    if nail_bin is not None:
        return str(nail_bin)
    if palm_bin is not None:
        return str(palm_bin)
    return ""


def fill_from_fusion(row: dict, packets: list[dict], k: float, x: float) -> None:
    result = fuse_packets(packets, k=k, uncertain_band=x)
    row["S"] = f"{result['S']:.10g}"
    row["prediction"] = result["label"]
    row["isUncertain"] = "1" if result["low_confidence"] else "0"
    for mod in result["modalities"]:
        name = mod["modality_name"]
        if name == "nail":
            row["nail_p"] = f"{mod['p']:.10g}"
            row["nail_q"] = f"{mod['q']:.10g}"
            row["nail_sq"] = f"{mod['contribution_sq']:.10g}"
        elif name == "palm":
            row["palm_p"] = f"{mod['p']:.10g}"
            row["palm_q"] = f"{mod['q']:.10g}"
            row["palm_sq"] = f"{mod['contribution_sq']:.10g}"


def run_one_id(
    patient_id: str,
    config_path: Path,
    python_exe: str,
    k: float,
    x: float,
    work_dir: Path,
) -> dict:
    row = empty_row(patient_id)
    nail_path, nail_class = find_nail_path(patient_id)
    palm_dir, palm_class = find_palm_dir(patient_id)
    row["true_label"] = true_label(nail_class, palm_class)

    cmd = [
        python_exe,
        str(RUN_PY),
        "--config",
        str(config_path),
        "--k",
        str(k),
        "--x",
        str(x),
    ]
    if nail_path is not None:
        cmd.extend(["--nail", str(nail_path)])
    if palm_dir is not None:
        cmd.extend(["--palm", str(palm_dir)])

    if nail_path is None and palm_dir is None:
        print(f"[{patient_id}] no nail crop or palm frames; writing NaNs", flush=True)
        return row

    packets_path = work_dir / f"{patient_id}_packets.json"
    cmd.extend(["--packets-out", str(packets_path)])
    print(
        f"[{patient_id}] nail={'yes' if nail_path else 'no'}  "
        f"palm={'yes' if palm_dir else 'no'}",
        flush=True,
    )
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0 or not packets_path.is_file():
        print(f"[{patient_id}] run.py failed (code={proc.returncode}); NaNs", flush=True)
        return row

    with open(packets_path, encoding="utf-8") as f:
        packets = json.load(f)
    fill_from_fusion(row, packets, k, x)
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run orchestrator/run.py for a list of IDs and write a CSV"
    )
    parser.add_argument("--ids", default=None, help="Comma or space separated IDs")
    parser.add_argument("--ids-file", default=None, help="Text (one ID per line) or CSV")
    parser.add_argument("--id-column", default="ID", help="CSV column for IDs")
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Orchestrator config JSON",
    )
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument("--x", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ids = parse_ids(args)
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = (REPO_ROOT / config_path).resolve()
        else:
            config_path = config_path.resolve()
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        fusion = config.get("fusion", {})
        k = float(args.k if args.k is not None else fusion.get("k", 1.0))
        x = float(args.x if args.x is not None else fusion.get("x", 0.2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.output_csv).expanduser()
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = out_path.parent / "_id_packets"
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for patient_id in ids:
        rows.append(
            run_one_id(
                patient_id,
                config_path,
                sys.executable,
                k,
                x,
                work_dir,
            )
        )

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}  n={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

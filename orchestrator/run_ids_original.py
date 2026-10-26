"""
ID wrapper for the original (non-synthetic) nail and palm models.

Nail crops come from data/nail_dataset/unas_cropped.
Anemic palm frames come from data/palm_dataset/non_synthetic.
Normal palm frames come from the real Normal folders under video_frame_selection.

Example:

    python orchestrator/run_ids_original.py --ids ID723,ID086 --output_csv outputs/fusion_original.csv
"""

from pathlib import Path

import run_ids

REPO_ROOT = Path(__file__).resolve().parent.parent
PALM_VFS = REPO_ROOT / "data" / "palm_dataset" / "video_frame_selection"
PALM_ORIGINAL = REPO_ROOT / "data" / "palm_dataset" / "non_synthetic"

run_ids.DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_original.json"
run_ids.NAIL_SYNTH_ROOT = None
run_ids.PALM_SEARCH_ROOTS = [
    PALM_ORIGINAL / "Leve" / "Leve",
    PALM_ORIGINAL / "Moderado" / "Moderado",
    PALM_VFS / "test" / "frames" / "Normal",
    PALM_VFS / "validation" / "validation" / "frames" / "Normal",
    PALM_VFS / "validation" / "frames" / "Normal",
    PALM_VFS / "Normal" / "Normal",
]


if __name__ == "__main__":
    raise SystemExit(run_ids.main())

"""
Offline nail inference: quality score, then frozen ResNet-50, then fusion packets.

Example:

    python nail_inference/infer.py --input path/to/crop.png
    python nail_inference/infer.py --input path/to/folder --json-out packets.json
    python nail_inference/infer.py --input path/to/folder --no-quality
    python nail_inference/infer.py --input path/to/folder --threshold 0.50 --q-method mean
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
NAIL_BRANCH = REPO_ROOT / "src" / "nail_branch"
if str(NAIL_BRANCH) not in sys.path:
    sys.path.insert(0, str(NAIL_BRANCH))

from image_quality import method_d_q_values, stats_for_path  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESIZE = 224
DEFAULT_CKPT = SCRIPT_DIR / "resnet50_binary_crop_frozen_best.pt"
DEFAULT_STATS = SCRIPT_DIR / "nail_quality_train_stats.json"


def collect_image_paths(input_path: Path, recursive: bool) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"unsupported image type: {input_path}")
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise ValueError(f"input is not a file or folder: {input_path}")
    pattern = "**/*" if recursive else "*"
    paths = [
        p.resolve()
        for p in sorted(input_path.glob(pattern))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not paths:
        raise ValueError(f"no images found in {input_path}")
    return paths


def load_stats(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def quality_for_path(path: Path, train_stats: dict, q_method: str) -> dict:
    row = stats_for_path(str(path))
    q_parts = method_d_q_values(row, train_stats)
    key = {"min": "q_min", "prod": "q_prod", "mean": "q_mean"}[q_method]
    return {
        "image_id": row["image_id"],
        "filename": row["filename"],
        "path": str(path),
        "q": float(q_parts[key]),
        "q_min": float(q_parts["q_min"]),
        "q_prod": float(q_parts["q_prod"]),
        "q_mean": float(q_parts["q_mean"]),
    }


def build_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((RESIZE, RESIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@torch.inference_mode()
def infer_probs(
    model: nn.Module,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> list[float]:
    tf = eval_transform()
    probs: list[float] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        tensors = []
        for p in batch_paths:
            with Image.open(p) as im:
                tensors.append(tf(im.convert("RGB")))
        x = torch.stack(tensors, dim=0).to(device)
        logits = model(x)
        p_anemia = torch.softmax(logits, dim=1)[:, 1]
        probs.extend(p_anemia.cpu().tolist())
    return probs


def make_packet(path: Path, p: float, q: float, threshold: float) -> dict:
    pred = int(p >= threshold)
    return {
        "modality_name": "nail",
        "available": True,
        "pred": pred,
        "p": float(p),
        "q": float(q),
        "t": float(threshold),
        "filename": path.name,
        "path": str(path),
    }


def format_packet(packet: dict) -> str:
    supports = "anemic" if packet["p"] > packet["t"] else "normal"
    return (
        f"  - {packet['filename']}\n"
        f"      path         : {packet['path']}\n"
        f"      modality     : {packet['modality_name']}\n"
        f"      available    : {packet['available']}\n"
        f"      p            : {packet['p']:.6f}\n"
        f"      t            : {packet['t']:.2f}\n"
        f"      pred         : {packet['pred']}  ({supports})\n"
        f"      q            : {packet['q']:.6f}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nail crop inference to fusion packets (input is a path, not an ID)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one image or a folder of images",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
        help="Path to resnet50_binary_crop_frozen_best.pt",
    )
    parser.add_argument(
        "--quality",
        dest="quality",
        action="store_true",
        help="Compute quality q from --stats-json (on by default for nail)",
    )
    parser.add_argument(
        "--no-quality",
        dest="quality",
        action="store_false",
        help="Skip quality; set q=1 for every image",
    )
    parser.set_defaults(quality=True)
    parser.add_argument(
        "--stats-json",
        default=str(DEFAULT_STATS),
        help="Train attribute stats JSON used only if quality is on",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="If set, write packets JSON to this path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Anemia threshold t (default: 0.50)",
    )
    parser.add_argument(
        "--q-method",
        choices=["mean", "min", "prod"],
        default="mean",
        help="Quality aggregate: q_mean (default), q_min, or q_prod",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If --input is a folder, include images in subfolders",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    stats_path = Path(args.stats_json).expanduser().resolve()

    if not 0.0 <= args.threshold <= 1.0:
        print("error: --threshold must be in [0, 1]", file=sys.stderr)
        return 2
    if not ckpt_path.is_file():
        print(f"error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1
    if args.quality and not stats_path.is_file():
        print(f"error: --quality is on but stats JSON not found: {stats_path}", file=sys.stderr)
        return 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("error: --device cuda requested but CUDA is unavailable", file=sys.stderr)
            return 1
        device = torch.device(args.device)

    try:
        paths = collect_image_paths(input_path, recursive=args.recursive)
        print(f"device: {device}")
        print(f"images: {len(paths)}")
        if args.quality:
            train_stats = load_stats(stats_path)
            print("computing quality...")
            qs = [
                float(quality_for_path(p, train_stats, args.q_method)["q"])
                for p in paths
            ]
        else:
            qs = [1.0] * len(paths)
            print("quality: off (q=1)")
        print("loading model...")
        model = build_model(ckpt_path, device)
        print("running inference...")
        probs = infer_probs(model, paths, device, args.batch_size)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    packets = [
        make_packet(path, p, q, args.threshold)
        for path, p, q in zip(paths, probs, qs)
    ]

    print()
    print("Nail packets")
    q_note = f"q_{args.q_method}" if args.quality else "q=1 (quality off)"
    print(f"  t={args.threshold:.2f}  {q_note}  n={len(packets)}")
    print()
    for packet in packets:
        print(format_packet(packet))
        print()

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "modality": "nail",
            "threshold": float(args.threshold),
            "quality_enabled": bool(args.quality),
            "q_method": f"q_{args.q_method}" if args.quality else None,
            "checkpoint": str(ckpt_path),
            "packets": packets,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

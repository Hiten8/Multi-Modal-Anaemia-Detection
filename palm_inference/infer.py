"""
Palm frame inference to fusion packets.

Input is an image file or a folder of frames (not patient IDs).
One packet per image. Quality q is optional (--quality / --no-quality).

Example:

    python palm_inference/infer.py --input data/palm_dataset/video_frame_selection/test/frames/Normal/ID007
    python palm_inference/infer.py --input path/to/frame.jpg --no-quality --json-out palm_packets.json
    python palm_inference/infer.py --input path/to/frames --recursive --quality --stats-json path/to/stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
NAIL_BRANCH = REPO_ROOT / "src" / "nail_branch"
if str(NAIL_BRANCH) not in sys.path:
    sys.path.insert(0, str(NAIL_BRANCH))

from image_quality import method_d_q_values, stats_for_path  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
DEFAULT_CKPT = SCRIPT_DIR / "palm_cnn_synthetic_finetuned_best.pth"
DEFAULT_STATS = SCRIPT_DIR / "palm_quality_train_stats.json"


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


def quality_for_path(path: Path, train_stats: dict, q_method: str) -> float:
    row = stats_for_path(str(path))
    q_parts = method_d_q_values(row, train_stats)
    key = {"min": "q_min", "prod": "q_prod", "mean": "q_mean"}[q_method]
    return float(q_parts[key])


def load_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)


def build_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    load_checkpoint(model, ckpt_path, device)
    model.to(device)
    model.eval()
    return model


class FrameDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            image = im.convert("RGB")
        return self.transform(image)


@torch.inference_mode()
def infer_probs(
    model: nn.Module,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> list[float]:
    tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    loader = DataLoader(
        FrameDataset(paths, tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    probs: list[float] = []
    for images in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs.extend(torch.sigmoid(logits).cpu().tolist())
    return probs


def make_packet(path: Path, p: float, q: float, threshold: float) -> dict:
    pred = int(p >= threshold)
    return {
        "modality_name": "palm",
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
        description="Palm frame inference to fusion packets (input is a path, not an ID)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one frame or a folder of frames",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
        help="Path to palm EfficientNet checkpoint (.pth)",
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
        "--quality",
        dest="quality",
        action="store_true",
        help="Compute quality q from --stats-json (off by default for palm)",
    )
    parser.add_argument(
        "--no-quality",
        dest="quality",
        action="store_false",
        help="Skip quality; set q=1 for every frame",
    )
    parser.set_defaults(quality=False)
    parser.add_argument(
        "--stats-json",
        default=str(DEFAULT_STATS),
        help="Train attribute stats JSON used only if --quality is set",
    )
    parser.add_argument(
        "--q-method",
        choices=["mean", "min", "prod"],
        default="mean",
        help="Quality aggregate when --quality is on (default: mean)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If --input is a folder, include images in subfolders",
    )
    parser.add_argument("--batch-size", type=int, default=32)
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
        print(
            f"error: --quality was set but stats JSON not found: {stats_path}\n"
            "Palm has no train-stats file yet. Use --no-quality, or pass "
            "--stats-json after you fit palm quality stats.",
            file=sys.stderr,
        )
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
        print(f"frames: {len(paths)}")
        qs: list[float]
        if args.quality:
            train_stats = load_stats(stats_path)
            print("computing quality...")
            qs = [quality_for_path(p, train_stats, args.q_method) for p in paths]
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
    print("Palm packets")
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
            "modality": "palm",
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

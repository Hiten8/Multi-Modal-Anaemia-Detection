"""
Orchestrator: config-driven modality inference, then late fusion.

Only modalities that get an input path on the command line are run.

Example:

    python orchestrator/run.py --config orchestrator/config.json ^
        --nail data/nail_dataset/unas_cropped/Normal/ID086.png ^
        --palm data/palm_dataset/video_frame_selection/test/frames/Normal/ID086

    python orchestrator/run.py --config orchestrator/config.json ^
        --nail path/to/crop.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "modalities" not in data:
        raise ValueError("config must be an object with a 'modalities' map")
    if not isinstance(data["modalities"], dict) or not data["modalities"]:
        raise ValueError("config.modalities must be a non-empty object")
    return data


def aggregate_packets(modality_name: str, packets: list[dict[str, Any]]) -> dict[str, Any]:
    """One fusion packet per modality (mean p/q if several frames/images)."""
    if not packets:
        raise ValueError(f"no packets returned for modality {modality_name}")
    t = float(packets[0]["t"])
    p_mean = sum(float(p["p"]) for p in packets) / len(packets)
    q_mean = sum(float(p["q"]) for p in packets) / len(packets)
    pred = int(p_mean >= t)
    out = {
        "modality_name": modality_name,
        "available": True,
        "pred": pred,
        "p": p_mean,
        "q": q_mean,
        "t": t,
        "n_inputs": len(packets),
        "filename": packets[0].get("filename"),
        "path": packets[0].get("path"),
    }
    if len(packets) > 1:
        out["paths"] = [p.get("path") for p in packets]
    return out


def run_modality_infer(
    name: str,
    spec: dict[str, Any],
    input_path: Path,
    json_out: Path,
    python_exe: str,
) -> list[dict[str, Any]]:
    script = resolve_repo_path(spec["infer_script"])
    ckpt = resolve_repo_path(spec["checkpoint"])
    if not script.is_file():
        raise FileNotFoundError(f"{name}: infer_script not found: {script}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"{name}: checkpoint not found: {ckpt}")
    if not input_path.exists():
        raise FileNotFoundError(f"{name}: input not found: {input_path}")

    quality = bool(spec.get("quality", False))
    cmd = [
        python_exe,
        str(script),
        "--input",
        str(input_path),
        "--checkpoint",
        str(ckpt),
        "--json-out",
        str(json_out),
        "--threshold",
        str(spec.get("threshold", 0.5)),
        "--q-method",
        str(spec.get("q_method", "mean")),
        "--batch-size",
        str(spec.get("batch_size", 16)),
        "--device",
        str(spec.get("device", "auto")),
        "--quality" if quality else "--no-quality",
    ]
    if spec.get("recursive"):
        cmd.append("--recursive")
    if quality:
        stats = spec.get("stats_json")
        if not stats:
            raise ValueError(f"{name}: quality is on but stats_json is missing")
        stats_path = resolve_repo_path(stats)
        if not stats_path.is_file():
            raise FileNotFoundError(f"{name}: stats_json not found: {stats_path}")
        cmd.extend(["--stats-json", str(stats_path)])

    print(f"[{name}] quality={'on' if quality else 'off'}", flush=True)
    print(f"[{name}] input={input_path}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"{name}: inference script exited with {proc.returncode}")
    with open(json_out, encoding="utf-8") as f:
        payload = json.load(f)
    packets = payload.get("packets", payload if isinstance(payload, list) else None)
    if not isinstance(packets, list):
        raise ValueError(f"{name}: inference JSON has no packets array")
    for packet in packets:
        packet["modality_name"] = name
        packet["available"] = True
    return packets


def run_fusion(
    fusion_spec: dict[str, Any],
    packets_path: Path,
    python_exe: str,
    k: float,
    x: float,
) -> int:
    script = resolve_repo_path(fusion_spec.get("script", "src/fusion/fuse.py"))
    if not script.is_file():
        raise FileNotFoundError(f"fusion script not found: {script}")
    cmd = [
        python_exe,
        str(script),
        "--json",
        str(packets_path),
        "--k",
        str(k),
        "--x",
        str(x),
    ]
    print(flush=True)
    print("Running fusion...", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


def build_parser(modality_names: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run configured modality inference scripts, then late fusion"
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "orchestrator" / "config.json"),
        help="Path to orchestrator config JSON",
    )
    for name in modality_names:
        parser.add_argument(
            f"--{name}",
            default=None,
            help=f"Image or folder path for modality '{name}' (omit to skip)",
        )
    parser.add_argument("--k", type=float, default=None, help="Override fusion k")
    parser.add_argument("--x", type=float, default=None, help="Override fusion |S| band")
    parser.add_argument(
        "--packets-out",
        default=None,
        help="Where to write the combined packet JSON (default: a temp file)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        default=str(REPO_ROOT / "orchestrator" / "config.json"),
    )
    pre_args, _ = pre.parse_known_args(argv)

    try:
        config_path = resolve_repo_path(pre_args.config)
        config = load_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    modality_names = list(config["modalities"].keys())
    parser = build_parser(modality_names)
    args = parser.parse_args(argv)

    selected: list[tuple[str, Path]] = []
    for name in modality_names:
        value = getattr(args, name.replace("-", "_"), None)
        if value:
            selected.append((name, Path(value).expanduser().resolve()))

    if not selected:
        flags = " ".join(f"--{n} PATH" for n in modality_names)
        print(
            "error: provide at least one modality input path "
            f"(config modalities: {', '.join(modality_names)}). Example: {flags}",
            file=sys.stderr,
        )
        return 2

    fusion_spec = config.get("fusion", {})
    k = float(args.k if args.k is not None else fusion_spec.get("k", 1.0))
    x = float(args.x if args.x is not None else fusion_spec.get("x", 0.2))

    python_exe = sys.executable
    combined: list[dict[str, Any]] = []

    try:
        with tempfile.TemporaryDirectory(prefix="fusion_orch_") as tmp:
            tmp_dir = Path(tmp)
            for name, input_path in selected:
                spec = config["modalities"][name]
                infer_json = tmp_dir / f"{name}_packets.json"
                raw_packets = run_modality_infer(
                    name, spec, input_path, infer_json, python_exe
                )
                packet = aggregate_packets(name, raw_packets)
                combined.append(packet)
                print(
                    f"[{name}] packet  p={packet['p']:.4f}  q={packet['q']:.4f}  "
                    f"t={packet['t']:.2f}  n_inputs={packet['n_inputs']}  "
                    f"pred={packet['pred']}",
                    flush=True,
                )

            if args.packets_out:
                packets_path = Path(args.packets_out).expanduser().resolve()
                packets_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                packets_path = tmp_dir / "fusion_packets.json"

            with open(packets_path, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2)
            print(f"wrote combined packets: {packets_path}", flush=True)

            code = run_fusion(fusion_spec, packets_path, python_exe, k, x)
            return code
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

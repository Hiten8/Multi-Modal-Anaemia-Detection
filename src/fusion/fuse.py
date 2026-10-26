"""
Late fusion from a modality-packet JSON.

    python src/fusion/fuse.py --json examples/fusion_packets.json
    python src/fusion/fuse.py --json examples/fusion_packets.json --k 1 --x 0.2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any


LOGIT_CLIP = 1e-6
Q_SUM_EPS = 1e-12


def logit(x: float, clip: float = LOGIT_CLIP) -> float:
    x = min(max(float(x), clip), 1.0 - clip)
    return math.log(x / (1.0 - x))


def signed_score(p: float, t: float, k: float, clip: float = LOGIT_CLIP) -> float:
    """s = tanh(k * (logit(p) - logit(t)))."""
    return math.tanh(float(k) * (logit(p, clip) - logit(t, clip)))


def as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{field} must be boolean (got {value!r})")


def as_unit_interval(value: Any, field: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number in [0, 1] (got {value!r})") from exc
    if not 0.0 <= x <= 1.0 or math.isnan(x):
        raise ValueError(f"{field} must be in [0, 1] (got {x})")
    return x


def load_packets(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "packets" in data:
        data = data["packets"]
    if not isinstance(data, list):
        raise ValueError(
            "JSON must be an array of packets, or an object with key 'packets'"
        )
    return data


def parse_packet(raw: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"packet[{index}] must be an object")

    name = raw.get("modality_name", raw.get("modality_id"))
    if not name or not isinstance(name, str):
        raise ValueError(f"packet[{index}] needs string field modality_name")

    available = as_bool(raw.get("available"), f"packet[{index}].available")
    packet = {
        "modality_name": name,
        "available": available,
        "pred": raw.get("pred"),
        "p": raw.get("p"),
        "q": raw.get("q"),
        "t": raw.get("t"),
    }
    if not available:
        return packet

    packet["p"] = as_unit_interval(raw.get("p"), f"packet[{index}].p")
    packet["q"] = as_unit_interval(raw.get("q"), f"packet[{index}].q")
    packet["t"] = as_unit_interval(raw.get("t"), f"packet[{index}].t")
    if raw.get("pred") is not None:
        packet["pred"] = int(as_bool(raw.get("pred"), f"packet[{index}].pred"))
    return packet


def fuse_packets(
    packets: list[dict[str, Any]],
    k: float = 1.0,
    logit_clip: float = LOGIT_CLIP,
    uncertain_band: float = 0.2,
) -> dict[str, Any]:
    available = [p for p in packets if p["available"]]
    if not available:
        raise ValueError("no available modalities (all available=false or empty list)")

    rows = []
    num = 0.0
    den = 0.0
    for p in available:
        s = signed_score(p["p"], p["t"], k, logit_clip)
        contrib = s * p["q"]
        supports = "anemic" if p["p"] > p["t"] else "normal"
        rows.append(
            {
                "modality_name": p["modality_name"],
                "p": p["p"],
                "t": p["t"],
                "q": p["q"],
                "s": s,
                "supports": supports,
                "contribution_sq": contrib,
            }
        )
        num += contrib
        den += p["q"]

    if den <= Q_SUM_EPS:
        raise ValueError(
            "sum of q over available modalities is 0; cannot fuse "
            "(all quality scores are zero)"
        )

    s_fused = num / den
    conflict = len({row["supports"] for row in rows}) > 1
    low_confidence = abs(s_fused) <= uncertain_band
    if low_confidence:
        label = "uncertain"
        pred = None
    else:
        label = "normal" if s_fused <= 0.0 else "anemic"
        pred = 0 if label == "normal" else 1

    return {
        "k": float(k),
        "n_available": len(available),
        "sum_q": den,
        "S": s_fused,
        "label": label,
        "pred": pred,
        "low_confidence": low_confidence,
        "uncertain_band": float(uncertain_band),
        "conflict": conflict,
        "modalities": rows,
        "skipped": [p["modality_name"] for p in packets if not p["available"]],
    }


def format_report(result: dict[str, Any]) -> str:
    flag = "yes  (|S| <= x)" if result["low_confidence"] else "no"
    lines = [
        "Fusion result",
        f"  k                 : {result['k']}",
        f"  x (uncertain band): {result['uncertain_band']}",
        f"  S                 : {result['S']:.6f}",
        f"  |S| <= x          : {flag}",
        f"  prediction        : {result['label']}  (pred={result['pred']})",
        f"  conflict          : {result['conflict']}",
        f"  available count   : {result['n_available']}",
    ]
    skipped = result.get("skipped") or []
    if skipped:
        lines.append(f"  skipped           : {', '.join(skipped)}")
    lines.append("")
    lines.append("Per available modality")
    for row in result["modalities"]:
        lines.extend(
            [
                f"  - {row['modality_name']}",
                f"      supports                  : {row['supports']}",
                f"      p                         : {row['p']:.6f}",
                f"      q                         : {row['q']:.6f}",
                f"      t                         : {row['t']:.6f}",
                f"      s                         : {row['s']:.6f}",
                f"      s*q                       : {row['contribution_sq']:.6f}",
            ]
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse modality packets (JSON) into a normal/anemic decision"
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to JSON: array of packets, or {\"packets\": [...]}",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=1.0,
        help="Tanh steepness for s = tanh(k * (logit(p)-logit(t))) (default: 1)",
    )
    parser.add_argument(
        "--x",
        type=float,
        default=0.2,
        dest="uncertain_band",
        help="Flag uncertain if |S| <= x (default: 0.2)",
    )
    parser.add_argument(
        "--logit-clip",
        type=float,
        default=LOGIT_CLIP,
        help=f"Clip p and t into (clip, 1-clip) before logit (default: {LOGIT_CLIP})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.k <= 0:
        print("error: --k must be > 0", file=sys.stderr)
        return 2
    if args.uncertain_band < 0:
        print("error: --x must be >= 0", file=sys.stderr)
        return 2
    try:
        raw_list = load_packets(args.json)
        packets = [parse_packet(item, i) for i, item in enumerate(raw_list)]
        result = fuse_packets(
            packets,
            k=args.k,
            logit_clip=args.logit_clip,
            uncertain_band=args.uncertain_band,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

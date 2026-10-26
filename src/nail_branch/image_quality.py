"""
Picture-quality stats for nail crops (fusion reliability).

Pixel stats are computed on the RGB image **after** resize to 224x224
(what ResNet sees, before ImageNet normalize). Original width/height
are taken from the file **before** resize.
"""

from __future__ import annotations

import os
import re

import cv2
import numpy as np

RESIZE = 224
HIGHLIGHT_V = 250
SHADOW_Y = 10


def parse_image_id(path: str) -> str:
    """Filename stem, e.g. ID176.png -> ID176."""
    return os.path.splitext(os.path.basename(path))[0]


def parse_numeric_id(image_id: str) -> int | None:
    m = re.search(r"(\d+)", image_id)
    return int(m.group(1)) if m else None


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb(rgb: np.ndarray, size: int = RESIZE) -> np.ndarray:
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def luma(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    return 0.299 * r + 0.587 * g + 0.114 * b


def picture_stats_after_resize(rgb224: np.ndarray) -> dict:
    """rgb224: uint8 HxWx3 after resize (typically 224x224)."""
    if rgb224.dtype != np.uint8:
        rgb224 = np.clip(rgb224, 0, 255).astype(np.uint8)

    y = luma(rgb224)
    hsv = cv2.cvtColor(rgb224, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float64)
    v = hsv[:, :, 2].astype(np.float64)

    gray = cv2.cvtColor(rgb224, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    n_pix = rgb224.shape[0] * rgb224.shape[1]
    highlight = float(np.mean(v > HIGHLIGHT_V))
    shadow = float(np.mean(y < SHADOW_Y))

    return {
        "brightness": float(np.mean(y)),
        "contrast": float(np.std(y)),
        "saturation": float(np.mean(s)),
        "highlight_frac": highlight,
        "shadow_frac": shadow,
        "sharpness": float(np.var(lap)),
        "n_pixels": int(n_pix),
    }


def original_geometry(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    area = int(w * h)
    return {
        "orig_w": int(w),
        "orig_h": int(h),
        "orig_area": area,
        "aspect_ratio": float(w / h) if h else 0.0,
    }


def stats_for_path(path: str, size: int = RESIZE) -> dict:
    rgb = load_rgb(path)
    geo = original_geometry(rgb)
    rgb224 = resize_rgb(rgb, size)
    pix = picture_stats_after_resize(rgb224)
    return {
        "image_id": parse_image_id(path),
        "filename": os.path.basename(path),
        **geo,
        **pix,
        "resize": int(size),
    }


PICTURE_ATTRS = [
    "brightness",
    "contrast",
    "saturation",
    "highlight_frac",
    "shadow_frac",
    "sharpness",
    "orig_w",
    "orig_h",
    "orig_area",
    "aspect_ratio",
]

# How each attribute maps z-score -> badness b (before exp(-b^2/2)).
# abs: |z|; low_bad: max(0, -z); high_bad: max(0, z)
BADNESS_MODE = {
    "brightness": "abs",
    "contrast": "low_bad",
    "saturation": "low_bad",
    "highlight_frac": "high_bad",
    "shadow_frac": "high_bad",
    "sharpness": "low_bad",
    "orig_w": "low_bad",
    "orig_h": "low_bad",
    "orig_area": "low_bad",
    "aspect_ratio": "abs",
}

EPS = 1e-8


def z_score(value: float, mean: float, std: float) -> float:
    return float((value - mean) / (std + EPS))


def badness_from_z(z: float, mode: str) -> float:
    if mode == "abs":
        return abs(z)
    if mode == "low_bad":
        return max(0.0, -z)
    if mode == "high_bad":
        return max(0.0, z)
    raise ValueError(f"unknown badness mode: {mode}")


def attribute_score(b: float) -> float:
    """Method D per-attribute score in (0, 1]."""
    return float(np.exp(-(b ** 2) / 2.0))


def attribute_scores_from_row(
    row: dict,
    train_stats: dict,
    attrs: list[str] | None = None,
) -> dict[str, float]:
    """Return s_i and b_i for each picture attribute."""
    attrs = attrs or PICTURE_ATTRS
    stats = train_stats["attributes"]
    out = {}
    for name in attrs:
        mu = stats[name]["mean"]
        sigma = stats[name]["std"]
        z = z_score(float(row[name]), mu, sigma)
        mode = BADNESS_MODE[name]
        b = badness_from_z(z, mode)
        out[f"b_{name}"] = b
        out[f"s_{name}"] = attribute_score(b)
    return out


def aggregate_q(scores: list[float], method: str) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    if method == "min":
        return float(np.min(arr))
    if method == "prod":
        return float(np.prod(arr))
    if method == "mean":
        return float(np.mean(arr))
    raise ValueError(f"unknown q method: {method}")


def method_d_q_values(
    row: dict,
    train_stats: dict,
    attrs: list[str] | None = None,
) -> dict[str, float]:
    parts = attribute_scores_from_row(row, train_stats, attrs)
    s_list = [parts[f"s_{name}"] for name in (attrs or PICTURE_ATTRS)]
    return {
        "q_min": aggregate_q(s_list, "min"),
        "q_prod": aggregate_q(s_list, "prod"),
        "q_mean": aggregate_q(s_list, "mean"),
        **parts,
    }


def summarize_train(rows: list[dict], attrs: list[str] | None = None) -> dict:
    """p5, p50, p95, mean, std, n per attribute (train only)."""
    attrs = attrs or PICTURE_ATTRS
    n = len(rows)
    out = {"n": n, "resize": RESIZE, "attributes": {}}
    for name in attrs:
        vals = np.array([float(r[name]) for r in rows], dtype=np.float64)
        p5, p50, p95 = np.percentile(vals, [5, 50, 95])
        out["attributes"][name] = {
            "p05": float(p5),
            "p50": float(p50),
            "p95": float(p95),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
            "n": int(n),
        }
    return out


def entropy_binary(p: float, eps: float = 1e-12) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(-p * np.log(p) - (1.0 - p) * np.log(1.0 - p))


def thresh_tag(t: float) -> str:
    """0.45 -> 045, 0.50 -> 050 (str(0.50) is '0.5', which would break CSV headers)."""
    return f"{float(t):.2f}".replace(".", "")


def model_side(p: float, thresholds=(0.45, 0.50, 0.60)) -> dict:
    p = float(p)
    d = {
        "p": p,
        "entropy": entropy_binary(p),
    }
    for t in thresholds:
        tag = thresh_tag(t)
        d[f"dist_t{tag}"] = abs(p - t)
        d[f"pred_t{tag}"] = int(p >= t)
    return d

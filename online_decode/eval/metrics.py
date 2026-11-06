"""客观指标：只用来发现技术缺陷，不用来对齐风格。

刻意不做的事：
  * 不算"与桶均值的距离"——本项目不要求输出一致；
  * 不给审美分数——那会诱导把"像不像样本"当成好坏。

指标存在的意义是给评价提供**可核验的锚点**：模型说"高光有溢出"，必须能对上
`clip_high_ratio` 的数值，否则这句话就是编的。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CLIP_HIGH_LEVEL = 250
CLIP_LOW_LEVEL = 5
ANALYSIS_LONG_EDGE = 512


def _load(path: str | Path, long_edge: int = ANALYSIS_LONG_EDGE) -> tuple[np.ndarray, int, int]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if max(width, height) > long_edge:
            ratio = long_edge / max(width, height)
            image = image.resize((max(1, int(width * ratio)), max(1, int(height * ratio))), Image.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0, width, height


def compute(path: str | Path, baseline_path: str | Path | None = None) -> dict[str, Any]:
    rgb, width, height = _load(path)
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    metrics: dict[str, Any] = {
        "path": str(path),
        "width": width,
        "height": height,
        "luma": {
            "p5": round(float(np.percentile(luma, 5)), 4),
            "p50": round(float(np.percentile(luma, 50)), 4),
            "p95": round(float(np.percentile(luma, 95)), 4),
            "mean": round(float(luma.mean()), 4),
        },
        "clip_high_ratio": round(float((luma >= CLIP_HIGH_LEVEL / 255).mean()), 5),
        "clip_low_ratio": round(float((luma <= CLIP_LOW_LEVEL / 255).mean()), 5),
        "channel_means": {
            "r": round(float(rgb[:, :, 0].mean()), 4),
            "g": round(float(rgb[:, :, 1].mean()), 4),
            "b": round(float(rgb[:, :, 2].mean()), 4),
        },
        "saturation": {},
        "sharpness": {},
    }

    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)
    metrics["saturation"] = {
        "p50": round(float(np.percentile(saturation, 50)), 4),
        "p95": round(float(np.percentile(saturation, 95)), 4),
    }

    laplacian = (
        -4 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:]
    )
    metrics["sharpness"] = {
        "laplacian_var": round(float(laplacian.var() * 1e4), 3),
    }

    if baseline_path:
        try:
            base, _, _ = _load(baseline_path)
            if base.shape == rgb.shape:
                base_luma = 0.2126 * base[:, :, 0] + 0.7152 * base[:, :, 1] + 0.0722 * base[:, :, 2]
                a = luma.ravel() - luma.mean()
                b = base_luma.ravel() - base_luma.mean()
                denom = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1e-9
                metrics["vs_baseline"] = {
                    "luma_corr": round(float((a * b).sum() / denom), 4),
                    "delta_mean": round(float(luma.mean() - base_luma.mean()), 4),
                }
        except Exception as exc:  # 基线缺失不该阻断本轮
            metrics["vs_baseline"] = {"error": str(exc)}

    return metrics


def flatten(metrics: dict[str, Any]) -> dict[str, float]:
    """把嵌套指标摊平成 `a.b.c -> 数值`，供 evidence 引用做存在性校验。"""
    out: dict[str, float] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else str(key), value)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            out[prefix] = float(node)

    walk("", metrics)
    return out


def lookup(flat: dict[str, float], ref: str) -> float | None:
    ref = ref.strip()
    if ref.startswith("metrics."):
        ref = ref[len("metrics."):]
    if ref in flat:
        return flat[ref]
    # 允许 "clip_high_ratio=0.004" 这种带等号的写法
    match = re.match(r"^([\w.]+)\s*=", ref)
    if match and match.group(1) in flat:
        return flat[match.group(1)]
    return None

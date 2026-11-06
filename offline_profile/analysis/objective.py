"""由客观特征派生仅有的两个硬约束：亮度分级与特征摘要。"""

from __future__ import annotations

from typing import Any


def objective_brightness(feature: dict, cfg: dict) -> str:
    """按亮度 P50 映射 bright/mid/dark；AI 不得无理由改写。"""
    try:
        p50 = float(feature["luminance"]["percentiles"]["p50"])
    except (KeyError, TypeError, ValueError):
        return "mid"
    if p50 < float(cfg["brightness_dark_max_p50"]):
        return "dark"
    if p50 >= float(cfg["brightness_bright_min_p50"]):
        return "bright"
    return "mid"


def feature_digest(feature: dict) -> dict[str, Any]:
    """给模型看的紧凑特征摘要；完整特征仍由调用方随请求传入。"""
    def pick(path: tuple[str, ...], default=None):
        node: Any = feature
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    return {
        "luminance": {
            "mean": pick(("luminance", "mean")),
            "std": pick(("luminance", "std")),
            "p5": pick(("luminance", "percentiles", "p5")),
            "p50": pick(("luminance", "percentiles", "p50")),
            "p95": pick(("luminance", "percentiles", "p95")),
            "near_black_ratio": pick(("luminance", "near_black_ratio")),
            "near_white_ratio": pick(("luminance", "near_white_ratio")),
        },
        "lab": {
            "L_mean": pick(("lab", "L", "mean")),
            "a_mean": pick(("lab", "a", "mean")),
            "b_mean": pick(("lab", "b", "mean")),
            "chroma_mean": pick(("lab", "chroma", "mean")),
            "chroma_p90": pick(("lab", "chroma", "p90")),
        },
        "saturation_mean": pick(("saturation", "mean")),
        "sharpness": pick(("sharpness", "laplacian_variance")),
        "palette": [
            {"hex": item.get("hex"), "ratio": item.get("ratio")}
            for item in (feature.get("palette") or [])[:5]
        ],
        "geometry": {
            "width": pick(("geometry", "output_width")),
            "height": pick(("geometry", "output_height")),
        },
        "was_grayscale": pick(("source", "was_grayscale")),
    }

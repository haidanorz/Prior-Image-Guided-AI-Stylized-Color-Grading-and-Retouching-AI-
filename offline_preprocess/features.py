"""客观图像特征计算。

全部统计作用于「已校正方向、已转 sRGB、已缩放、尚未 JPEG 编码」的内存图像，
禁止从输出 JPEG 反读统计。

分组：luminance / rgb / lab / hue / saturation / palette / spatial / sharpness /
similarity / exif_summary。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .color import (
    histogram_normalized,
    percentile_values,
    rgb_to_lab,
    rgb_to_luminance,
    saturation_hsv,
)
from .palette import extract_palette
from .similarity import PHASH_ALGORITHM, phash64

LUMINANCE_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
CHANNEL_PERCENTILES = (5, 50, 95)
LAB_PERCENTILES = (5, 50, 95)
CHROMA_PERCENTILES = (50, 90)
SATURATION_PERCENTILES = (50, 90)

SHARPNESS_METHOD = "laplacian3x3_v1"


def compute_features(
    pixels: np.ndarray, cfg: dict[str, Any], exif_summary: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[str]]:
    """pixels: HxWx3 uint8（sRGB）。返回 (特征分组字典, 警告列表)。"""
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8, got {pixels.shape}")
    if pixels.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {pixels.dtype}")

    warnings: list[str] = []
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    near_black = int(cfg["near_black_threshold"])
    near_white = int(cfg["near_white_threshold"])

    luminance = rgb_to_luminance(pixels)
    lightness, lab_a, lab_b = rgb_to_lab(pixels)
    chroma = np.hypot(lab_a, lab_b)
    saturation = saturation_hsv(pixels)

    features: dict[str, Any] = {}
    features["luminance"] = _luminance_group(luminance, near_black, near_white, int(cfg["luminance_bins"]))
    features["rgb"] = _rgb_group(pixels, near_black, near_white)
    features["lab"] = _lab_group(lightness, lab_a, lab_b, chroma)
    features["hue"] = _hue_group(lab_a, lab_b, chroma, cfg)
    if features["hue"]["chromatic_pixel_ratio"] == 0.0:
        warnings.append("no_chromatic_pixels")
    features["saturation"] = _saturation_group(saturation)
    features["palette"] = _palette_group(lightness, lab_a, lab_b, cfg)
    features["spatial"] = _spatial_group(lightness, chroma)
    features["sharpness"] = _sharpness_group(luminance)
    features["similarity"] = {
        "phash": phash64(_luma_u8(luminance)),
        "algorithm": PHASH_ALGORITHM,
    }
    features["exif_summary"] = exif_summary

    # 记录网格尺寸，便于下游解释 3x3 单元的像素范围
    features["spatial"]["image_width"] = width
    features["spatial"]["image_height"] = height
    return features, warnings


def _luma_u8(luminance: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(luminance * 255.0 + 0.5), 0, 255).astype(np.uint8)


def _luminance_group(luminance: np.ndarray, near_black: int, near_white: int, bins: int) -> dict:
    flat = luminance.ravel()
    values = percentile_values(flat, LUMINANCE_PERCENTILES)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(ddof=0)),
        "percentiles": {
            f"p{p}": v for p, v in zip(LUMINANCE_PERCENTILES, values)
        },
        "histogram": {
            "bins": bins,
            "range": [0.0, 1.0],
            "counts": histogram_normalized(flat, bins, (0.0, 1.0)),
        },
        "near_black_ratio": float(np.count_nonzero(flat <= near_black / 255.0) / flat.size),
        "near_white_ratio": float(np.count_nonzero(flat >= near_white / 255.0) / flat.size),
        "thresholds_8bit": {"near_black": near_black, "near_white": near_white},
        "percentile_method": "linear",
    }


def _rgb_group(pixels: np.ndarray, near_black: int, near_white: int) -> dict:
    group: dict[str, Any] = {}
    for index, name in enumerate(("r", "g", "b")):
        channel_u8 = pixels[..., index]
        channel = channel_u8.astype(np.float64) / 255.0
        p5, p50, p95 = percentile_values(channel, CHANNEL_PERCENTILES)
        group[name] = {
            "mean": float(channel.mean()),
            "std": float(channel.std(ddof=0)),
            "p5": p5,
            "p50": p50,
            "p95": p95,
            "low_clip_ratio": float(np.count_nonzero(channel_u8 <= near_black) / channel_u8.size),
            "high_clip_ratio": float(np.count_nonzero(channel_u8 >= near_white) / channel_u8.size),
        }
    return group


def _lab_group(lightness: np.ndarray, a: np.ndarray, b: np.ndarray, chroma: np.ndarray) -> dict:
    group: dict[str, Any] = {}
    for name, values in (("L", lightness), ("a", a), ("b", b)):
        p5, p50, p95 = percentile_values(values, LAB_PERCENTILES)
        group[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "p5": p5,
            "p50": p50,
            "p95": p95,
        }
    p50, p90 = percentile_values(chroma, CHROMA_PERCENTILES)
    group["chroma"] = {
        "mean": float(chroma.mean()),
        "std": float(chroma.std(ddof=0)),
        "p50": p50,
        "p90": p90,
    }
    group["space"] = "CIE Lab D65"
    return group


def _hue_group(a: np.ndarray, b: np.ndarray, chroma: np.ndarray, cfg: dict) -> dict:
    bins = int(cfg["hue_bins"])
    min_chroma = float(cfg["hue_min_chroma"])
    mask = chroma >= min_chroma
    total = int(chroma.size)
    chromatic_ratio = float(np.count_nonzero(mask) / total) if total else 0.0

    histogram = [0.0] * bins
    if chromatic_ratio > 0.0:
        hue = np.degrees(np.arctan2(b[mask], a[mask])) % 360.0
        weights = chroma[mask]
        index = np.clip(np.floor(hue / (360.0 / bins)).astype(np.int64), 0, bins - 1)
        counts = np.bincount(index, weights=weights, minlength=bins).astype(np.float64)
        weight_sum = float(counts.sum())
        if weight_sum > 0:
            histogram = [float(c / weight_sum) for c in counts]

    return {
        "bins": bins,
        "range_degrees": [0.0, 360.0],
        "chroma_weighted": True,
        "min_chroma": min_chroma,
        "histogram": histogram,
        "chromatic_pixel_ratio": chromatic_ratio,
    }


def _saturation_group(saturation: np.ndarray) -> dict:
    p50, p90 = percentile_values(saturation, SATURATION_PERCENTILES)
    return {
        "mean": float(saturation.mean()),
        "std": float(saturation.std(ddof=0)),
        "p50": p50,
        "p90": p90,
        "space": "HSV",
    }


def _palette_group(lightness: np.ndarray, a: np.ndarray, b: np.ndarray, cfg: dict) -> list[dict]:
    lab_pixels = np.stack([lightness.ravel(), a.ravel(), b.ravel()], axis=-1)
    return extract_palette(
        lab_pixels,
        k=int(cfg["palette_colors"]),
        seed=int(cfg["random_seed"]),
        max_sample=int(cfg["palette_max_sample_pixels"]),
        max_iter=int(cfg["palette_max_iterations"]),
    )


def _spatial_group(lightness: np.ndarray, chroma: np.ndarray) -> dict:
    height, width = lightness.shape
    row_edges = _edges(height)
    col_edges = _edges(width)
    cells: list[dict] = []
    for row in range(3):
        for col in range(3):
            r0, r1 = row_edges[row], row_edges[row + 1]
            c0, c1 = col_edges[col], col_edges[col + 1]
            cell_l = lightness[r0:r1, c0:c1]
            cell_c = chroma[r0:r1, c0:c1]
            cells.append(
                {
                    "row": row,
                    "col": col,
                    "mean_l": float(cell_l.mean()) if cell_l.size else 0.0,
                    "mean_chroma": float(cell_c.mean()) if cell_c.size else 0.0,
                    "pixel_count": int(cell_l.size),
                }
            )
    return {"grid": [3, 3], "cells": cells}


def _edges(length: int) -> list[int]:
    """3 等分边界，整除时严格均分（不丢像素）。"""
    base = length // 3
    remainder = length % 3
    edges = [0]
    cursor = 0
    for index in range(3):
        cursor += base + (1 if index < remainder else 0)
        edges.append(cursor)
    return edges


def _sharpness_group(luminance: np.ndarray) -> dict:
    """3x3 拉普拉斯方差，亮度按 0..255 计，仅作质量参考。"""
    luma = luminance * 255.0
    padded = np.pad(luma, 1, mode="edge")
    laplacian = (
        4.0 * padded[1:-1, 1:-1]
        - padded[:-2, 1:-1]
        - padded[2:, 1:-1]
        - padded[1:-1, :-2]
        - padded[1:-1, 2:]
    )
    return {
        "laplacian_variance": float(laplacian.var(ddof=0)),
        "method": SHARPNESS_METHOD,
        "luminance_scale": "0-255",
    }

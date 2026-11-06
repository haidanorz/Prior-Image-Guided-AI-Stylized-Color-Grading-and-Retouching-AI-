"""尺寸计算与单次重采样。

规则（需求文档 PP-006）：
- scale = min(1, L / max(W, H))，只用像素尺寸，不参考 DPI / 文件大小。
- 长边直接定点为 L，短边按 floor(x + 0.5) 正数四舍五入，禁止截断与银行家舍入。
- 不放大、不裁剪、不补边、不拉伸。
- 每张图最多一次重采样，先算尺寸再统一执行。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

from .errors import ErrorCode, PreprocessError, Stage, WarningCode
from .version import ROUNDING_RULE_VERSION, RESAMPLE_FILTER


@dataclass(frozen=True)
class TargetGeometry:
    width: int
    height: int
    scale: float
    resampled: bool
    warning: str | None


def compute_target_size(width: int, height: int, long_edge: int) -> TargetGeometry:
    if width < 1 or height < 1:
        raise PreprocessError(ErrorCode.RESIZE_FAILED, Stage.RESIZE, f"非法输入尺寸：{width}x{height}")
    longest = max(width, height)
    if longest <= long_edge:
        return TargetGeometry(width, height, 1.0, False, None)

    warning: str | None = None
    if width >= height:
        new_w = long_edge
        new_h = int(math.floor(height * long_edge / width + 0.5))
    else:
        new_h = long_edge
        new_w = int(math.floor(width * long_edge / height + 0.5))

    if new_w < 1:
        new_w = 1
        warning = WarningCode.EXTREME_RESIZE_ROUNDING
    if new_h < 1:
        new_h = 1
        warning = WarningCode.EXTREME_RESIZE_ROUNDING

    scale = min(1.0, long_edge / float(longest))
    return TargetGeometry(new_w, new_h, scale, True, warning)


def resize_once(image: Image.Image, target: TargetGeometry) -> Image.Image:
    """按目标尺寸执行一次 Lanczos 重采样；scale=1 时原样返回。"""
    if not target.resampled:
        return image
    try:
        resampling = Image.Resampling.LANCZOS
    except AttributeError:  # Pillow < 9.1
        resampling = Image.LANCZOS
    try:
        return image.resize((target.width, target.height), resampling)
    except Exception as exc:
        raise PreprocessError(
            ErrorCode.RESIZE_FAILED,
            Stage.RESIZE,
            f"重采样失败：{image.size} -> {target.width}x{target.height}：{exc}",
        ) from exc


def geometry_record(
    decoded: tuple[int, int],
    oriented: tuple[int, int],
    target: TargetGeometry,
    long_edge: int,
    orientation: int,
    orientation_applied: bool,
) -> dict:
    return {
        "decoded_width": int(decoded[0]),
        "decoded_height": int(decoded[1]),
        "oriented_width": int(oriented[0]),
        "oriented_height": int(oriented[1]),
        "output_width": int(target.width),
        "output_height": int(target.height),
        "aspect_ratio": round(target.width / target.height, 6),
        "scale": round(target.scale, 6),
        "target_long_edge": int(long_edge),
        "resampled": bool(target.resampled),
        "resample_filter": RESAMPLE_FILTER if target.resampled else None,
        "rounding_rule": ROUNDING_RULE_VERSION,
        "orientation": int(orientation),
        "orientation_applied": bool(orientation_applied),
    }

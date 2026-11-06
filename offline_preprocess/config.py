"""配置加载、校验、解析与配置哈希。

配置文件为 JSON 对象，只允许本模块声明的键；未知键视为配置错误（防止拼写错误
静默生效）。解析后的配置写入 config.resolved.json，并以其规范化哈希作为
config_hash 贯穿清单与特征文件。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .errors import FatalCode, FatalError
from .util import atomic_write_json, canonical_json

# 默认值（需求文档 §9）
DEFAULTS: dict[str, Any] = {
    "recursive": True,
    "long_edge": 1600,
    "jpeg_quality": 90,
    "min_long_edge": 512,
    "alpha_policy": "skip",
    "missing_icc_policy": "assume_srgb_warn",
    "luminance_bins": 64,
    "hue_bins": 36,
    "palette_colors": 5,
    "near_black_threshold": 2,  # 8-bit 灰度级，等价 2/255
    "near_white_threshold": 253,  # 8-bit 灰度级，等价 253/255
    "near_duplicate_enabled": True,
    "near_duplicate_hamming": 5,
    "random_seed": 0,
    "strip_preview_exif": True,
    "overwrite": False,
    # 以下为实现补充项，均有明确默认值并参与 config_hash
    "hue_min_chroma": 5.0,
    "palette_max_sample_pixels": 50000,
    "palette_max_iterations": 40,
    "include_exif_summary": True,
}

_ALLOWED_KEYS = set(DEFAULTS)
_BOOL_KEYS = {
    "recursive",
    "near_duplicate_enabled",
    "strip_preview_exif",
    "overwrite",
    "include_exif_summary",
}
_INT_KEYS = {
    "long_edge",
    "jpeg_quality",
    "min_long_edge",
    "luminance_bins",
    "hue_bins",
    "palette_colors",
    "near_black_threshold",
    "near_white_threshold",
    "near_duplicate_hamming",
    "random_seed",
    "palette_max_sample_pixels",
    "palette_max_iterations",
}

_ALPHA_POLICIES = {"skip"}
_ICC_POLICIES = {"assume_srgb_warn"}


def _fail(message: str) -> None:
    raise FatalError(FatalCode.CONFIG_INVALID, message)


def _parse_threshold(key: str, value: Any) -> int:
    """接受 8-bit 级别整数，或 "N/255" 形式的字符串。"""
    if isinstance(value, str):
        text = value.strip()
        if "/" in text:
            num, den = text.split("/", 1)
            try:
                num_i, den_i = int(num), int(den)
            except ValueError:
                _fail(f"{key} 不是合法的分数：{value!r}")
            if den_i != 255:
                _fail(f"{key} 的分数分母必须是 255，当前为 {value!r}")
            value = num_i
        else:
            try:
                value = int(text)
            except ValueError:
                _fail(f"{key} 不是整数或 'N/255'：{value!r}")
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{key} 必须是 8-bit 整数或 'N/255'，当前为 {value!r}")
    if not 0 <= value <= 255:
        _fail(f"{key} 必须在 0..255，当前为 {value}")
    return value


def load_config_file(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        _fail(f"配置文件不存在：{path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        _fail(f"配置文件不是合法 JSON：{path}：{exc}")
    if not isinstance(data, dict):
        _fail(f"配置文件顶层必须是对象：{path}")
    return data


def resolve_config(user_config: dict[str, Any] | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """合并默认值 → 配置文件 → 命令行覆盖，并做类型与范围校验。"""
    merged: dict[str, Any] = dict(DEFAULTS)
    for source in (user_config or {}, overrides or {}):
        for key, value in source.items():
            if key not in _ALLOWED_KEYS:
                _fail(f"未知配置项：{key}（允许项：{', '.join(sorted(_ALLOWED_KEYS))}）")
            merged[key] = value

    resolved: dict[str, Any] = {}
    for key, value in merged.items():
        if key in _BOOL_KEYS:
            if not isinstance(value, bool):
                _fail(f"{key} 必须是布尔值，当前为 {value!r}")
            resolved[key] = value
        elif key in ("near_black_threshold", "near_white_threshold"):
            resolved[key] = _parse_threshold(key, value)
        elif key in _INT_KEYS:
            if isinstance(value, bool) or not isinstance(value, int):
                _fail(f"{key} 必须是整数，当前为 {value!r}")
            resolved[key] = value
        else:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                resolved[key] = float(value)
            elif isinstance(value, str):
                resolved[key] = value
            else:
                _fail(f"{key} 类型不支持：{value!r}")

    if resolved["long_edge"] < 1:
        _fail("long_edge 必须 >= 1")
    if not 1 <= resolved["jpeg_quality"] <= 100:
        _fail("jpeg_quality 必须在 1..100")
    if resolved["min_long_edge"] < 1:
        _fail("min_long_edge 必须 >= 1")
    if resolved["luminance_bins"] < 2:
        _fail("luminance_bins 必须 >= 2")
    if resolved["hue_bins"] < 2:
        _fail("hue_bins 必须 >= 2")
    if not 1 <= resolved["palette_colors"] <= 32:
        _fail("palette_colors 必须在 1..32")
    if not 0 <= resolved["near_duplicate_hamming"] <= 64:
        _fail("near_duplicate_hamming 必须在 0..64")
    if resolved["near_black_threshold"] >= resolved["near_white_threshold"]:
        _fail("near_black_threshold 必须小于 near_white_threshold")
    if resolved["hue_min_chroma"] < 0:
        _fail("hue_min_chroma 必须 >= 0")
    if resolved["palette_max_sample_pixels"] < 100:
        _fail("palette_max_sample_pixels 必须 >= 100")
    if resolved["palette_max_iterations"] < 1:
        _fail("palette_max_iterations 必须 >= 1")
    if resolved["alpha_policy"] not in _ALPHA_POLICIES:
        _fail(f"alpha_policy 首版只支持 {sorted(_ALPHA_POLICIES)}，当前为 {resolved['alpha_policy']!r}")
    if resolved["missing_icc_policy"] not in _ICC_POLICIES:
        _fail(
            "missing_icc_policy 首版只支持 "
            f"{sorted(_ICC_POLICIES)}，当前为 {resolved['missing_icc_policy']!r}"
        )
    if resolved["strip_preview_exif"] is not True:
        _fail("strip_preview_exif 首版只支持 true（预览一律不写 EXIF/GPS）")

    return resolved


def config_hash(resolved: dict[str, Any]) -> str:
    """16 位小写十六进制配置哈希（sha256 前 16 位）。"""
    digest = hashlib.sha256(canonical_json(resolved).encode("utf-8")).hexdigest()
    return digest[:16]


def write_resolved_config(path: str, resolved: dict[str, Any], cfg_hash: str) -> None:
    payload = {
        "config": resolved,
        "config_hash": cfg_hash,
        "note": "本文件为合并默认值与命令行参数后的最终配置；修改任一影响输出的项都会改变 config_hash。",
    }
    atomic_write_json(path, payload)

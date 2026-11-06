"""字段清单访问层：唯一事实来源是生成的 fields_manifest.json。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..errors import Codes, DecodeError

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "data" / "fields_manifest.json"

# 指纹里的逻辑键 → SDK 键。
# 指纹是"摄影语言"，XMP 是"SDK 方言"，两者不能混用；这一层是唯一的翻译点。
LOGICAL_TO_SDK: dict[str, str] = {
    "exposure_ev": "Exposure2012",
    "contrast": "Contrast2012",
    "highlights": "Highlights2012",
    "shadows": "Shadows2012",
    "whites": "Whites2012",
    "blacks": "Blacks2012",
    "clarity": "Clarity2012",
    "vibrance": "Vibrance",
    "saturation": "Saturation",
    "dehaze": "Dehaze",
    # 以下三项在插件里属于 tier=ext，默认 apply 不写 → 守卫会判 unsupported
    "sharpening_amount": "SharpenAmount",
    "grain_amount": "GrainAmount",
    "vignette_amount": "PostCropVignetteAmount",
    # 相对量：需要 r00 基线锚定，不能直接写
    "temperature_shift": "Temperature",
    "tint_shift": "Tint",
    # 色相类：指纹用角度，XMP 也是角度，直通
    "shadow_tint_hue": "SplitToningShadowHue",
    "highlight_tint_hue": "SplitToningHighlightHue",
}

# 语义是"增量"而不是"绝对值"的键
SHIFT_SDKS = {"Temperature", "Tint"}
SHIFT_LOGICAL = {"temperature_shift", "tint_shift"}

WB_PRESETS = {
    "As Shot", "Auto", "Daylight", "Cloudy", "Shade", "Tungsten",
    "Fluorescent", "Flash", "Custom",
}


class Fields:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.raw = manifest
        self.fields: dict[str, dict] = manifest["fields"]
        self.aliases: dict[str, str] = manifest.get("aliases") or {}
        self.v1: set[str] = set(manifest.get("v1_fields") or [])

    @classmethod
    @lru_cache(maxsize=4)
    def load(cls, path: str | Path | None = None) -> "Fields":
        target = Path(path) if path else DEFAULT_MANIFEST
        if not target.is_file():
            raise DecodeError(
                Codes.CONFIG_INVALID,
                f"缺少字段清单：{target}。先运行 python -m online_decode.tools.dump_fields",
                stage="config",
            )
        return cls(json.loads(target.read_text(encoding="utf-8")))

    # ---- 解析 ----
    def canonical(self, key: str) -> str | None:
        """把模型/指纹给的任意写法收敛到 manifest 里的规范键。"""
        if key in self.fields:
            return key
        mapped = LOGICAL_TO_SDK.get(key)
        if mapped and mapped in self.fields:
            return mapped
        alias = self.aliases.get(key)
        if alias and alias in self.fields:
            return alias
        return None

    def spec(self, key: str) -> dict | None:
        canonical = self.canonical(key)
        return self.fields.get(canonical) if canonical else None

    def is_writable(self, key: str) -> bool:
        canonical = self.canonical(key)
        return canonical in self.v1 if canonical else False

    def range_of(self, key: str) -> tuple[float, float] | None:
        spec = self.spec(key)
        if not spec or not spec.get("range"):
            return None
        low, high = spec["range"]
        return float(low), float(high)

    def type_of(self, key: str) -> str:
        spec = self.spec(key)
        return (spec or {}).get("type", "number")

    def group_of(self, key: str) -> str:
        spec = self.spec(key)
        return (spec or {}).get("group", "")

    def value_span(self, key: str) -> float:
        rng = self.range_of(key)
        if rng:
            return rng[1] - rng[0]
        return 100.0

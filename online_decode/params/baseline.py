"""基线锚定：把「相对量」换成「绝对值」。

指纹说的是 `temperature_shift: +350`，LR 只认 `Temperature: 5850`。
没有这一层，同一句"偏暖一点"落在不同原片上会得到完全不同的色温，
而且报告里也无法解释这个绝对值是怎么来的。
"""

from __future__ import annotations

from typing import Any

TEMPERATURE_RANGE = (2000.0, 50000.0)
TINT_RANGE = (-150.0, 150.0)
DYNAMIC_RANGES = {"Temperature": TEMPERATURE_RANGE, "Tint": TINT_RANGE}


def baseline_value(baseline: dict[str, Any], sdk: str) -> float | None:
    settings = (baseline or {}).get("settings") or {}
    value = settings.get(sdk)
    return float(value) if isinstance(value, (int, float)) else None


def resolve(
    sdk: str,
    value: Any,
    baseline: dict[str, Any],
    *,
    is_shift: bool,
) -> tuple[Any, list[str], str | None]:
    """返回 (resolved, notes, error)。error 非空表示无法安全锚定。"""
    notes: list[str] = []
    if not is_shift:
        return value, notes, None

    base = baseline_value(baseline, sdk)
    if base is None:
        return value, notes, f"基线快照里没有 {sdk}，无法把相对量锚定成绝对值"

    resolved = base + float(value)
    low, high = DYNAMIC_RANGES.get(sdk, (float("-inf"), float("inf")))
    if resolved < low:
        notes.append(f"{sdk} {resolved:.0f} 低于下限 {low:.0f}，已钳制")
        resolved = low
    elif resolved > high:
        notes.append(f"{sdk} {resolved:.0f} 高于上限 {high:.0f}，已钳制")
        resolved = high
    notes.append(f"{sdk} 以 r00 基线 {base:.0f} 锚定：{value:+.0f} → {resolved:.0f}")
    return resolved, notes, None

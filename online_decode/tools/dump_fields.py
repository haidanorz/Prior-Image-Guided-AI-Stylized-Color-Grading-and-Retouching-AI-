"""从插件源码生成机器可读的字段清单（fields_manifest.json）。

为什么不用手写第二份白名单：字段名在本项目里已经分裂过一次
（`SplitToning*` 与 `ColorGrade*` 混用、PV2012 后缀）。手抄必然漂移，
而漂移的表现是"某些调整被静默丢弃"——最难查的一类问题。

因此：唯一事实来源仍是 `XmpMapping.lua`，本脚本只做提取；控制器只读产物。
生成后会与插件 capabilities.json 里的 `feedback_fields_count` 对数，不一致就失败。

用法：
    python -m online_decode.tools.dump_fields            # 写入默认位置
    python -m online_decode.tools.dump_fields --check    # 只校验，不写
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LUA = ROOT / "AI-XMP-Color-Importer.lrdevplugin" / "src" / "XmpMapping.lua"
OUT = ROOT / "online_decode" / "data" / "fields_manifest.json"
CAPS = Path.home() / "Pictures" / "Lightroom" / "AI-XMP-Color-Importer.lrdevplugin" / "queue" / "capabilities.json"

FIELD_BLOCK = re.compile(r"^FIELDS\.(\w+)\s*=\s*\{(.*?)^\}", re.S | re.M)
HSL_LOOP = re.compile(r'FIELDS\["(\w+)"\s*\.\.\s*c\]\s*=\s*\{(.*?)^\}', re.S | re.M)
COLOR_LIST = re.compile(r"XmpMapping\.COLOR_NAMES\s*=\s*\{(.*?)\}", re.S)
ALIAS_BLOCK = re.compile(r"XmpMapping\.ALIASES\s*=\s*\{(.*?)^\}", re.S | re.M)


def _num(text: str) -> float | int:
    value = float(text)
    return int(value) if value.is_integer() else value


def _range(block: str) -> list | None:
    match = re.search(r"range\s*=\s*\{\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\}", block)
    if not match:
        return None
    return [_num(match.group(1)), _num(match.group(2))]


def _attr(block: str, name: str) -> str | None:
    match = re.search(rf'\b{name}\s*=\s*"([^"]*)"', block)
    return match.group(1) if match else None


def _colors(source: str) -> list[str]:
    match = COLOR_LIST.search(source)
    if not match:
        raise SystemExit("找不到 XmpMapping.COLOR_NAMES")
    return re.findall(r'"([A-Za-z]+)"', match.group(1))


def _hsl(kind: str, color: str) -> dict:
    return {
        "sdk": f"{kind}{color}",
        "type": "number",
        "group": "hsl",
        "tier": "v1",
        "range": [-100, 100],
        "display": f"{color} {kind.replace('Adjustment', '')}",
        "colorChannel": color,
    }


def build() -> dict:
    if not LUA.is_file():
        raise SystemExit(f"找不到 XmpMapping.lua：{LUA}")
    source = LUA.read_text(encoding="utf-8")

    fields: dict[str, dict] = {}

    for name, block in FIELD_BLOCK.findall(source):
        sdk = _attr(block, "sdk")
        if not sdk:
            continue
        fields[name] = {
            "sdk": sdk,
            "type": _attr(block, "type") or "number",
            "group": _attr(block, "group") or "",
            "tier": _attr(block, "tier") or "v1",
            "range": _range(block),
            "display": _attr(block, "display") or name,
        }

    # HSL 24 个字段是循环生成的，源码里没有逐个定义
    for prefix, kind in (
        ("HueAdjustment", "HueAdjustment"),
        ("SaturationAdjustment", "SaturationAdjustment"),
        ("LuminanceAdjustment", "LuminanceAdjustment"),
    ):
        if f'FIELDS["{prefix}"' not in source:
            continue
        for color in _colors(source):
            fields[f"{prefix}{color}"] = _hsl(kind, color)

    aliases: dict[str, str] = {}
    alias_match = ALIAS_BLOCK.search(source)
    if alias_match:
        for key, value in re.findall(r"(\w+)\s*=\s*\"(\w+)\"", alias_match.group(1)):
            aliases[key] = value

    v1 = {k: v for k, v in fields.items() if v["tier"] == "v1"}
    ext = {k: v for k, v in fields.items() if v["tier"] != "v1"}

    import hashlib

    return {
        "manifest_version": "1.0.0",
        "source": "AI-XMP-Color-Importer.lrdevplugin/src/XmpMapping.lua",
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "fields_version": "1.0.0",
        "counts": {"v1": len(v1), "ext": len(ext), "total": len(fields)},
        "fields": dict(sorted(fields.items())),
        "v1_fields": sorted(v1),
        "aliases": dict(sorted(aliases.items())),
    }


def expected_count() -> int | None:
    if not CAPS.is_file():
        return None
    try:
        return int(json.loads(CAPS.read_text(encoding="utf-8")).get("feedback_fields_count"))
    except (ValueError, TypeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 fields_manifest.json")
    parser.add_argument("--check", action="store_true", help="只校验产物是否最新")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    manifest = build()
    want = expected_count()
    got = manifest["counts"]["v1"]
    line = f"v1 字段 {got} 个，ext {manifest['counts']['ext']} 个"
    if want is not None:
        line += f"；插件声明 {want} 个"
        if want != got:
            print(f"[FAIL] {line} —— 数量不一致，白名单与插件已经漂移", file=sys.stderr)
            return 2
    print(line)

    out = Path(args.out)
    if args.check:
        old = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None
        if old != manifest:
            print(f"[FAIL] 产物不是最新：{out}", file=sys.stderr)
            return 3
        print(f"[ok] {out} 是最新的")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] 写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""解析模型产出的 XMP。

为什么需要这一层：模型现在直接交 XMP 文件，但 Lightroom 对不认识的属性名是
**静默丢弃**——不报错、不提示，表现就是"这轮调了但看不出变化"。
所以写入前必须把文件读回来，检查键名、类型、值域，再决定哪些能进 LR。

与插件 `src/XmpParser.lua` 的关系：
  * 插件负责最终权威解析（它要处理别名、白平衡枚举、曲线序列）；
  * 这里只做控制器侧的**预检**，覆盖 v1 标量字段，够拦住绝大多数错误；
  * 两边不一致时以插件为准：控制器的职责是"别把明显错的送进去"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# 这些是预设元数据，不是 Develop 参数；插件也会静默忽略
METADATA_KEYS = {
    "PresetType", "Version", "ProcessVersion", "HasSettings", "AlreadyApplied",
    "Cluster", "UUID", "Name", "ShortName", "SortName", "Group", "Description",
    "Copyright", "ContactInfo", "CameraModelRestriction",
    "SupportsAmount", "SupportsColor", "SupportsMonochrome",
    "SupportsHighDynamicRange", "SupportsNormalDynamicRange",
    "SupportsSceneReferred", "SupportsOutputReferred",
    "AIColorSchema", "AIVersion", "AIColorVersion", "AIModel",
}

ATTR_RE = re.compile(r'crs:([A-Za-z0-9_]+)\s*=\s*"([^"]*)"')
XPACKET_HINT = ("x:xmpmeta", "rdf:Description")


@dataclass
class ParsedXmp:
    values: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    has_sequences: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.values) and not self.issues


def _coerce(raw: str) -> Any:
    text = raw.strip()
    if text in ("True", "true"):
        return True
    if text in ("False", "false"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse(text: str) -> ParsedXmp:
    parsed = ParsedXmp()

    if not isinstance(text, str) or not text.strip():
        parsed.issues.append("XMP 内容为空")
        return parsed
    if not any(hint in text for hint in XPACKET_HINT):
        parsed.issues.append("不像 XMP：缺少 x:xmpmeta / rdf:Description")

    for key, raw in ATTR_RE.findall(text):
        if key in METADATA_KEYS:
            parsed.metadata[key] = raw
            continue
        value = _coerce(raw)
        if key in parsed.values and parsed.values[key] != value:
            parsed.duplicates.append(key)
        parsed.values[key] = value

    # 曲线、HSL 之类的序列不在属性里；出现说明模型用了元素写法，本层读不到
    if "rdf:Seq" in text or "rdf:Bag" in text:
        parsed.has_sequences = True
        parsed.issues.append("XMP 含 rdf:Seq/rdf:Bag 序列（如色调曲线），首版控制器不解析这类写法")

    return parsed


def parse_file(path: str) -> ParsedXmp:
    with open(path, encoding="utf-8") as handle:
        return parse(handle.read())

"""把台账里的写入值渲染成 Lightroom 能解析的 XMP。

为什么不让模型产出 XMP：字段名、类型、转义只要有一处不对，LR 会**静默丢弃**
（不报错），表现为"这轮调了但看不出变化"。放在代码里可以用测试固定住。

只支持 v1 白名单里的标量字段（数值 / 布尔 / 枚举）。曲线是 rdf:Seq，
首版不由模型设定；恢复已有曲线走 restore_settings，不经过这里。
"""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape, quoteattr

XMP_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"


def fmt_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _attr_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return fmt_number(value)
    return str(value)


def render(values: dict[str, Any], *, label: str = "Online decode round") -> str:
    attrs: list[tuple[str, str]] = [
        ("crs:PresetType", "Normal"),
        ("crs:Version", "15.0"),
        ("crs:HasSettings", "True"),
    ]
    for key in sorted(values):
        value = values[key]
        if value is None:
            continue
        if isinstance(value, (list, tuple, dict)):
            raise ValueError(f"{key} 是复合值，XMP 渲染层只接受标量")
        attrs.append((f"crs:{key}", _attr_value(value)))

    lines = [
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>',
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="prior-color-decode">',
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        '  <rdf:Description rdf:about=""',
        f'    xmlns:crs="{XMP_NS}"',
    ]
    for key, value in attrs:
        lines.append(f"    {key}={quoteattr(value)}")
    lines.extend(
        [
            "  >",
            "   <crs:Name>",
            "    <rdf:Alt>",
            f'     <rdf:li xml:lang="x-default">{escape(label)}</rdf:li>',
            "    </rdf:Alt>",
            "   </crs:Name>",
            "   <crs:Group>",
            "    <rdf:Alt>",
            '     <rdf:li xml:lang="x-default">Online Decode</rdf:li>',
            "    </rdf:Alt>",
            "   </crs:Group>",
            "  </rdf:Description>",
            " </rdf:RDF>",
            "</x:xmpmeta>",
            '<?xpacket end="w"?>',
            "",
        ]
    )
    return "\n".join(lines)

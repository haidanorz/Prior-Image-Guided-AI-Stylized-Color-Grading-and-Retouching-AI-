"""参数语义知识层（L3_cross_project_reference）。

为什么单独一层、且不并入指纹包：
  本层内容是通用 Lightroom 操作知识（哪个滑块管直方图哪一段、动了会牵动什么），
  不是这位摄影师的风格证据。按《离线建档后处理架构设计》§12.4，
  `L3_cross_project` 明文禁止作为技能卡生成进指纹包，也不得参与
  `global + bucket_delta + condition_delta` 的数值合成——否则评价阶段
  会把「通用知识」误认成「这个人的习惯」，风格证据链就脏了。

所以本层只做一件事：在模型动笔之前，把「这个参数该不该由它承担」讲清楚，
减少无效试错（例如往 90–100% 的纯白区压 Highlights）。

注入点（三处，都只读不写数值）：
  * brief 任务：`request["scene"]["parameter_knowledge"]`，decode_next 原样透出；
  * plan 任务：`request["parameter_knowledge"]`，decode_next 的 plan 分支显式透出
    （plan 分支不返回 brief 全文，所以不能只挂在 brief 上）；
  * 落档：`brief["parameter_knowledge"]`，只记 id / hash / 精简视图，供审计。

冲突优先级：指纹 anchors / narrative / guardrail 一律高于本层。
本层不得改写锚点，也不构成偏离锚点的理由。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import Codes, DecodeError
from ..util import canonical_json, read_json, sha256_bytes

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "parameter_knowledge.json"
RELATIVE_REF = "online_decode/data/parameter_knowledge.json"

# 逐条给细节的组（数据里也有 detail_groups，这里只作缺省兜底）
DEFAULT_DETAIL_GROUPS = ("whitebalance", "basic")


def load(path: str | Path | None = None) -> dict[str, Any]:
    """读取知识文件并复算哈希。文件缺失或结构不对是硬错误。"""
    target = Path(path) if path else DEFAULT_PATH
    data = read_json(target)
    if not isinstance(data, dict):
        raise DecodeError(
            Codes.PACKAGE_INVALID,
            f"参数知识文件缺失或不可解析：{target}",
            stage="preflight",
        )
    if not data.get("entries"):
        raise DecodeError(
            Codes.PACKAGE_INVALID,
            f"参数知识文件没有 entries：{target}",
            stage="preflight",
        )
    return data


def compute_hash(data: dict[str, Any]) -> str:
    payload = {k: v for k, v in data.items() if k != "knowledge_hash"}
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def hash_ok(data: dict[str, Any]) -> bool:
    declared = data.get("knowledge_hash")
    if not declared:
        return False
    return declared == compute_hash(data)


def compact(data: dict[str, Any], *, path: str | Path | None = None) -> dict[str, Any]:
    """生成注入给模型的精简视图，按组分层。

    为什么分层而不是全量：
      实测 brief 请求包 8696 字符、plan 12766 字符。87 条带原文引用的完整条目
      约 7900 字符，直接塞进去会让 brief 涨近一倍，而且每轮 plan 都重复一遍。
      更关键的是，多数条目对模型是冗余的——`SaturationAdjustmentRed` 一看名字
      就知道是「红的饱和度」，教程里真正不可推断的增量只有两类：
        (a) 影调控件各自负责直方图哪一段（tone_zone），
        (b) 什么时候动了也白动 / 什么时候会过头（caution）。
      这两类集中在 basic 与 whitebalance，所以只对这两组逐条展开；
      其余组给字段名 + 一句组级说明（group_notes），既保证模型知道字段存在，
      又不必为 24 条 HSL 付 24 份篇幅。

    `hash_verified=False` 只作提示，不阻断出片：
      本层不是数值事实来源，哈希不符最多说明有人手改过文件，
      该信息进报告即可，不该让一张照片卡在这里。这与锚点/曲线不同——
      那些是硬约束，必须拦。
    """
    entries = data.get("entries") or []
    detail_groups = tuple(data.get("detail_groups") or DEFAULT_DETAIL_GROUPS)
    group_notes = data.get("group_notes") or {}

    params: dict[str, Any] = {}
    other_groups: dict[str, dict[str, Any]] = {}
    blocked: list[str] = []

    for entry in entries:
        field = entry.get("field")
        if not field:
            continue
        if not entry.get("writable"):
            blocked.append(field)
            continue

        group = entry.get("group") or ""
        if group not in detail_groups:
            bucket = other_groups.setdefault(group, {"fields": []})
            bucket["fields"].append(field)
            continue

        item: dict[str, Any] = {}
        effect = entry.get("effect") or {}
        if effect.get("decrease"):
            item["down"] = effect["decrease"]
        if effect.get("increase"):
            item["up"] = effect["increase"]
        zone = entry.get("tone_zone") or (data.get("tone_zones") or {}).get(field)
        if isinstance(zone, list) and len(zone) == 2:
            item["zone"] = zone
        cautions = entry.get("caution") or []
        if cautions:
            item["caution"] = cautions[0]
        params[field] = item

    for group, bucket in other_groups.items():
        note = group_notes.get(group)
        if note:
            bucket["note"] = note

    workflow = data.get("workflow") or {}
    zones = {
        k: v
        for k, v in (data.get("tone_zones") or {}).items()
        if not k.startswith("_") and isinstance(v, list)
    }

    return {
        "knowledge_id": data.get("knowledge_id"),
        "layer": data.get("layer"),
        "status": data.get("status"),
        "scope": data.get("scope"),
        "knowledge_hash": data.get("knowledge_hash"),
        "hash_verified": hash_ok(data),
        "full_ref": RELATIVE_REF if path is None else str(path),
        "bounds": (data.get("bounds") or {}).get("reason"),
        "conflict_rule": (data.get("bounds") or {}).get("conflict_rule"),
        "workflow": {
            "order": workflow.get("order"),
            "summary": workflow.get("summary"),
            "equal_opposite": (workflow.get("equal_opposite_rule") or {}).get("rule"),
        },
        "tone_zones": zones,
        "params": params,
        "other_groups": other_groups,
        "blocked_fields": blocked,
        "blocked_reason": (
            "以下字段本接口写不进（tier=ext 或曲线为 rdf:Seq），"
            "在 XMP 里写它们会被守卫拒绝并浪费一轮，不要尝试。"
        ),
    }


def build(path: str | Path | None = None) -> dict[str, Any]:
    """一步到位：读文件 → 校验哈希 → 出精简视图。"""
    return compact(load(path), path=path)

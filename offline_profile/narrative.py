"""桶级语言描述：逐图描述证据 → 父代理归纳 → 桶 narrative.json。

边界：
- 只对训练侧 core 成员生成描述，不使用留出；
- 描述只取文字证据，不带 estimate/delta/central 数值；
- narrative 只挂在桶上，全局指纹不添加语言描述；
- narrative 不改动任何数值档案，只提供语境与意象的检索线索。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from .util import atomic_write_bytes, atomic_write_json, canonical_hash, read_json, read_jsonl

NARRATIVE_SCHEMA_VERSION = "1.0.0"

_TEXT_FIELDS = (
    "summary",
    "mood",
    "subject_context",
    "lighting_context",
    "color_language",
    "composition_tendencies",
)
_LIST_FIELDS = ("mood_keywords", "recurring_motifs", "exceptions", "cautions")
_ALLOWED = {
    "narrative_schema_version",
    "bucket_id",
    "status",
    "language",
    "summary",
    "mood",
    "mood_keywords",
    "subject_context",
    "lighting_context",
    "color_language",
    "composition_tendencies",
    "recurring_motifs",
    "exceptions",
    "cautions",
    "evidence_refs",
    "source",
    "narrative_hash",
}


def collect_bucket_descriptions(build_root: str, bucket_id: str) -> dict:
    build_root = os.path.abspath(build_root)
    stats_path = os.path.join(build_root, "buckets", bucket_id, "stats.json")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"缺少桶统计：{stats_path}")
    stats = read_json(stats_path)
    members = sorted(stats.get("members_train") or [])
    if not members:
        raise ValueError(f"{bucket_id} 没有训练侧成员")

    profile = read_json(os.path.join(build_root, "profile.json"))
    assignments = {item["image_id"]: item for item in read_jsonl(os.path.join(build_root, "assignments.jsonl"))}
    task_by_id = {item["image_id"]: item for item in read_jsonl(os.path.join(build_root, "mcp", "tasks.jsonl"))}

    rows = []
    for image_id in members:
        task = task_by_id.get(image_id)
        if not task or not task.get("normalized_path"):
            continue
        path = os.path.join(build_root, task["normalized_path"])
        if not os.path.isfile(path):
            continue
        analysis = read_json(path)
        assignment = assignments.get(image_id, {})
        rows.append(_description(analysis, assignment))

    if not rows:
        raise ValueError(f"{bucket_id} 没有可用的逐图描述")
    rows.sort(key=lambda item: item["image_id"])
    descriptions_path = os.path.join(build_root, "buckets", bucket_id, "descriptions.jsonl")
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_bytes(descriptions_path, text.encode("utf-8"))
    descriptions_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    prompt_path = os.path.join(build_root, "buckets", bucket_id, "narrative_prompt.md")
    prompt = _prompt(bucket_id, len(rows), descriptions_path, descriptions_hash, profile)
    atomic_write_bytes(prompt_path, prompt.encode("utf-8"))
    return {
        "bucket_id": bucket_id,
        "n": len(rows),
        "descriptions_file": f"buckets/{bucket_id}/descriptions.jsonl",
        "descriptions_hash": descriptions_hash,
        "prompt_file": f"buckets/{bucket_id}/narrative_prompt.md",
        "prompt": prompt,
    }


def submit_bucket_narrative(build_root: str, bucket_id: str, payload: dict) -> dict:
    build_root = os.path.abspath(build_root)
    stats_path = os.path.join(build_root, "buckets", bucket_id, "stats.json")
    stats = read_json(stats_path)
    members = set(stats.get("members_train") or [])
    descriptions_path = os.path.join(build_root, "buckets", bucket_id, "descriptions.jsonl")
    if not os.path.isfile(descriptions_path):
        raise FileNotFoundError(f"先运行 narrative-collect：{descriptions_path}")
    descriptions_hash = _sha256_file(descriptions_path)

    unknown = set(payload) - _ALLOWED
    if unknown:
        raise ValueError(f"narrative 包含未登记字段：{sorted(unknown)}")
    if payload.get("bucket_id") not in (None, bucket_id):
        raise ValueError("bucket_id 与目标桶不一致")
    problems = []
    for field in _TEXT_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{field} 必须是非空字符串")
        elif len(value) > 1200:
            problems.append(f"{field} 超过 1200 字符")
    for field in _LIST_FIELDS:
        value = payload.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            problems.append(f"{field} 必须是字符串数组")
    refs = payload.get("evidence_refs")
    if not isinstance(refs, list) or len(refs) < 3:
        problems.append("evidence_refs 至少 3 个 image_id")
    else:
        bad = [item for item in refs if item not in members]
        if bad:
            problems.append(f"evidence_refs 不在训练成员中：{bad[:5]}")
    if not isinstance(payload.get("recurring_motifs"), list) or len(payload["recurring_motifs"]) < 2:
        problems.append("recurring_motifs 至少 2 条")
    if problems:
        raise ValueError("；".join(problems))

    for field in _TEXT_FIELDS + _LIST_FIELDS:
        text = payload.get(field)
        values = text if isinstance(text, list) else [text]
        for value in values:
            if re.search(r"\b-?\d+(?:\.\d+)?\s*(?:EV|index|°)\b", value):
                raise ValueError(f"{field} 不能包含数值参数（delta/central/滑块值）")

    narrative = {
        "narrative_schema_version": NARRATIVE_SCHEMA_VERSION,
        "bucket_id": bucket_id,
        "status": "draft",
        "language": payload.get("language", "zh"),
        **{field: payload[field] for field in _TEXT_FIELDS + _LIST_FIELDS},
        "evidence_refs": list(dict.fromkeys(payload["evidence_refs"])),
        "source": {
            "build_id": read_json(os.path.join(build_root, "profile.json")).get("build_id"),
            "descriptions_file": f"buckets/{bucket_id}/descriptions.jsonl",
            "descriptions_hash": descriptions_hash,
            "stats_file": f"buckets/{bucket_id}/stats.json",
            "stats_hash": _sha256_file(stats_path),
        },
    }
    narrative["narrative_hash"] = canonical_hash(narrative)
    atomic_write_json(os.path.join(build_root, "buckets", bucket_id, "narrative.json"), narrative)
    return narrative


def _description(analysis: dict, assignment: dict) -> dict:
    scene = analysis.get("scene_observation", {})
    visual = analysis.get("visual_observation", {})
    hypotheses = analysis.get("edit_hypotheses", {})
    return {
        "image_id": analysis["image_id"],
        "scene_primary": assignment.get("bucket_id") or assignment.get("candidate_bucket"),
        "lighting": scene.get("lighting"),
        "brightness": scene.get("brightness"),
        "backlit": scene.get("backlit"),
        "subject": scene.get("subject"),
        "environment_notes": scene.get("environment_notes"),
        "overall_impression": analysis.get("overall_impression"),
        "narrative_observation": analysis.get("narrative_observation"),
        "visual_observation": visual,
        "edit_evidence": {
            key: item.get("evidence")
            for key, item in hypotheses.items()
            if item.get("evidence")
        },
        "quality_issues": analysis.get("quality", {}).get("issues", []),
        "local_adjustments_suspected": [
            {"region": item.get("region"), "kind": item.get("kind"), "evidence": item.get("evidence")}
            for item in analysis.get("local_adjustments_suspected", [])
        ],
        "warnings": analysis.get("warnings", []),
    }


def _prompt(bucket_id: str, count: int, descriptions_path: str, descriptions_hash: str, profile: dict) -> str:
    return f"""# 桶级语言归纳任务

你是一个风格档案的语言归纳器。请只根据给定的逐图描述，为桶 `{bucket_id}` 写一份
“可检索的视觉语言摘要”。这是桶风格的一部分，不是全局风格。

## 输入

- 逐图描述：`{descriptions_path}`（{count} 张训练侧 core 成员，JSONL）
- descriptions_hash：`{descriptions_hash}`
- build_id：`{profile.get("build_id")}`

请先完整读取描述文件，再归纳。只使用训练成员，不要引入留出图片或外部知识。

## 规则

1. 不写任何数值参数（delta、central、滑块值、EV、角度）。数值档案是唯一事实源。
2. 只总结跨图反复出现的语境、光线、色彩、构图与情绪；单图差异放 `exceptions`。
3. 正文不要出现图片文件名或 image_id；证据 ID 单独放 `evidence_refs`。
4. 不要写“万能话术”；每条判断都必须能在 descriptions 中找到依据。
5. 输出一个 JSON 对象，不要 markdown，不要解释。

## 输出字段

```json
{{
  "language": "zh",
  "summary": "2-4 句，说明这个桶共同的视觉语言与语境",
  "mood": "一句话的情绪/氛围",
  "mood_keywords": ["3-6 个词"],
  "subject_context": "反复出现的主体与场景语境",
  "lighting_context": "光线条件与处理倾向",
  "color_language": "色彩关系与染色倾向（不写数值）",
  "composition_tendencies": "构图与空间组织倾向",
  "recurring_motifs": ["至少 2 条反复出现的视觉母题"],
  "exceptions": ["1-4 条明显偏离的样本类型"],
  "cautions": ["1-4 条适用边界或误用风险"],
  "evidence_refs": ["img_...", "img_...", "至少 3 个训练成员 image_id"]
}}
```
"""


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

"""CreativeBrief：首次写入 LR 之前的"作战计划"。

为什么必须先有 brief 再动 LR：
  它是本次会话唯一的风格护栏。n 越小，后面越没有机会纠正方向性错误，
  所以 `preserve / avoid / goals` 必须在第一轮之前写死并可复核。

brief 里的目标必须"能用眼睛复核"——这不是文风要求，是评价阶段的判据来源：
`evaluation.style_alignment` 要对着 goals 逐条判断，空泛目标会让评价失去锚点。
"""

from __future__ import annotations

from typing import Any

from ..errors import Codes, DecodeError
from .resolver import PackageInfo, references

VAGUE_WORDS = ("更好看", "更高级", "高级感", "氛围感更强", "更有质感", "更有feel")
MIN_GOALS = 2
MAX_GOALS = 5


def build(
    *,
    session_id: str,
    photo: str,
    image_id: str,
    pkg: PackageInfo,
    observation: dict[str, Any],
    decision: dict[str, Any],
    narrative: dict[str, Any] | None,
    response: dict[str, Any],
    max_rounds: int,
    user_intent: str | None,
    strength: float,
) -> dict[str, Any]:
    refs = references(pkg, decision.get("bucket_id"))

    brief = {
        "brief_schema_version": "1.0.0",
        "session_id": session_id,
        "photo": photo,
        "image_id": image_id,
        "fingerprint": {
            "fingerprint_id": pkg.fingerprint_id,
            "fingerprint_hash": pkg.fingerprint_hash,
            "schema_version": pkg.schema_version,
            "status": pkg.status,
            "application_ready": pkg.application_ready,
        },
        "scene": {
            "observation": observation,
            "decision": decision,
            "narrative_ref": None,
            "narrative_hash": None,
            "branch_rationale": response.get("branch_rationale") or "",
            "hit_cautions": list(response.get("hit_cautions") or []),
            "hit_exceptions": list(response.get("hit_exceptions") or []),
        },
        "preserve": list(response.get("preserve") or []),
        "goals": list(response.get("goals") or []),
        "avoid": list(response.get("avoid") or []),
        "parameter_references": refs,
        "target_corrections": list(response.get("target_corrections") or []),
        # 模型的首轮产出是 XMP 文件（需求 8.2 的原始口径）；adjustments 仅作兼容保留
        "initial_xmp": response.get("initial_xmp") or "",
        "initial_param_notes": dict(response.get("param_notes") or {}),
        "initial_plan": list(response.get("initial_plan") or []),
        "uncertainties": list(response.get("uncertainties") or []),
        "user_intent": user_intent,
        "style_strength": strength,
        "max_rounds": max_rounds,
    }

    if narrative:
        brief["scene"]["narrative_ref"] = narrative.get("_file")
        brief["scene"]["narrative_hash"] = narrative.get("narrative_hash")
        brief["scene"]["narrative_summary"] = narrative.get("summary")
        brief["scene"]["mood"] = narrative.get("mood")
        brief["scene"]["color_language"] = narrative.get("color_language")
        brief["scene"]["cautions"] = list(narrative.get("cautions") or [])
        brief["scene"]["exceptions"] = list(narrative.get("exceptions") or [])

    problems = validate(brief)
    if problems:
        raise DecodeError(
            Codes.MODEL_RESPONSE_INVALID,
            "CreativeBrief 不完整：" + "；".join(problems),
            stage="brief",
            detail={"problems": problems},
        )
    return brief


def validate(brief: dict[str, Any]) -> list[str]:
    problems: list[str] = []

    rationale = (brief.get("scene") or {}).get("branch_rationale") or ""
    if len(rationale.strip()) < 10:
        problems.append("scene.branch_rationale 为空或过短（需说明属于桶内哪条支系）")

    goals = brief.get("goals") or []
    if not MIN_GOALS <= len(goals) <= MAX_GOALS:
        problems.append(f"goals 必须 {MIN_GOALS}–{MAX_GOALS} 条，当前 {len(goals)}")
    for goal in goals:
        if not isinstance(goal, str) or len(goal.strip()) < 4:
            problems.append("goals 含空项或过短")
            break
        if any(word in goal for word in VAGUE_WORDS):
            problems.append(f"goal 不可复核：{goal}")
            break

    if not brief.get("preserve"):
        problems.append("preserve 为空（至少要说明要保住什么）")
    if not brief.get("avoid"):
        problems.append("avoid 为空（至少要说明不能出现什么）")
    if not (brief.get("initial_xmp") or brief.get("initial_plan")):
        problems.append("initial_xmp 为空（首轮必须给出一份可直接写入的 XMP）")
    if not brief.get("parameter_references"):
        problems.append("parameter_references 为空（指纹没提供任何可参考参数）")

    return problems

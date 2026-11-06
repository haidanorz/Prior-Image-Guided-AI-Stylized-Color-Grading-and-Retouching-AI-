"""评价核验：让"我看过了"变成可检查的事实。

三条规则（对应架构 8.1）：
  1. `metric` 引用的数值必须在本轮 metrics.json 里真实存在，且偏差在容差内；
  2. `anchor` 必须是受控枚举（九个方位 + sky/subject/shadow/highlight/whole）；
  3. 任何非 met 的判定都必须至少有一条 evidence，否则该评价作废。

没有这三条，模型可以写一段"高光略有溢出、整体偏冷"的漂亮话，
而实际根本没打开过那张图。
"""

from __future__ import annotations

from typing import Any

from . import metrics as metrics_mod

ANCHORS = {
    "upper_left", "upper_center", "upper_right",
    "middle_left", "center", "middle_right",
    "lower_left", "lower_center", "lower_right",
    "sky", "subject", "shadow", "highlight", "whole",
}

VERDICTS = {"met", "partially_met", "not_met", "n/a"}
IMPROVEMENTS = {"better", "same", "worse", "uncertain"}
DECISIONS = {"continue", "stop_and_select", "restore_previous_best", "abort"}

VALUE_TOLERANCE = 0.02


def verify(evaluation: dict[str, Any], metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []

    for field in ("style_alignment", "branch_alignment", "technical_quality", "content_preservation"):
        block = evaluation.get(field)
        if not isinstance(block, dict):
            problems.append(f"{field} 缺失或不是对象")
        elif block.get("verdict") not in VERDICTS:
            problems.append(f"{field}.verdict 非法：{block.get('verdict')!r}")

    if evaluation.get("improvement") not in IMPROVEMENTS:
        problems.append(f"improvement 非法：{evaluation.get('improvement')!r}")
    if evaluation.get("decision") not in DECISIONS:
        problems.append(f"decision 非法：{evaluation.get('decision')!r}")

    evidence = evaluation.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        problems.append("evidence 为空：没有可核验的观察依据")
        evidence = []

    flat = metrics_mod.flatten(metrics or {})
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            problems.append(f"evidence[{index}] 不是对象")
            continue
        anchor = item.get("anchor")
        if anchor not in ANCHORS:
            problems.append(f"evidence[{index}].anchor 非法：{anchor!r}")
        if not str(item.get("observation") or "").strip():
            problems.append(f"evidence[{index}].observation 为空")

        metric_ref = item.get("metric")
        if metric_ref:
            actual = metrics_mod.lookup(flat, str(metric_ref))
            if actual is None:
                problems.append(f"evidence[{index}].metric 在 metrics 里不存在：{metric_ref}")
            else:
                claimed = item.get("value")
                if isinstance(claimed, (int, float)) and abs(float(claimed) - actual) > VALUE_TOLERANCE:
                    problems.append(
                        f"evidence[{index}] 数值与实测不符：声称 {claimed}，实测 {actual}"
                    )

    # 非 met 的判定必须有依据
    for field in ("style_alignment", "branch_alignment", "technical_quality", "content_preservation"):
        block = evaluation.get(field)
        if isinstance(block, dict) and block.get("verdict") in {"partially_met", "not_met"} and not evidence:
            problems.append(f"{field} 判为 {block.get('verdict')} 却没有 evidence")

    return not problems, problems

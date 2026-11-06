"""早停与最佳轮。

规则优先于分数：
  1. 有硬性技术失败的轮次不得成为最佳；
  2. 先比 preserve / cautions 违规，再比风格目标；
  3. 接近时选更保守、缺陷更少的；
  4. 仍无法判断就选更早的——不默认最后一轮。

刻意不用 0–100 的"审美总分"：单分数会把"目标达成"与"技术缺陷"混在一起，
而且换个提示词分数就漂，历史轮次之间不可比。
"""

from __future__ import annotations

from typing import Any

VERDICT_RANK = {"met": 2, "partially_met": 1, "n/a": 1, "not_met": 0, None: 0}
IMPROVEMENT_RANK = {"better": 3, "same": 2, "uncertain": 1, "worse": 0, None: 1}


def _hard_failure(record: dict[str, Any]) -> bool:
    return not record.get("ok", False)


def _count(record: dict[str, Any], key: str) -> int:
    value = (record.get("evaluation") or {}).get(key) or []
    return len(value) if isinstance(value, list) else int(bool(value))


def rank_key(record: dict[str, Any]) -> tuple:
    evaluation = record.get("evaluation") or {}
    style = evaluation.get("style_alignment") or {}
    technical = evaluation.get("technical_quality") or {}
    defects = technical.get("defects") or []
    improvement = evaluation.get("improvement")
    # 链式更优：improvement=better 的语义是"比上一轮更好"，属于递进关系，
    # 因此越晚的 better 越优（否则两轮同为 met+better 时会因"更早优先"把
    # 第 2 轮的真实改善吞掉）。same/worse/uncertain 仍按"更早更保守"排序。
    round_no = record.get("round", 0)
    round_key = -round_no if improvement == "better" else round_no
    return (
        1 if _hard_failure(record) else 0,
        _count(record, "preserve_violations"),
        _count(record, "caution_violations"),
        -VERDICT_RANK.get(style.get("verdict"), 0),
        -IMPROVEMENT_RANK.get(improvement, 1),
        len(defects) if isinstance(defects, list) else 0,
        round_key,
    )


def pick_best(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [r for r in rounds if not _hard_failure(r)]
    if not candidates:
        return None
    best = min(candidates, key=rank_key)
    # 轮次状态哈希：优先取 state_hash（apply 后完整快照的读数）；
    # apply 读数在刚导入的重 RAW 上可能不完整，不能作为恢复目标的权威哈希。
    best_hash = (
        best.get("state_hash")
        or (best.get("apply") or {}).get("settings_hash")
    )
    return {
        "round": best.get("round"),
        "settings_hash": best_hash,
        "snapshot": best.get("snapshot") or {},
        "preview": (best.get("render") or {}).get("path"),
        "reason": _reason(best, rounds),
        "rank": list(rank_key(best)),
    }


def _reason(best: dict[str, Any], rounds: list[dict[str, Any]]) -> str:
    evaluation = best.get("evaluation") or {}
    parts = [f"第 {best.get('round')} 轮"]
    verdict = (evaluation.get("style_alignment") or {}).get("verdict")
    if verdict:
        parts.append(f"风格目标 {verdict}")
    improvement = evaluation.get("improvement")
    if improvement:
        parts.append(f"improvement={improvement}")
    defects = (evaluation.get("technical_quality") or {}).get("defects") or []
    parts.append("无技术缺陷" if not defects else f"缺陷 {len(defects)} 项")
    if len(rounds) > 1:
        parts.append("按 preserve/cautions → 目标 → 保守程度 → 更早轮次排序")
    return "；".join(parts)


def should_stop(
    rounds: list[dict[str, Any]],
    *,
    max_rounds: int,
    stop_on_single_no_progress: bool,
    has_best: bool,
    min_rounds: int = 1,
) -> tuple[bool, str | None]:
    if not rounds:
        return False, None

    last = rounds[-1]
    evaluation = last.get("evaluation") or {}
    decision = evaluation.get("decision")
    improvement = evaluation.get("improvement")

    if evaluation.get("decision") == "abort":
        return True, "aborted_by_model"

    # 强制最少轮数（2026-11-06 用户决策）：未满 min_rounds 时只认 abort。
    # 理由：首轮交的是桶中心参数（已是"解"而非"探索"），一轮即 stop 会让
    # "达到桶中心"被误判为"本图最优"；第 2 轮才有机会做本图收敛或确认无改善。
    if len(rounds) < int(min_rounds):
        return False, None

    if decision == "stop_and_select":
        return True, "goal_met"
    if decision == "restore_previous_best":
        return True, "model_restore"

    if has_best:
        if improvement == "worse":
            return True, "regression"
        if stop_on_single_no_progress:
            # 首轮还没"建立基线之外的东西"，此时判无改善就停会把整场会话压成一轮。
            if len(rounds) >= 2 and improvement in ("same", "uncertain"):
                return True, "no_progress"
        elif len(rounds) >= 2:
            # 长预算档（n>=4，默认 n=5 即此档）：连续两轮无改善才停。
            # 只写文档不实现的话，5 轮会一路跑到预算耗尽，等于没有早停。
            last_two = [
                (item.get("evaluation") or {}).get("improvement")
                for item in rounds[-2:]
            ]
            if all(value in ("same", "worse") for value in last_two):
                return True, "no_progress"

    if len(rounds) >= max_rounds:
        return True, "budget"
    return False, None

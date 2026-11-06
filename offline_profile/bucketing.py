"""场景分桶：软候选、硬成员等级。

只以 scene_primary 建桶；other 永不成为正式桶；lighting/brightness/backlit
只作为条件属性保留在 assignment 中，不参与建桶。
"""

from __future__ import annotations

from .version import TAXONOMY_VERSION


def assign_scene(analysis: dict, cfg: dict) -> dict:
    candidates = sorted(
        (
            {"label": item["label"], "score": float(item["score"])}
            for item in analysis["scene_observation"]["primary_candidates"]
        ),
        key=lambda item: (-item["score"], item["label"]),
    )
    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    margin = round(top["score"] - second["score"], 6) if second else round(top["score"], 6)

    label = top["label"]
    if label == "other":
        level, reason = "unassigned", "other_label"
    elif top["score"] >= float(cfg["core_top_score"]) and margin >= float(cfg["core_margin"]):
        level, reason = "core", "core_thresholds_met"
    elif top["score"] >= float(cfg["associate_top_score"]):
        level = "associate"
        reason = (
            "core_score_not_met"
            if top["score"] < float(cfg["core_top_score"])
            else "core_margin_not_met"
        )
    else:
        level, reason = "unassigned", "below_associate_threshold"

    return {
        "image_id": analysis["image_id"],
        "taxonomy_version": TAXONOMY_VERSION,
        "level": level,
        "candidate_bucket": label,
        "bucket_id": label if level == "core" else None,
        "top_label": label,
        "top_score": round(top["score"], 6),
        "second_label": second["label"] if second else None,
        "second_score": round(second["score"], 6) if second else None,
        "margin": margin,
        "reason": reason,
        "candidates": candidates,
        "scene_conditions": {
            "lighting": analysis["scene_observation"]["lighting"],
            "brightness": analysis["scene_observation"]["brightness"],
            "backlit": analysis["scene_observation"]["backlit"],
        },
    }


def assignment_summary(assignments: list[dict]) -> dict:
    levels = {"core": 0, "associate": 0, "unassigned": 0}
    buckets: dict[str, int] = {}
    for item in assignments:
        levels[item["level"]] = levels.get(item["level"], 0) + 1
        if item["level"] == "core":
            buckets[item["bucket_id"]] = buckets.get(item["bucket_id"], 0) + 1
    return {
        "levels": levels,
        "core_by_bucket": dict(sorted(buckets.items())),
        "other_unassigned": sum(
            1 for item in assignments if item["level"] == "unassigned" and item["top_label"] == "other"
        ),
    }

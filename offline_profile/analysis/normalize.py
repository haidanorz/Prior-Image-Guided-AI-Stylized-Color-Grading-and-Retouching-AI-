"""分析响应规范化与语义校验。

原则：结构非法、数值越界、缺证据、ID 不匹配一律判为非法，不静默 clamp、
不补造证据；只做两件允许的确定性修正——候选分数归一化、亮度以客观特征为准。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from ..contract import edit_parameter_index, schema_errors
from ..errors import InvalidAnalysis
from ..intake import IntakeImage
from ..version import ANALYSIS_SCHEMA_VERSION, PROMPT_VERSION
from .adapter import ModelInfo
from .objective import objective_brightness

_TOL = 1e-9


@dataclass
class NormalizedAnalysis:
    payload: dict
    warnings: list[str] = field(default_factory=list)
    score_normalized: bool = False
    brightness_overridden: bool = False


def normalize_analysis(
    raw_text: str,
    *,
    image: IntakeImage,
    expected_fingerprint: str,
    config: dict,
    model_info: ModelInfo,
) -> NormalizedAnalysis:
    text = raw_text.strip().lstrip("\ufeff")
    warnings: list[str] = []

    if text.startswith("```"):
        text, stripped = _strip_fence(text)
        if stripped:
            warnings.append("markdown_fence_stripped")

    try:
        payload = json.loads(text, parse_constant=_reject_constant)
    except Exception as exc:
        raise InvalidAnalysis("响应不是合法 JSON", [str(exc)]) from exc
    if not isinstance(payload, dict):
        raise InvalidAnalysis("响应顶层必须是对象")

    # 模型身份由适配器提供，模型自报的信息只作为提示，不参与事实判断
    reported = payload.get("model_info")
    if reported != model_info.as_dict():
        warnings.append("model_info_corrected")
    payload["model_info"] = model_info.as_dict()

    errors = schema_errors(payload)
    if errors:
        raise InvalidAnalysis("Schema 校验失败", errors)

    _check_identity(payload, image, expected_fingerprint)
    _validate_narrative(payload)

    score_normalized = _normalize_candidates(payload, warnings)
    brightness_overridden = _enforce_brightness(payload, image, config, warnings)
    _validate_edit_hypotheses(payload)

    existing = payload.get("warnings") or []
    merged: list[str] = []
    for item in list(existing) + warnings:
        if item not in merged:
            merged.append(item)
    payload["warnings"] = merged

    return NormalizedAnalysis(
        payload=payload,
        warnings=merged,
        score_normalized=score_normalized,
        brightness_overridden=brightness_overridden,
    )


def _strip_fence(text: str) -> tuple[str, bool]:
    first_newline = text.find("\n")
    if first_newline == -1:
        return text, False
    body = text[first_newline + 1 :].rstrip()
    if body.endswith("```"):
        body = body[:-3].rstrip()
    return body, True


def _reject_constant(value: str):
    raise ValueError(f"非法 JSON 常量：{value}")


def _check_identity(payload: dict, image: IntakeImage, expected_fingerprint: str) -> None:
    problems: list[str] = []
    if payload.get("analysis_schema_version") != ANALYSIS_SCHEMA_VERSION:
        problems.append(
            f"analysis_schema_version={payload.get('analysis_schema_version')!r}，期望 {ANALYSIS_SCHEMA_VERSION}"
        )
    if payload.get("prompt_version") != PROMPT_VERSION:
        problems.append(f"prompt_version={payload.get('prompt_version')!r}，期望 {PROMPT_VERSION}")
    if payload.get("image_id") != image.image_id:
        problems.append(f"image_id={payload.get('image_id')!r}，期望 {image.image_id}")
    if payload.get("source_sha256") != image.source_sha256:
        problems.append("source_sha256 与输入不一致")
    if payload.get("input_fingerprint") != expected_fingerprint:
        problems.append("input_fingerprint 与当前输入不一致")
    if problems:
        raise InvalidAnalysis("分析响应与输入不匹配", problems)


_PARAM_UNIT_PATTERN = re.compile(r"\b[-+]?\d+(?:\.\d+)?\s*(?:EV|index|°|degrees?)\b", re.IGNORECASE)


def _validate_narrative(payload: dict) -> None:
    narrative = payload["narrative_observation"]
    problems = []
    for field, value in narrative.items():
        values = value if isinstance(value, list) else [value]
        for text in values:
            if _PARAM_UNIT_PATTERN.search(str(text)):
                problems.append(f"narrative_observation.{field} 不得包含数值参数：{text}")
    if problems:
        raise InvalidAnalysis("narration 混入数值参数", problems)


def _normalize_candidates(payload: dict, warnings: list[str]) -> bool:
    candidates = payload["scene_observation"]["primary_candidates"]
    deduped: dict[str, float] = {}
    duplicate = False
    for item in candidates:
        label = item["label"]
        score = float(item["score"])
        if label in deduped:
            duplicate = True
            deduped[label] = max(deduped[label], score)
        else:
            deduped[label] = score

    total = sum(deduped.values())
    if total <= 0:
        raise InvalidAnalysis("候选分数之和必须大于 0")

    adjusted = duplicate
    normalized: list[dict] = []
    for label, score in deduped.items():
        value = round(score / total, 6)
        if abs(value - score) > _TOL:
            adjusted = True
        normalized.append({"label": label, "score": value})

    residual = round(1.0 - sum(item["score"] for item in normalized), 6)
    if abs(residual) > 0:
        normalized.sort(key=lambda item: (-item["score"], item["label"]))
        normalized[0]["score"] = round(normalized[0]["score"] + residual, 6)
        adjusted = True

    normalized.sort(key=lambda item: (-item["score"], item["label"]))
    for item in normalized:
        if not -_TOL <= item["score"] <= 1.0 + _TOL:
            raise InvalidAnalysis(f"归一化后候选分数越界：{item}")
        item["score"] = min(1.0, max(0.0, round(item["score"], 6)))

    payload["scene_observation"]["primary_candidates"] = normalized
    if duplicate:
        warnings.append("duplicate_candidate_labels")
    if adjusted:
        warnings.append("candidate_scores_renormalized")
    return adjusted


def _enforce_brightness(payload: dict, image: IntakeImage, config: dict, warnings: list[str]) -> bool:
    objective = objective_brightness(image.feature, config)
    reported = payload["scene_observation"]["brightness"]
    if reported == objective:
        return False
    payload["scene_observation"]["brightness"] = objective
    warnings.append("brightness_overridden_by_objective")
    return True


def _in_circular_range(estimate: float, lower: float, upper: float) -> bool:
    est = estimate % 360.0
    if lower <= upper:
        return lower - _TOL <= est <= upper + _TOL
    return est >= lower - _TOL or est <= upper + _TOL


def _validate_edit_hypotheses(payload: dict) -> None:
    registry = edit_parameter_index()
    hypotheses = payload["edit_hypotheses"]
    problems: list[str] = []
    for key, slot in hypotheses.items():
        spec = registry.get(key)
        if spec is None:
            problems.append(f"未登记的参数键：{key}")
            continue

        estimate = float(slot["estimate"])
        lower, upper = float(slot["range"][0]), float(slot["range"][1])
        if not all(math.isfinite(v) for v in (estimate, lower, upper)):
            problems.append(f"{key} 含非有限数值")
            continue
        circular = spec["kind"] == "circular"
        if not circular and lower > upper + _TOL:
            problems.append(f"{key} 的 range 下界大于上界：{slot['range']}")
            continue
        if circular:
            if not _in_circular_range(estimate, lower, upper):
                problems.append(f"{key} 的 estimate 不在环形 range 内：{estimate} ∉ [{lower}, {upper}]")
                continue
        elif estimate < lower - _TOL or estimate > upper + _TOL:
            problems.append(f"{key} 的 estimate 不在 range 内：{estimate} ∉ [{lower}, {upper}]")
            continue

        lo_bound, hi_bound = float(spec["min"]), float(spec["max"])
        if estimate < lo_bound - _TOL or estimate > hi_bound + _TOL:
            problems.append(f"{key} 的 estimate 越界：{estimate} ∉ [{lo_bound}, {hi_bound}]")
            continue
        if lower < lo_bound - _TOL or upper > hi_bound + _TOL:
            problems.append(f"{key} 的 range 越界：[{lower}, {upper}] ⊄ [{lo_bound}, {hi_bound}]")
            continue

        if circular:
            slot["estimate"] = round(estimate % 360.0, 6)
            slot["range"] = [round(lower, 6), round(upper, 6)]
        else:
            slot["estimate"] = round(estimate, 6)
            slot["range"] = [round(lower, 6), round(upper, 6)]
        slot["confidence"] = round(float(slot["confidence"]), 6)

    if problems:
        raise InvalidAnalysis("edit_hypotheses 校验失败", problems)

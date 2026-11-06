"""合成预处理输入与 Mock 分析响应。"""

from __future__ import annotations

import json
import os

from offline_profile.config import config_hash, resolve_config
from offline_profile.intake import image_fingerprint, load_input
from offline_profile.version import ANALYSIS_SCHEMA_VERSION, PROMPT_VERSION

MODEL_INFO = {"provider": "mock", "model": "canned-response-v1", "version": "1.0.0", "is_mock": True}


def build_input(root: str, count: int = 12, p50: float = 0.45, near_groups: list[list[int]] | None = None) -> dict:
    input_dir = os.path.join(root, "preprocess_out")
    preview_dir = os.path.join(input_dir, "previews")
    feature_dir = os.path.join(input_dir, "features")
    source_dir = os.path.join(root, "sources")
    for path in (preview_dir, feature_dir, source_dir):
        os.makedirs(path, exist_ok=True)

    ids = [f"img_{index:016x}" for index in range(1, count + 1)]
    records = []
    for index, image_id in enumerate(ids, start=1):
        source_sha = f"{index:064x}"
        source_path = os.path.join(source_dir, f"{image_id}.jpg")
        with open(source_path, "wb") as handle:
            handle.write(f"source-{index}".encode("utf-8"))
        preview_rel = f"previews/{image_id}.jpg"
        feature_rel = f"features/{image_id}.json"
        with open(os.path.join(input_dir, preview_rel), "wb") as handle:
            handle.write(f"preview-{index}".encode("utf-8"))
        feature = {
            "feature_schema_version": "1.0.0",
            "preprocess_version": "1.0.0",
            "image_id": image_id,
            "source_sha256": source_sha,
            "luminance": {"percentiles": {"p50": p50}},
            "source": {"was_grayscale": False},
            "geometry": {"output_width": 1600, "output_height": 1067},
        }
        with open(os.path.join(input_dir, feature_rel), "w", encoding="utf-8") as handle:
            json.dump(feature, handle, ensure_ascii=False)
        import hashlib

        preview_sha = hashlib.sha256(f"preview-{index}".encode("utf-8")).hexdigest()
        records.append(
            {
                "image_id": image_id,
                "status": "ready",
                "source_path": source_path,
                "source_sha256": source_sha,
                "source_size_bytes": 100,
                "preview_path": preview_rel,
                "feature_path": feature_rel,
                "preview_sha256": preview_sha,
                "duplicate_of": None,
                "skip_reason": None,
                "error_code": None,
                "warnings": [],
                "preprocess_version": "1.0.0",
                "feature_schema_version": "1.0.0",
                "config_hash": "deadbeefdeadbeef",
                "processed_at": "2026-11-04T00:00:00Z",
            }
        )

    with open(os.path.join(input_dir, "manifest.jsonl"), "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    groups = []
    if near_groups:
        for group_index, members in enumerate(near_groups, start=1):
            groups.append(
                {
                    "group_id": f"nd_{group_index:04d}",
                    "members": [{"image_id": ids[index - 1], "phash": "0" * 16} for index in members],
                    "pairs": [],
                }
            )
    with open(os.path.join(input_dir, "near_duplicates.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": "2026-11-04T00:00:00Z",
                "algorithm": "dct_phash_64_v1",
                "threshold": 5,
                "review_required": True,
                "groups": groups,
            },
            handle,
            ensure_ascii=False,
        )

    return {"root": root, "input_dir": input_dir, "ids": ids, "records": records}


def build_analyses(input_dir: str, cfg: dict | None = None, **overrides) -> dict:
    cfg = cfg or resolve_config()
    cfg_hash = config_hash(cfg)
    intake = load_input(input_dir)
    analyses = {}
    for image in intake.images:
        fingerprint = image_fingerprint(
            image,
            analysis_config_hash=cfg_hash,
            prompt_version=PROMPT_VERSION,
            model_info=MODEL_INFO,
            schema_version=ANALYSIS_SCHEMA_VERSION,
        )
        analyses[image.image_id] = base_analysis(image, fingerprint, **overrides)
    return analyses


def base_analysis(image, fingerprint: str, **overrides) -> dict:
    payload = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "image_id": image.image_id,
        "source_sha256": image.source_sha256,
        "input_fingerprint": fingerprint,
        "model_info": dict(MODEL_INFO),
        "prompt_version": PROMPT_VERSION,
        "scene_observation": {
            "primary_candidates": [
                {"label": "portrait", "score": 0.8},
                {"label": "natural_landscape", "score": 0.15},
                {"label": "other", "score": 0.05},
            ],
            "lighting": "daylight",
            "subject": "a person standing outdoors",
            "brightness": "mid",
            "backlit": False,
            "backlit_evidence": "the face is evenly lit with no rim light",
            "environment_notes": "open shade near a wall",
        },
        "visual_observation": {
            "contrast": "neutral",
            "shadow_detail": "neutral",
            "highlight_rolloff": "neutral",
            "black_level": "neutral",
            "warmth": "neutral",
            "saturation_global": "neutral",
            "saturation_skin": "neutral",
            "saturation_sky": "neutral",
            "saturation_foliage": "neutral",
            "shadow_tint_strength": "neutral",
            "highlight_tint_strength": "neutral",
            "grain_visibility": "neutral",
            "vignette_visibility": "neutral",
            "sharpness_impression": "neutral",
            "clarity_impression": "neutral",
        },
        "narrative_observation": {
            "mood": "安静而克制的户外氛围",
            "atmosphere": "光线柔和，画面节奏平缓，没有强烈的戏剧冲突。",
            "light_quality": "漫射日光把阴影抬亮，天空保持浅淡。",
            "sense_of_place": "开阔的户外场景，空间层次由近处主体和远处背景拉开。",
            "seasonal_cues": "植被与光线暗示温和季节，但无法确认具体时令。",
            "composition_rhythm": "主体略偏中心，前景与背景之间留有呼吸空间。",
            "keywords": ["安静", "漫射光", "开阔"],
            "basis": "柔和阴影、浅色天空、低反差的主体与背景关系",
        },
        "edit_hypotheses": {
            "exposure_ev": {
                "estimate": 0.2,
                "range": [0.0, 0.4],
                "confidence": 0.6,
                "evidence": "midtones sit slightly above the measured baseline",
            }
        },
        "quality": {"overall": "good", "issues": [], "confidence": 0.8},
        "local_adjustments_suspected": [],
        "overall_impression": "A clean outdoor portrait with restrained color.",
        "warnings": [],
    }
    payload.update(overrides)
    return payload


def write_mock(mock_dir: str, analyses: dict) -> None:
    os.makedirs(mock_dir, exist_ok=True)
    for image_id, payload in analyses.items():
        with open(os.path.join(mock_dir, f"{image_id}.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

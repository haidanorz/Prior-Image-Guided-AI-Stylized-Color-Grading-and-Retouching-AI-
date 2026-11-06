"""构建目录完整性校验（只读）。"""

from __future__ import annotations

import os
from typing import Any

from .bucketing import assign_scene
from .config import resolve_config
from .contract import schema_errors
from .errors import AnalysisStatus
from .splitting import split_hash_of
from .util import canonical_hash, read_json, read_jsonl


def verify_build(build_root: str) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []

    def add(code: str, message: str, image_id: str | None = None) -> None:
        problems.append({"code": code, "image_id": image_id, "message": message})

    required = [
        "manifest.jsonl",
        "assignments.jsonl",
        "evaluation_split.json",
        "episodes.jsonl",
        "run_report.json",
        "config.resolved.json",
    ]
    for name in required:
        if not os.path.isfile(os.path.join(build_root, name)):
            add("MISSING_FILE", f"缺少必需文件：{name}")
    if problems:
        return {"ok": False, "checked": 0, "problems": problems, "build_root": build_root}

    manifest = read_jsonl(os.path.join(build_root, "manifest.jsonl"))
    assignments = read_jsonl(os.path.join(build_root, "assignments.jsonl"))
    split = read_json(os.path.join(build_root, "evaluation_split.json"))
    report = read_json(os.path.join(build_root, "run_report.json"))
    resolved = read_json(os.path.join(build_root, "config.resolved.json"))
    cfg = resolve_config(resolved.get("config"))
    episodes = read_jsonl(os.path.join(build_root, "episodes.jsonl"))

    assignment_by_id = {item["image_id"]: item for item in assignments}
    if len(assignment_by_id) != len(assignments):
        add("ASSIGNMENT_DUPLICATE", "assignments.jsonl 存在重复 image_id")

    normalized_count = 0
    for row in manifest:
        image_id = row["image_id"]
        status = row.get("analysis_status")
        if status == AnalysisStatus.NORMALIZED:
            normalized_count += 1
            normalized_path = os.path.join(build_root, row["normalized_path"]) if row.get("normalized_path") else None
            if not normalized_path or not os.path.isfile(normalized_path):
                add("NORMALIZED_MISSING", "规范化分析文件缺失", image_id)
                continue
            try:
                payload = read_json(normalized_path)
            except Exception as exc:
                add("NORMALIZED_UNREADABLE", f"规范化分析不可读：{exc}", image_id)
                continue
            errors = schema_errors(payload)
            for message in errors:
                add("SCHEMA_INVALID", message, image_id)
            assignment = assignment_by_id.get(image_id)
            if assignment is None:
                add("ASSIGNMENT_MISSING", "缺少分桶记录", image_id)
            else:
                expected = assign_scene(payload, cfg)
                for key in ("level", "bucket_id", "top_score", "margin"):
                    if assignment.get(key) != expected.get(key):
                        add(
                            "ASSIGNMENT_MISMATCH",
                            f"{key} 与重新判定不一致：{assignment.get(key)!r} != {expected.get(key)!r}",
                            image_id,
                        )
        else:
            if row.get("normalized_path"):
                add("STALE_NORMALIZED_PATH", "非 normalized 记录不应带 normalized_path", image_id)
            assignment = assignment_by_id.get(image_id)
            if assignment is not None:
                add("ASSIGNMENT_ORPHAN", "未成功分析的图片不应有分桶记录", image_id)

    for image_id in assignment_by_id:
        match = next((row for row in manifest if row["image_id"] == image_id), None)
        if match is None:
            add("ASSIGNMENT_UNKNOWN_IMAGE", "分桶记录不在 manifest 中", image_id)
        elif match.get("analysis_status") != AnalysisStatus.NORMALIZED:
            add("ASSIGNMENT_STATUS", "分桶记录对应的分析状态不是 normalized", image_id)

    expected_counts = report.get("analysis", {}).get("counts", {})
    if expected_counts.get("normalized") != normalized_count:
        add("REPORT_COUNT", "run_report 的 normalized 计数与 manifest 不一致")
    if len(manifest) != report.get("preprocess", {}).get("manifest_total", len(manifest)) - report.get("preprocess", {}).get("non_ready", 0):
        add("MANIFEST_COUNT", "manifest 行数与 run_report 的 ready 数不一致")

    _verify_split(split, assignment_by_id, manifest, add)
    for index, row in enumerate(episodes, start=1):
        for key in ("episode_id", "build_id", "event_type", "action", "result"):
            if not row.get(key):
                add("EPISODE_INVALID", f"episodes.jsonl 第 {index} 行缺少 {key}")

    return {
        "ok": not problems,
        "checked": len(manifest),
        "normalized": normalized_count,
        "problems": problems,
        "build_root": build_root,
    }


def _verify_split(split: dict, assignment_by_id: dict, manifest: list[dict], add) -> None:
    side: dict[str, str] = {}
    for bucket_id, entry in split.get("buckets", {}).items():
        members = set(entry.get("train", [])) | set(entry.get("holdout", []))
        overlap = set(entry.get("train", [])) & set(entry.get("holdout", []))
        if overlap:
            add("SPLIT_OVERLAP", f"{bucket_id} 的训练与留出重叠：{sorted(overlap)}")
        core_ids = {
            image_id
            for image_id, item in assignment_by_id.items()
            if item["level"] == "core" and item["bucket_id"] == bucket_id
        }
        status = entry.get("split_status")
        if status in ("split", "kfold") and members != core_ids:
            add("SPLIT_MEMBERSHIP", f"{bucket_id} 的划分成员与 core 集合不一致")
        if status in ("split", "kfold"):
            for image_id in members:
                if image_id in side:
                    add("SPLIT_DUPLICATE", f"{image_id} 出现在多个桶的划分中", image_id)
                side[image_id] = "train" if image_id in entry["train"] else "holdout"
        if status == "kfold":
            folds = entry.get("folds") or []
            flat = [image_id for fold in folds for image_id in fold]
            if len(folds) < 2:
                add("KFOLD_TOO_FEW", f"{bucket_id} 标为 kfold 但折数不足")
            if len(flat) != len(set(flat)):
                add("KFOLD_DUPLICATE", f"{bucket_id} 的折成员重复")
            if set(flat) != core_ids:
                add("KFOLD_MEMBERSHIP", f"{bucket_id} 的折成员未覆盖全部 core")
            if any(not fold for fold in folds):
                add("KFOLD_EMPTY", f"{bucket_id} 存在空折")
        for unit in entry.get("near_duplicate_units", []):
            if unit and not (set(unit) <= set(entry.get("train", [])) or set(unit) <= set(entry.get("holdout", []))):
                add("NEAR_DUPLICATE_LEAK", f"{bucket_id} 的近重复组跨越训练/留出：{unit}")

    for row in manifest:
        expected = side.get(row["image_id"])
        if row.get("split") != expected:
            add("SPLIT_MANIFEST", f"manifest 的 split 字段应为 {expected!r}", row["image_id"])

    recomputed = split_hash_of(split)
    if recomputed != split.get("split_hash"):
        add("SPLIT_HASH", "evaluation_split.json 的 split_hash 与内容不一致")

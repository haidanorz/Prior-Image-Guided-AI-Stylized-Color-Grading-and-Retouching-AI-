"""批处理编排：输入校验 → 单图分析 → 规范化 → 分桶 → 划分 → 落盘。"""

from __future__ import annotations

import os
import platform
import time
from typing import Any, Callable

from .analysis.adapter import build_adapter
from .analysis.cache import ResponseCache
from .analysis.runner import AnalysisRunner
from .bucketing import assign_scene, assignment_summary
from .config import config_hash, load_config_file, resolve_config, write_resolved_config
from .contract import load_parameters, load_taxonomy, registry_hash, taxonomy_hash
from .episodes import EpisodeStore
from .errors import AnalysisStatus, FatalCode, FatalError
from .intake import load_input
from .splitting import build_evaluation_split
from .store import BuildStore
from .util import (
    atomic_write_json,
    now_utc_iso,
    read_json,
    relative_posix,
    write_jsonl_atomic,
)
from .version import (
    ANALYSIS_SCHEMA_VERSION,
    APPLICATION_READY,
    PARAMETER_REGISTRY_VERSION,
    PROFILE_SCHEMA_VERSION,
    PROMPT_VERSION,
    TAXONOMY_VERSION,
    VALIDATION_SCOPE,
)

ProgressFn = Callable[[str], None]


def run(
    input_dir: str,
    build_root: str,
    *,
    config_path: str | None = None,
    mock_dir: str | None = None,
    overrides: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    started_at = now_utc_iso()
    clock = time.perf_counter()
    log = progress or (lambda _message: None)

    input_dir = os.path.abspath(input_dir)
    build_root = os.path.abspath(build_root)
    user_config = load_config_file(config_path) if config_path else {}
    cfg = resolve_config(user_config, overrides)
    cfg_hash = config_hash(cfg)

    _validate_build_root(input_dir, build_root, cfg, cfg_hash)
    os.makedirs(build_root, exist_ok=True)

    intake = load_input(input_dir)
    build_id = _build_id(intake.fingerprint)
    log(
        f"输入 {intake.ready_count} 张 ready（共 {intake.manifest_total} 条），"
        f"input_fingerprint={intake.fingerprint[:12]}"
    )

    store = BuildStore(build_root)
    cache = ResponseCache(store.cache_dir)
    episodes = EpisodeStore(build_id=build_id, path=os.path.join(build_root, "episodes.jsonl"))
    try:
        adapter = build_adapter(cfg["adapter"], mock_dir=mock_dir)
    except Exception as exc:
        raise FatalError(FatalCode.CONFIG_INVALID, f"模型适配器配置失败：{exc}") from exc
    runner = AnalysisRunner(
        adapter=adapter,
        cache=cache,
        store=store,
        episodes=episodes,
        cfg=cfg,
        cfg_hash=cfg_hash,
        build_id=build_id,
    )

    results = []
    assignments: list[dict] = []
    for index, image in enumerate(intake.images, start=1):
        result = runner.run(image)
        results.append(result)
        if result.status == AnalysisStatus.NORMALIZED and result.payload is not None:
            assignment = assign_scene(result.payload, cfg)
            assignments.append(assignment)
        detail = result.error_code or result.status
        log(f"[{index}/{intake.ready_count}] {image.image_id} {detail}")

    assignments.sort(key=lambda item: item["image_id"])
    split = build_evaluation_split(assignments, intake.near_duplicate_groups, cfg)
    split["validation_scope"] = VALIDATION_SCOPE
    split["application_ready"] = APPLICATION_READY
    split_side = _split_side_map(split)

    manifest_rows = _build_manifest_rows(intake, results, assignments, split_side, build_root)
    write_jsonl_atomic(os.path.join(build_root, "manifest.jsonl"), manifest_rows)
    write_jsonl_atomic(os.path.join(build_root, "assignments.jsonl"), assignments)
    atomic_write_json(os.path.join(build_root, "evaluation_split.json"), split)
    episodes.write()
    write_resolved_config(
        os.path.join(build_root, "config.resolved.json"),
        cfg,
        cfg_hash,
        {"input_dir": input_dir, "build_root": build_root, "mock_dir": os.path.abspath(mock_dir) if mock_dir else None},
    )

    counts = _status_counts(results)
    coverage = counts["normalized"] / max(1, intake.ready_count)
    summary = assignment_summary(assignments)
    warnings = list(intake.warnings)
    if coverage < float(cfg["analysis_coverage_min"]):
        warnings.append(
            f"analysis_coverage {coverage:.3f} 低于阈值 {cfg['analysis_coverage_min']}，不得发布正式档案"
        )
    if adapter.model_info.is_mock:
        warnings.append("model_info.is_mock=true：本轮为 Mock 分析，不能作为真实风格证据发布")
    elif adapter.model_info.provider == "agent":
        warnings.append(
            "agent_in_the_loop：响应由宿主会话生成，非固定模型端点；"
            "仅能依赖保存的原始响应回放，换会话不保证一致"
        )

    report = {
        "build_id": build_id,
        "started_at": started_at,
        "finished_at": now_utc_iso(),
        "duration_seconds": round(time.perf_counter() - clock, 3),
        "input_dir": input_dir,
        "build_root": build_root,
        "input_fingerprint": intake.fingerprint,
        "preprocess": {
            "preprocess_version": intake.versions["preprocess_version"],
            "feature_schema_version": intake.versions["feature_schema_version"],
            "config_hash": intake.preprocess_config_hash,
            "manifest_total": intake.manifest_total,
            "non_ready": intake.non_ready,
        },
        "analysis": {
            "adapter": adapter.model_info.as_dict(),
            "prompt_version": PROMPT_VERSION,
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "counts": counts,
            "coverage": round(coverage, 6),
            "coverage_min": float(cfg["analysis_coverage_min"]),
            "coverage_ok": coverage >= float(cfg["analysis_coverage_min"]),
        },
        "assignments": summary,
        "split": {
            "split_hash": split["split_hash"],
            "split_seed": split["split_seed"],
            "buckets": {
                bucket_id: {
                    "status": entry["status"],
                    "split_status": entry["split_status"],
                    "n_core": entry["n_core"],
                    "n_train": entry["n_train"],
                    "n_holdout": entry["n_holdout"],
                }
                for bucket_id, entry in split["buckets"].items()
            },
        },
        "episodes": len(episodes.rows),
        "config_hash": cfg_hash,
        "versions": {
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "parameter_registry_version": PARAMETER_REGISTRY_VERSION,
            "taxonomy_hash": taxonomy_hash(),
            "parameter_registry_hash": registry_hash(),
        },
        "validation_scope": VALIDATION_SCOPE,
        "application_ready": APPLICATION_READY,
        "warnings": warnings,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    atomic_write_json(os.path.join(build_root, "run_report.json"), report)

    return {
        "build_id": build_id,
        "build_root": build_root,
        "report": report,
        "manifest_path": os.path.join(build_root, "manifest.jsonl"),
        "assignments_path": os.path.join(build_root, "assignments.jsonl"),
        "split_path": os.path.join(build_root, "evaluation_split.json"),
        "report_path": os.path.join(build_root, "run_report.json"),
    }


def _validate_build_root(input_dir: str, build_root: str, cfg: dict, cfg_hash: str) -> None:
    input_key = os.path.normcase(input_dir)
    build_key = os.path.normcase(build_root)
    if input_key == build_key:
        raise FatalError(FatalCode.BUILD_ROOT_INVALID, "build_root 不得等于输入目录")
    try:
        if os.path.commonpath([input_key, build_key]) == input_key:
            raise FatalError(FatalCode.BUILD_ROOT_INVALID, "build_root 不得位于输入目录内")
    except ValueError:
        pass

    if not os.path.isdir(build_root):
        return
    entries = os.listdir(build_root)
    if not entries:
        return
    config_path = os.path.join(build_root, "config.resolved.json")
    if not os.path.isfile(config_path):
        raise FatalError(
            FatalCode.BUILD_ROOT_INVALID,
            f"build_root 非空但没有 config.resolved.json，拒绝写入：{build_root}",
        )
    try:
        previous = read_json(config_path)
    except Exception as exc:
        raise FatalError(FatalCode.BUILD_STATE_INCOMPATIBLE, f"旧配置不可读：{exc}") from exc
    if (
        previous.get("config_hash") != cfg_hash
        or previous.get("versions", {}).get("prompt_version") != PROMPT_VERSION
        or previous.get("versions", {}).get("analysis_schema_version") != ANALYSIS_SCHEMA_VERSION
        or previous.get("versions", {}).get("taxonomy_version") != TAXONOMY_VERSION
        or previous.get("versions", {}).get("parameter_registry_version") != PARAMETER_REGISTRY_VERSION
    ) and not cfg["overwrite"]:
        raise FatalError(
            FatalCode.BUILD_STATE_INCOMPATIBLE,
            "build_root 中的旧构建与当前配置/版本不一致；请换目录或加 --overwrite，"
            "禁止把新旧分析响应拼进同一次构建。",
        )


def _build_id(fingerprint: str) -> str:
    stamp = now_utc_iso().replace("-", "").replace(":", "").replace("T", "_").replace("Z", "Z")
    return f"build_{stamp}_{fingerprint[:8]}"


def _status_counts(results) -> dict[str, int]:
    counts = {"normalized": 0, "invalid": 0, "error": 0}
    for result in results:
        if result.status == AnalysisStatus.NORMALIZED:
            counts["normalized"] += 1
        elif result.status == AnalysisStatus.INVALID:
            counts["invalid"] += 1
        else:
            counts["error"] += 1
    return counts


def _split_side_map(split: dict) -> dict[str, str]:
    side: dict[str, str] = {}
    for entry in split["buckets"].values():
        for image_id in entry["train"]:
            side[image_id] = "train"
        for image_id in entry["holdout"]:
            side[image_id] = "holdout"
    return side


def _build_manifest_rows(intake, results, assignments, split_side, build_root: str) -> list[dict]:
    by_id = {assignment["image_id"]: assignment for assignment in assignments}
    rows: list[dict] = []
    for image, result in zip(intake.images, results):
        assignment = by_id.get(image.image_id)
        row = {
            "image_id": image.image_id,
            "source_sha256": image.source_sha256,
            "preview_path": image.preview_rel,
            "feature_path": image.feature_rel,
            "analysis_status": result.status,
            "cached": result.cached,
            "attempts": result.attempts,
            "assignment_level": assignment["level"] if assignment else None,
            "bucket_id": assignment["bucket_id"] if assignment else None,
            "split": split_side.get(image.image_id),
            "warnings": result.warnings,
            "error_code": result.error_code,
            "raw_path": relative_posix(
                os.path.join(build_root, "analyses", "raw", f"{image.image_id}.json"), build_root
            ),
            "normalized_path": (
                relative_posix(
                    os.path.join(build_root, "analyses", "normalized", f"{image.image_id}.json"),
                    build_root,
                )
                if result.status == AnalysisStatus.NORMALIZED
                else None
            ),
        }
        rows.append(row)
    return rows

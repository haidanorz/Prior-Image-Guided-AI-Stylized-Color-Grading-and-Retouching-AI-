"""Agent 工具编排：Python 出任务，agent/子代理看图，提交 JSON，Python 收口。

状态目录：
    <build_root>/mcp/state.json
    <build_root>/mcp/tasks.jsonl
    <build_root>/mcp/prompts/<image_id>.txt
"""

from __future__ import annotations

import json
import os
from typing import Any

from .analysis.adapter import ModelInfo
from .analysis.normalize import normalize_analysis
from .analysis.prompt import build_prompt
from .bucketing import assign_scene, assignment_summary
from .config import VERSIONS, config_hash, resolve_config, write_resolved_config
from .episodes import EpisodeStore
from .errors import AnalysisStatus, ErrorCode, FatalCode, FatalError, EpisodeType, InvalidAnalysis
from .intake import IntakeImage, image_fingerprint, load_input
from .splitting import build_evaluation_split
from .store import BuildStore
from .util import (
    atomic_write_json,
    now_utc_iso,
    read_json,
    read_jsonl,
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

FLOW_VERSION = "1.0.0"
DEFAULT_AGENT_MODEL = "agent-vision"
DEFAULT_AGENT_VERSION = "2026-11-04"


def start_build(
    input_dir: str,
    build_root: str,
    *,
    limit: int = 0,
    overwrite: bool = False,
    agent_model: str = DEFAULT_AGENT_MODEL,
    agent_version: str = DEFAULT_AGENT_VERSION,
) -> dict[str, Any]:
    input_dir = os.path.abspath(input_dir)
    build_root = os.path.abspath(build_root)
    if not os.path.isdir(input_dir):
        raise FatalError(FatalCode.INPUT_INVALID, f"输入目录不存在：{input_dir}")
    if os.path.normcase(input_dir) == os.path.normcase(build_root):
        raise FatalError(FatalCode.BUILD_ROOT_INVALID, "build_root 不得等于输入目录")
    try:
        if os.path.commonpath([os.path.normcase(input_dir), os.path.normcase(build_root)]) == os.path.normcase(input_dir):
            raise FatalError(FatalCode.BUILD_ROOT_INVALID, "build_root 不得位于输入目录内")
    except ValueError:
        pass
    if limit < 0:
        raise FatalError(FatalCode.CONFIG_INVALID, "limit 必须 >= 0")

    cfg = resolve_config({"adapter": "agent"})
    cfg_hash = config_hash(cfg)
    agent_info = {
        "provider": "agent",
        "model": agent_model,
        "version": agent_version,
        "is_mock": False,
    }

    mcp_dir = os.path.join(build_root, "mcp")
    state_path = os.path.join(mcp_dir, "state.json")
    if os.path.isfile(state_path):
        state = read_json(state_path)
        same = (
            state.get("input_dir") == input_dir
            and state.get("config_hash") == cfg_hash
            and state.get("limit") == limit
            and state.get("agent_model_info") == agent_info
        )
        if state.get("status") == "finalized" and not overwrite:
            raise FatalError(FatalCode.BUILD_STATE_INCOMPATIBLE, "该 build_root 已 finalize，请换新目录")
        if same:
            return _start_summary(state, read_jsonl(os.path.join(mcp_dir, "tasks.jsonl")), resumed=True)
        if not overwrite:
            raise FatalError(
                FatalCode.BUILD_STATE_INCOMPATIBLE,
                "build_root 已有不同输入/配置/limit 的 agent 构建；请换目录或 overwrite=true",
            )
    elif os.path.isdir(build_root) and os.listdir(build_root):
        if not overwrite:
            raise FatalError(
                FatalCode.BUILD_ROOT_INVALID,
                f"build_root 非空且不是 agent 构建，拒绝混用：{build_root}",
            )

    os.makedirs(mcp_dir, exist_ok=True)
    os.makedirs(os.path.join(mcp_dir, "prompts"), exist_ok=True)
    intake = load_input(input_dir)
    images = intake.images[:limit] if limit else intake.images
    if not images:
        raise FatalError(FatalCode.INPUT_INVALID, "没有可分配的任务")

    build_id = f"agent_{_stamp()}_{intake.fingerprint[:8]}"
    store = BuildStore(build_root)
    tasks: list[dict] = []
    for image in images:
        fingerprint = image_fingerprint(
            image,
            analysis_config_hash=cfg_hash,
            prompt_version=PROMPT_VERSION,
            model_info=agent_info,
            schema_version=ANALYSIS_SCHEMA_VERSION,
        )
        prompt = build_prompt(image, input_fingerprint=fingerprint, cfg=cfg)
        prompt_path = os.path.join(mcp_dir, "prompts", f"{image.image_id}.txt")
        _write_text(prompt_path, prompt.text)
        tasks.append(
            {
                "image_id": image.image_id,
                "source_path": image.source_path,
                "source_sha256": image.source_sha256,
                "preview_path": image.preview_path,
                "preview_rel": image.preview_rel,
                "feature_path": image.feature_path,
                "feature_rel": image.feature_rel,
                "preview_sha256": image.preview_sha256,
                "feature_sha256": image.feature_sha256,
                "input_fingerprint": fingerprint,
                "prompt_path": prompt_path,
                "prompt_sha256": prompt.sha256,
                "prompt_version": PROMPT_VERSION,
                "status": "pending",
                "attempts": 0,
                "error_code": None,
                "error_message": None,
                "details": [],
                "warnings": [],
                "normalized_path": None,
                "submitted_at": None,
            }
        )

    write_jsonl_atomic(os.path.join(mcp_dir, "tasks.jsonl"), tasks)
    state = {
        "flow_version": FLOW_VERSION,
        "build_id": build_id,
        "status": "open",
        "input_dir": input_dir,
        "build_root": build_root,
        "input_fingerprint": intake.fingerprint,
        "limit": limit,
        "task_count": len(tasks),
        "config": cfg,
        "config_hash": cfg_hash,
        "versions": VERSIONS,
        "agent_model_info": agent_info,
        "preprocess": {
            "manifest_total": intake.manifest_total,
            "non_ready": intake.non_ready,
            "preprocess_version": intake.versions["preprocess_version"],
            "feature_schema_version": intake.versions["feature_schema_version"],
            "config_hash": intake.preprocess_config_hash,
        },
        "near_duplicate_groups": intake.near_duplicate_groups,
        "created_at": now_utc_iso(),
        "updated_at": now_utc_iso(),
    }
    atomic_write_json(state_path, state)
    write_resolved_config(
        os.path.join(build_root, "config.resolved.json"),
        cfg,
        cfg_hash,
        {"input_dir": input_dir, "build_root": build_root, "flow": "agent"},
    )
    _ensure_episodes_file(build_root, build_id)
    return _start_summary(state, tasks, resumed=False)


def list_tasks(build_root: str, *, status: str = "open", limit: int = 3) -> dict[str, Any]:
    state, tasks = _load_open_state(build_root)
    status = status.lower()
    if status == "open":
        selected = [task for task in tasks if task["status"] in ("pending", "invalid")]
    elif status == "all":
        selected = tasks
    elif status in ("pending", "invalid", "submitted"):
        selected = [task for task in tasks if task["status"] == status]
    else:
        raise FatalError(FatalCode.CONFIG_INVALID, f"未知 status：{status}")
    selected = sorted(selected, key=lambda item: item["image_id"])
    limit = max(1, min(int(limit), 10))
    batch = selected[:limit]
    return {
        "ok": True,
        "build_id": state["build_id"],
        "status": state["status"],
        "counts": _task_counts(tasks),
        "remaining": len(selected),
        "tasks": [
            {
                "image_id": task["image_id"],
                "preview_path": task["preview_path"],
                "feature_path": task["feature_path"],
                "prompt_path": task["prompt_path"],
                "input_fingerprint": task["input_fingerprint"],
                "status": task["status"],
                "attempts": task["attempts"],
                "last_error": task["error_message"],
            }
            for task in batch
        ],
        "submission_shape": {
            "submissions": [
                {
                    "image_id": "<from task>",
                    "analysis": {
                        "scene_observation": "{...}",
                        "visual_observation": "{...}",
                        "edit_hypotheses": "{...}",
                        "quality": "{...}",
                        "local_adjustments_suspected": [],
                        "overall_impression": "...",
                        "warnings": [],
                    },
                }
            ],
            "note": "analysis 可省略身份字段；profile_submit 会注入 image_id/source_sha256/input_fingerprint/版本/model_info",
        },
    }


def submit_batch(build_root: str, submissions: Any, *, replace: bool = False) -> dict[str, Any]:
    state, tasks = _load_open_state(build_root)
    items = _coerce_submissions(submissions)
    if not items:
        raise FatalError(FatalCode.CONFIG_INVALID, "submissions 不能为空")
    if len(items) > 10:
        raise FatalError(FatalCode.CONFIG_INVALID, "单次最多提交 10 张，避免超时")

    cfg = resolve_config(state["config"])
    agent_info = ModelInfo(
        provider=state["agent_model_info"]["provider"],
        model=state["agent_model_info"]["model"],
        version=state["agent_model_info"]["version"],
        is_mock=bool(state["agent_model_info"].get("is_mock", False)),
    )
    task_by_id = {task["image_id"]: task for task in tasks}
    store = BuildStore(build_root)
    episodes = EpisodeStore(build_id=state["build_id"], path=os.path.join(build_root, "episodes.jsonl"))
    results: list[dict] = []

    for item in items:
        image_id = item.get("image_id")
        task = task_by_id.get(image_id)
        if task is None:
            results.append({"image_id": image_id, "accepted": False, "error": "unknown image_id"})
            continue
        if task["status"] == "submitted" and not replace:
            results.append({"image_id": image_id, "accepted": False, "error": "already submitted"})
            continue
        try:
            analysis = item.get("analysis")
            if isinstance(analysis, str):
                analysis = json.loads(analysis)
            if not isinstance(analysis, dict):
                raise ValueError("analysis 必须是对象")
            image = _image_from_task(task)
            payload, overridden = _assemble_payload(analysis, task, agent_info)
            normalized = normalize_analysis(
                json.dumps(payload, ensure_ascii=False),
                image=image,
                expected_fingerprint=task["input_fingerprint"],
                config=cfg,
                model_info=agent_info,
            )
        except InvalidAnalysis as exc:
            task["status"] = "invalid"
            task["attempts"] += 1
            task["error_code"] = ErrorCode.ANALYSIS_INVALID
            task["error_message"] = exc.message
            task["details"] = exc.details[:20]
            episodes.add(
                EpisodeType.ANALYSIS_INVALID,
                image_id=image_id,
                context={"source": "agent_submission", "details": exc.details[:20]},
                action="profile_submit",
                result="rejected",
            )
            results.append(
                {
                    "image_id": image_id,
                    "accepted": False,
                    "error": exc.message,
                    "details": exc.details[:20],
                }
            )
            continue
        except Exception as exc:
            task["status"] = "invalid"
            task["attempts"] += 1
            task["error_code"] = ErrorCode.INTERNAL_ERROR
            task["error_message"] = f"{type(exc).__name__}: {exc}"
            task["details"] = []
            results.append({"image_id": image_id, "accepted": False, "error": task["error_message"]})
            continue

        raw_path = store.write_raw(
            image_id,
            {
                "image_id": image_id,
                "input_fingerprint": task["input_fingerprint"],
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": task["prompt_sha256"],
                "model_info": agent_info.as_dict(),
                "source": "agent_submission",
                "attempts": [
                    {
                        "attempt": task["attempts"] + 1,
                        "cached": False,
                        "text": json.dumps(analysis, ensure_ascii=False),
                        "error": None,
                    }
                ],
                "submitted_at": now_utc_iso(),
            },
        )
        normalized_path = store.write_normalized(image_id, normalized.payload)
        previous_invalid = task["status"] == "invalid" or task["attempts"] > 0
        task["status"] = "submitted"
        task["attempts"] += 1
        task["error_code"] = None
        task["error_message"] = None
        task["details"] = []
        task["warnings"] = normalized.warnings
        task["normalized_path"] = relative_posix(normalized_path, build_root)
        task["raw_path"] = relative_posix(raw_path, build_root)
        task["submitted_at"] = now_utc_iso()
        if previous_invalid:
            episodes.add(
                EpisodeType.ANALYSIS_RECOVERED,
                image_id=image_id,
                context={"attempt": task["attempts"]},
                action="profile_submit",
                result="accepted_after_correction",
            )
        results.append(
            {
                "image_id": image_id,
                "accepted": True,
                "warnings": normalized.warnings,
                "identity_fields_overridden": overridden,
                "normalized_path": task["normalized_path"],
            }
        )

    write_jsonl_atomic(os.path.join(build_root, "mcp", "tasks.jsonl"), tasks)
    episodes.write()
    _touch_state(build_root, state)
    return {
        "ok": True,
        "build_id": state["build_id"],
        "counts": _task_counts(tasks),
        "results": results,
    }


def finalize_build(build_root: str) -> dict[str, Any]:
    state, tasks = _load_open_state(build_root)
    cfg = resolve_config(state["config"])
    payloads: dict[str, dict] = {}
    for task in tasks:
        if task["status"] == "submitted" and task.get("normalized_path"):
            path = os.path.join(build_root, task["normalized_path"])
            if os.path.isfile(path):
                payloads[task["image_id"]] = read_json(path)

    assignments = [assign_scene(payload, cfg) for payload in payloads.values()]
    assignments.sort(key=lambda item: item["image_id"])
    split = build_evaluation_split(assignments, state["near_duplicate_groups"], cfg)
    split["validation_scope"] = VALIDATION_SCOPE
    split["application_ready"] = APPLICATION_READY
    split_side = _split_side_map(split)
    assignment_by_id = {item["image_id"]: item for item in assignments}

    manifest_rows = []
    for task in tasks:
        assignment = assignment_by_id.get(task["image_id"])
        if task["image_id"] in payloads:
            status = AnalysisStatus.NORMALIZED
            error_code = None
        elif task["status"] == "invalid":
            status = AnalysisStatus.INVALID
            error_code = task.get("error_code") or ErrorCode.ANALYSIS_INVALID
        else:
            status = AnalysisStatus.ERROR
            error_code = "NOT_SUBMITTED"
        manifest_rows.append(
            {
                "image_id": task["image_id"],
                "source_sha256": task["source_sha256"],
                "preview_path": task["preview_rel"],
                "feature_path": task["feature_rel"],
                "analysis_status": status,
                "cached": False,
                "attempts": task["attempts"],
                "assignment_level": assignment["level"] if assignment else None,
                "bucket_id": assignment["bucket_id"] if assignment else None,
                "split": split_side.get(task["image_id"]),
                "warnings": task.get("warnings", []),
                "error_code": error_code,
                "raw_path": task.get("raw_path"),
                "normalized_path": task.get("normalized_path"),
            }
        )

    write_jsonl_atomic(os.path.join(build_root, "manifest.jsonl"), manifest_rows)
    write_jsonl_atomic(os.path.join(build_root, "assignments.jsonl"), assignments)
    atomic_write_json(os.path.join(build_root, "evaluation_split.json"), split)

    counts = {"normalized": 0, "invalid": 0, "error": 0}
    for row in manifest_rows:
        counts[row["analysis_status"]] = counts.get(row["analysis_status"], 0) + 1
    coverage = counts["normalized"] / max(1, len(tasks))
    warnings = []
    if coverage < float(cfg["analysis_coverage_min"]):
        warnings.append(f"analysis_coverage {coverage:.3f} 低于阈值 {cfg['analysis_coverage_min']}")
    warnings.append(
        "agent_in_the_loop：分析由宿主 agent/子代理生成，非固定模型端点；"
        "仅能依赖保存的原始提交回放"
    )
    report = {
        "build_id": state["build_id"],
        "started_at": state["created_at"],
        "finished_at": now_utc_iso(),
        "duration_seconds": None,
        "input_dir": state["input_dir"],
        "build_root": build_root,
        "input_fingerprint": state["input_fingerprint"],
        "preprocess": {
            "manifest_total": len(tasks),
            "non_ready": 0,
            "preprocess_version": state["preprocess"]["preprocess_version"],
            "feature_schema_version": state["preprocess"]["feature_schema_version"],
            "config_hash": state["preprocess"]["config_hash"],
        },
        "analysis": {
            "adapter": state["agent_model_info"],
            "prompt_version": PROMPT_VERSION,
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "counts": counts,
            "coverage": round(coverage, 6),
            "coverage_min": float(cfg["analysis_coverage_min"]),
            "coverage_ok": coverage >= float(cfg["analysis_coverage_min"]),
        },
        "assignments": assignment_summary(assignments),
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
        "episodes": _episode_count(build_root),
        "config_hash": state["config_hash"],
        "versions": {
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "parameter_registry_version": PARAMETER_REGISTRY_VERSION,
        },
        "validation_scope": VALIDATION_SCOPE,
        "application_ready": APPLICATION_READY,
        "warnings": warnings,
        "software": {"python": _python_version()},
    }
    atomic_write_json(os.path.join(build_root, "run_report.json"), report)
    state["status"] = "finalized"
    state["finalized_at"] = now_utc_iso()
    state["updated_at"] = now_utc_iso()
    atomic_write_json(os.path.join(build_root, "mcp", "state.json"), state)

    return {
        "ok": True,
        "build_id": state["build_id"],
        "counts": counts,
        "coverage": report["analysis"]["coverage"],
        "assignments": report["assignments"],
        "split": report["split"],
        "warnings": warnings,
        "build_root": build_root,
    }


def preview_build(build_root: str) -> dict[str, Any]:
    """只读检查点：用已提交分析试算分桶与划分，不写任何文件、不 finalize。"""
    build_root = os.path.abspath(build_root)
    state = read_json(os.path.join(build_root, "mcp", "state.json"))
    tasks = read_jsonl(os.path.join(build_root, "mcp", "tasks.jsonl"))
    cfg = resolve_config(state["config"])
    assignments = []
    for task in tasks:
        if task["status"] == "submitted" and task.get("normalized_path"):
            path = os.path.join(build_root, task["normalized_path"])
            if os.path.isfile(path):
                assignments.append(assign_scene(read_json(path), cfg))
    assignments.sort(key=lambda item: item["image_id"])
    split = build_evaluation_split(assignments, state["near_duplicate_groups"], cfg)
    return {
        "ok": True,
        "build_id": state["build_id"],
        "status": state["status"],
        "counts": _task_counts(tasks),
        "assignments": assignment_summary(assignments),
        "buckets": {
            bucket_id: {
                "status": entry["status"],
                "split_status": entry["split_status"],
                "n_core": entry["n_core"],
                "holdout_target": entry["holdout_target"],
                "reason": entry["reason"],
            }
            for bucket_id, entry in split["buckets"].items()
        },
        "split_hash": split["split_hash"],
        "note": "只读试算；未写入 assignments/evaluation_split，也未 finalize",
    }


def build_status(build_root: str) -> dict[str, Any]:
    state_path = os.path.join(os.path.abspath(build_root), "mcp", "state.json")
    if not os.path.isfile(state_path):
        raise FatalError(FatalCode.BUILD_ROOT_INVALID, f"不是 agent 构建：{build_root}")
    state = read_json(state_path)
    tasks = read_jsonl(os.path.join(os.path.abspath(build_root), "mcp", "tasks.jsonl"))
    return {
        "ok": True,
        "build_id": state["build_id"],
        "status": state["status"],
        "task_count": state["task_count"],
        "counts": _task_counts(tasks),
        "created_at": state["created_at"],
        "updated_at": state["updated_at"],
    }


def _load_open_state(build_root: str) -> tuple[dict, list[dict]]:
    build_root = os.path.abspath(build_root)
    state_path = os.path.join(build_root, "mcp", "state.json")
    if not os.path.isfile(state_path):
        raise FatalError(FatalCode.BUILD_ROOT_INVALID, f"缺少 agent state：{state_path}")
    state = read_json(state_path)
    if state.get("status") != "open":
        raise FatalError(FatalCode.BUILD_STATE_INCOMPATIBLE, "构建已 finalize，不能继续提交")
    tasks = read_jsonl(os.path.join(build_root, "mcp", "tasks.jsonl"))
    return state, tasks


def _image_from_task(task: dict) -> IntakeImage:
    feature = read_json(task["feature_path"])
    return IntakeImage(
        image_id=task["image_id"],
        source_path=task["source_path"],
        source_sha256=task["source_sha256"],
        preview_path=task["preview_path"],
        preview_rel=task["preview_rel"],
        feature_path=task["feature_path"],
        feature_rel=task["feature_rel"],
        preview_sha256=task["preview_sha256"],
        feature_sha256=task["feature_sha256"],
        feature=feature,
        record={},
    )


def _assemble_payload(analysis: dict, task: dict, agent_info: ModelInfo) -> tuple[dict, list[str]]:
    payload = dict(analysis)
    identity = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "image_id": task["image_id"],
        "source_sha256": task["source_sha256"],
        "input_fingerprint": task["input_fingerprint"],
        "prompt_version": PROMPT_VERSION,
        "model_info": agent_info.as_dict(),
    }
    overridden: list[str] = []
    for key, value in identity.items():
        if payload.get(key) not in (None, value):
            overridden.append(key)
        payload[key] = value
    return payload, overridden


def _coerce_submissions(value: Any) -> list[dict]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise FatalError(FatalCode.CONFIG_INVALID, "submissions 必须是数组")
    return value


def _task_counts(tasks: list[dict]) -> dict[str, int]:
    counts = {"pending": 0, "invalid": 0, "submitted": 0}
    for task in tasks:
        counts[task["status"]] = counts.get(task["status"], 0) + 1
    return counts


def _split_side_map(split: dict) -> dict[str, str]:
    side: dict[str, str] = {}
    for entry in split["buckets"].values():
        for image_id in entry["train"]:
            side[image_id] = "train"
        for image_id in entry["holdout"]:
            side[image_id] = "holdout"
    return side


def _start_summary(state: dict, tasks: list[dict], *, resumed: bool) -> dict[str, Any]:
    counts = _task_counts(tasks)
    return {
        "ok": True,
        "resumed": resumed,
        "build_id": state["build_id"],
        "build_root": state["build_root"],
        "task_count": state["task_count"],
        "counts": counts,
        "next": [
            {
                "image_id": task["image_id"],
                "preview_path": task["preview_path"],
                "prompt_path": task["prompt_path"],
            }
            for task in tasks[:3]
        ],
        "usage": "调用 profile_tasks 取任务；分析后调用 profile_submit；全部提交后 profile_finalize",
    }


def _ensure_episodes_file(build_root: str, build_id: str) -> None:
    path = os.path.join(build_root, "episodes.jsonl")
    if not os.path.exists(path):
        EpisodeStore(build_id=build_id, path=path).write()


def _episode_count(build_root: str) -> int:
    path = os.path.join(build_root, "episodes.jsonl")
    return len(read_jsonl(path)) if os.path.isfile(path) else 0


def _touch_state(build_root: str, state: dict) -> None:
    state["updated_at"] = now_utc_iso()
    atomic_write_json(os.path.join(build_root, "mcp", "state.json"), state)


def _write_text(path: str, text: str) -> None:
    from .util import atomic_write_bytes

    atomic_write_bytes(path, text.encode("utf-8"))


def _stamp() -> str:
    return now_utc_iso().replace("-", "").replace(":", "").replace("T", "_").replace("Z", "Z")


def _python_version() -> str:
    import platform

    return platform.python_version()

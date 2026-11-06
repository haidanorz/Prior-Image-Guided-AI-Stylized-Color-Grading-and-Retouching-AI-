"""配置：默认值、校验、解析与配置哈希。

阈值全部集中在这里；prompt 与代码不得再写第二套数字。
路径属于运行时参数，不进入 config_hash；影响输出的行为参数与版本进入。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .errors import FatalCode, FatalError
from .util import atomic_write_json, canonical_json
from .version import (
    ANALYSIS_SCHEMA_VERSION,
    PARAMETER_REGISTRY_VERSION,
    PROMPT_VERSION,
    TAXONOMY_VERSION,
)

DEFAULTS: dict[str, Any] = {
    "core_top_score": 0.65,
    "core_margin": 0.15,
    "associate_top_score": 0.45,
    "full_bucket_min_core": 8,
    "provisional_bucket_min_core": 5,
    "parameter_min_confidence": 0.30,
    "profile_shape": "bucket_primary",
    "validation_mode": "kfold",
    "cv_folds": 10,
    "bootstrap_iterations": 200,
    "holdout_ratio": 0.20,
    "split_seed": 0,
    "analysis_max_retries": 2,
    "analysis_coverage_min": 0.90,
    "analysis_min_mean_hypotheses": 6.0,
    "analysis_min_key_coverage": 0.65,
    "brightness_dark_max_p50": 0.25,
    "brightness_bright_min_p50": 0.55,
    "adapter": "mock",
    "overwrite": False,
    "experiment_feedback_enabled": False,
    "experiment_skill_library_enabled": False,
}

_FLOAT_KEYS = {
    "core_top_score",
    "core_margin",
    "associate_top_score",
    "parameter_min_confidence",
    "holdout_ratio",
    "analysis_coverage_min",
    "analysis_min_mean_hypotheses",
    "analysis_min_key_coverage",
    "brightness_dark_max_p50",
    "brightness_bright_min_p50",
}
_INT_KEYS = {
    "cv_folds",
    "bootstrap_iterations",
    "full_bucket_min_core",
    "provisional_bucket_min_core",
    "split_seed",
    "analysis_max_retries",
}
_BOOL_KEYS = {"overwrite", "experiment_feedback_enabled", "experiment_skill_library_enabled"}
_ALLOWED = set(DEFAULTS)

VERSIONS = {
    "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
    "prompt_version": PROMPT_VERSION,
    "taxonomy_version": TAXONOMY_VERSION,
    "parameter_registry_version": PARAMETER_REGISTRY_VERSION,
}


def _fail(message: str) -> None:
    raise FatalError(FatalCode.CONFIG_INVALID, message)


def load_config_file(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        _fail(f"配置文件不存在：{path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        _fail(f"配置文件不是合法 JSON：{path}：{exc}")
    if not isinstance(data, dict):
        _fail(f"配置文件顶层必须是对象：{path}")
    return data


def resolve_config(user_config: dict[str, Any] | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULTS)
    for source in (user_config or {}, overrides or {}):
        for key, value in source.items():
            if key not in _ALLOWED:
                _fail(f"未知配置项：{key}（允许项：{', '.join(sorted(_ALLOWED))}）")
            merged[key] = value

    resolved: dict[str, Any] = {}
    for key, value in merged.items():
        if key in _BOOL_KEYS:
            if not isinstance(value, bool):
                _fail(f"{key} 必须是布尔值，当前为 {value!r}")
            resolved[key] = value
        elif key in _INT_KEYS:
            if isinstance(value, bool) or not isinstance(value, int):
                _fail(f"{key} 必须是整数，当前为 {value!r}")
            resolved[key] = value
        elif key in _FLOAT_KEYS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _fail(f"{key} 必须是数值，当前为 {value!r}")
            resolved[key] = float(value)
        else:
            resolved[key] = value

    if not 0.0 <= resolved["associate_top_score"] < resolved["core_top_score"] <= 1.0:
        _fail("必须满足 0 <= associate_top_score < core_top_score <= 1")
    if not 0.0 <= resolved["core_margin"] <= 1.0:
        _fail("core_margin 必须在 0..1")
    if resolved["provisional_bucket_min_core"] < 2:
        _fail("provisional_bucket_min_core 必须 >= 2")
    if resolved["full_bucket_min_core"] <= resolved["provisional_bucket_min_core"]:
        _fail("full_bucket_min_core 必须大于 provisional_bucket_min_core")
    if not 0.0 <= resolved["parameter_min_confidence"] <= 1.0:
        _fail("parameter_min_confidence 必须在 0..1")
    if not 0.0 < resolved["holdout_ratio"] < 1.0:
        _fail("holdout_ratio 必须在 (0,1)")
    if resolved["profile_shape"] not in ("bucket_primary", "global_plus_delta"):
        _fail("profile_shape 只能是 bucket_primary 或 global_plus_delta")
    if resolved["validation_mode"] not in ("kfold", "holdout"):
        _fail("validation_mode 只能是 kfold 或 holdout")
    if not 2 <= resolved["cv_folds"] <= 20:
        _fail("cv_folds 必须在 2..20")
    if not 0 <= resolved["bootstrap_iterations"] <= 5000:
        _fail("bootstrap_iterations 必须在 0..5000")
    if not 0 <= resolved["split_seed"] <= 2**32 - 1:
        _fail("split_seed 必须在 0..2^32-1")
    if not 0 <= resolved["analysis_max_retries"] <= 5:
        _fail("analysis_max_retries 必须在 0..5")
    if not 0.0 <= resolved["analysis_coverage_min"] <= 1.0:
        _fail("analysis_coverage_min 必须在 0..1")
    if resolved["analysis_min_mean_hypotheses"] < 0:
        _fail("analysis_min_mean_hypotheses 必须 >= 0")
    if not 0.0 <= resolved["analysis_min_key_coverage"] <= 1.0:
        _fail("analysis_min_key_coverage 必须在 0..1")
    if not 0.0 < resolved["brightness_dark_max_p50"] < resolved["brightness_bright_min_p50"] < 1.0:
        _fail("必须满足 0 < brightness_dark_max_p50 < brightness_bright_min_p50 < 1")
    if resolved["adapter"] not in ("mock", "replay", "agent"):
        _fail(
            "adapter 只支持 mock / replay / agent；接真实模型前请先实现对应 adapter 并在这里登记，"
            f"当前为 {resolved['adapter']!r}"
        )

    return resolved


def config_hash(resolved: dict[str, Any]) -> str:
    # overwrite 只影响写入策略，不改变输出口径，不参与哈希
    hashed = {key: value for key, value in resolved.items() if key != "overwrite"}
    payload = {"config": hashed, "versions": VERSIONS}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def write_resolved_config(path: str, resolved: dict[str, Any], cfg_hash: str, runtime: dict[str, Any]) -> None:
    atomic_write_json(
        path,
        {
            "config": resolved,
            "versions": VERSIONS,
            "runtime": runtime,
            "config_hash": cfg_hash,
            "note": "行为参数与版本共同决定 config_hash；路径不参与哈希。",
        },
    )

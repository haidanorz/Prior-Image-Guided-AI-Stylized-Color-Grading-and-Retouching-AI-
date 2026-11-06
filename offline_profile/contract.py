"""契约加载：分析 Schema、分类体系、参数登记表。"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from .util import canonical_hash

_SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema")


def _load(name: str) -> dict:
    with open(os.path.join(_SCHEMA_DIR, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    return _load("taxonomy.json")


@lru_cache(maxsize=1)
def load_parameters() -> dict:
    return _load("parameters.json")


@lru_cache(maxsize=1)
def load_analysis_schema() -> dict:
    return _load("analysis.schema.json")


@lru_cache(maxsize=1)
def scene_labels() -> tuple[str, ...]:
    return tuple(item["label"] for item in load_taxonomy()["labels"])


@lru_cache(maxsize=1)
def edit_parameter_index() -> dict[str, dict]:
    return {item["key"]: item for item in load_parameters()["edit_parameters"]}


@lru_cache(maxsize=1)
def visual_attribute_levels() -> dict[str, tuple[str, ...]]:
    return {item["key"]: tuple(item["levels"]) for item in load_parameters()["visual_attributes"]}


def registry_hash() -> str:
    return canonical_hash(load_parameters())


def taxonomy_hash() -> str:
    return canonical_hash(load_taxonomy())


def schema_errors(payload: Any) -> list[str]:
    """JSON Schema 结构校验；返回可读错误列表（已排序，确定性）。"""
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise RuntimeError("校验分析 JSON 需要 jsonschema，请先安装依赖") from exc

    validator = Draft202012Validator(load_analysis_schema())
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda e: (tuple(str(part) for part in e.absolute_path), e.message),
    )
    messages: list[str] = []
    for error in errors:
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        messages.append(f"{location}: {error.message}")
    return messages

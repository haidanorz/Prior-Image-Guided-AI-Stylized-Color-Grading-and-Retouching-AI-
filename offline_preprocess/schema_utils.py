"""JSON Schema 加载与校验（jsonschema 为可选依赖）。"""

from __future__ import annotations

import json
import os
from typing import Any

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema", "feature.schema.json")


def load_feature_schema() -> dict[str, Any] | None:
    if not os.path.isfile(_SCHEMA_PATH):
        return None
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def has_jsonschema() -> bool:
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return False
    return True


def validate_feature_payload(payload: dict, schema: dict) -> list[str]:
    """返回可读的校验错误列表；未安装 jsonschema 时返回空列表。"""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return []

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    messages: list[str] = []
    for error in errors:
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        messages.append(f"{location}: {error.message}")
    return messages

"""JSON Schema 契约校验。

为什么单独一层：
  * 提示词会漂、模型会自由发挥；把"至少必须有哪些字段"写成可执行契约，
    比在提示词里重复强调有用；
  * 校验失败走已有的"允许重做一次、不消耗 LR 轮次"通道，不会直接毁掉会话；
  * 测试可以直接拿真实会话的请求/响应过一遍 schema，防止改代码时悄悄改坏协议。

`jsonschema` 未安装时不静默放行——缺依赖却假装校验过，比不校验更危险。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .errors import Codes, DecodeError

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"

REQUEST = "model_request"
SCENE = "model_scene"
BRIEF = "model_brief"
PLAN = "model_plan"
EVALUATION = "model_evaluation"

RESPONSE_SCHEMA = {
    "scene": SCENE,
    "brief": BRIEF,
    "plan": PLAN,
    "evaluate": EVALUATION,
}

ALL_SCHEMAS = (REQUEST, SCENE, BRIEF, PLAN, EVALUATION)


@lru_cache(maxsize=8)
def load(name: str) -> dict[str, Any]:
    if name not in ALL_SCHEMAS:
        raise KeyError(f"未知 schema：{name}")
    path = SCHEMA_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 schema 文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate(name: str, payload: Any) -> list[str]:
    """返回问题列表；空列表表示通过。"""
    try:
        import jsonschema
    except ImportError as exc:
        raise DecodeError(
            Codes.CONFIG_INVALID,
            "缺少 jsonschema 依赖，无法做契约校验：pip install jsonschema",
            stage="schema",
        ) from exc

    validator = jsonschema.Draft7Validator(load(name))
    problems: list[str] = []
    for error in sorted(validator.iter_errors(payload), key=lambda item: list(item.absolute_path)):
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        problems.append(f"{location}: {error.message}")
    return problems


def validate_or_raise(name: str, payload: Any, *, stage: str) -> None:
    problems = validate(name, payload)
    if problems:
        raise DecodeError(
            Codes.MODEL_RESPONSE_INVALID,
            f"{name} 契约校验失败：" + "；".join(problems[:3]),
            stage=stage,
            detail={"schema": name, "problems": problems},
        )

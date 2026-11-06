"""模型适配器协议与 Mock 实现。

本轮只接 Mock：从目录读取预置响应，用于离线开发、契约测试和回放。
真实模型接入时实现同一协议即可，规范化、缓存与审计逻辑不变。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from ..errors import AdapterError
from ..intake import IntakeImage
from .prompt import PromptBundle


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model: str
    version: str
    is_mock: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "version": self.version,
            "is_mock": self.is_mock,
        }


@dataclass
class AnalysisRequest:
    image: IntakeImage
    prompt: PromptBundle
    input_fingerprint: str


@dataclass
class ModelResponse:
    text: str
    model_info: ModelInfo


class AnalysisAdapter(Protocol):
    model_info: ModelInfo

    def analyze(self, request: AnalysisRequest) -> ModelResponse: ...


class MockAdapter:
    """从 <mock_dir>/<image_id>.json 读取响应。

    文件可以是分析 JSON 本身，也可以是 {"raw_text": "..."} 包装，便于测试
    非法格式、重试和错误隔离。
    """

    def __init__(self, mock_dir: str, model_info: ModelInfo | None = None):
        self.mock_dir = os.path.abspath(mock_dir)
        self.model_info = model_info or ModelInfo(
            provider="mock", model="canned-response-v1", version="1.0.0", is_mock=True
        )

    def analyze(self, request: AnalysisRequest) -> ModelResponse:
        path = os.path.join(self.mock_dir, f"{request.image.image_id}.json")
        if not os.path.isfile(path):
            raise AdapterError(f"Mock 响应缺失：{path}", retryable=False)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            raise AdapterError(f"Mock 响应不可读：{exc}", retryable=False) from exc

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return ModelResponse(text=text, model_info=self.model_info)

        if isinstance(payload, dict) and isinstance(payload.get("raw_text"), str):
            return ModelResponse(text=payload["raw_text"], model_info=self.model_info)
        return ModelResponse(
            text=json.dumps(payload, ensure_ascii=False, sort_keys=True), model_info=self.model_info
        )


class ReplayAdapter(MockAdapter):
    """回放冻结响应集（例如宿主会话生成的分析）。

    目录中必须有 `_model_info.json`，否则无法区分真实模型、Agent 分析与 Mock。
    这个适配器不调用任何模型，只负责把已保存的响应送入规范化与审计链路。
    """

    def __init__(self, response_dir: str):
        path = os.path.join(os.path.abspath(response_dir), "_model_info.json")
        if not os.path.isfile(path):
            raise AdapterError(
                f"replay 适配器需要 {path}（provider/model/version/is_mock）", retryable=False
            )
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            info = ModelInfo(
                provider=str(payload["provider"]),
                model=str(payload["model"]),
                version=str(payload["version"]),
                is_mock=bool(payload.get("is_mock", False)),
            )
        except Exception as exc:
            raise AdapterError(f"replay 的 _model_info.json 不可读：{exc}", retryable=False) from exc
        super().__init__(response_dir, model_info=info)


def build_adapter(name: str, *, mock_dir: str | None) -> AnalysisAdapter:
    if name not in ("mock", "replay"):
        raise AdapterError(f"未知适配器：{name!r}（支持 mock / replay）", retryable=False)
    if not mock_dir:
        raise AdapterError("需要 --mock-dir 指向响应目录（mock 与 replay 共用）", retryable=False)
    if not os.path.isdir(mock_dir):
        raise AdapterError(f"响应目录不存在：{mock_dir}", retryable=False)
    if name == "replay":
        return ReplayAdapter(mock_dir)
    return MockAdapter(mock_dir)

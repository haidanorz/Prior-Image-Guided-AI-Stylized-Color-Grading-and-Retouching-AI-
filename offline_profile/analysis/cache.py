"""模型响应缓存。

键 = 图片/特征/模型/prompt/Schema 指纹 + prompt 正文哈希。任何影响分析结果的
因素变化都会自然失效，禁止把新旧响应拼接进同一次构建。
"""

from __future__ import annotations

import json
import os
from ..util import atomic_write_json, now_utc_iso, sha256_bytes
from .adapter import ModelInfo, ModelResponse


class ResponseCache:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    @staticmethod
    def key(input_fingerprint: str, prompt_sha256: str) -> str:
        return sha256_bytes(f"{input_fingerprint}|{prompt_sha256}".encode("utf-8"))

    def _path(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.json")

    def get(self, key: str) -> ModelResponse | None:
        path = self._path(key)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            info = payload["model_info"]
            model_info = ModelInfo(
                provider=str(info["provider"]),
                model=str(info["model"]),
                version=str(info["version"]),
                is_mock=bool(info.get("is_mock", False)),
            )
            return ModelResponse(text=str(payload["text"]), model_info=model_info)
        except Exception:
            return None

    def put(self, key: str, response: ModelResponse, *, input_fingerprint: str, prompt_version: str) -> None:
        atomic_write_json(
            self._path(key),
            {
                "cache_key": key,
                "input_fingerprint": input_fingerprint,
                "prompt_version": prompt_version,
                "model_info": response.model_info.as_dict(),
                "text": response.text,
                "created_at": now_utc_iso(),
            },
        )

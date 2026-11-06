"""OpenAI 兼容视觉 API 驱动：让完整闭环可以一次性自动跑完。

为什么单独有这个实现：
  MCP 宿主不支持 sampling（server 不能反向调用宿主模型），
  所以任务编排是默认路径。若想"一句话全自动"，只能由 server 自己调外部 VLM。
  没有配置 API key 时明确报错，绝不假装能自动跑。

环境变量：
  ONLINE_DECODE_VISION_URL    OpenAI 兼容的 chat/completions 地址
  ONLINE_DECODE_VISION_KEY    API key（必需）
  ONLINE_DECODE_VISION_MODEL  模型名，默认 gpt-4o

只用标准库 urllib，不引入新依赖；图片用 PIL 缩到长边 1024 再 base64，
避免把 25MB 的预览直接塞进请求体。
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..errors import Codes
from ..ports import DriverError

DEFAULT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o"
MAX_IMAGE_EDGE = 1024
REQUEST_TIMEOUT = 180

SYSTEM_PROMPT = """你是在线个性化调色解码器的视觉决策模块。
每次只回答一个任务，输出**严格 JSON**（不要 markdown 代码块、不要解释文字）。
任务由请求包里的 task 字段决定：scene / brief / plan / evaluate。

- scene: 只看 original 图判断场景，输出 scene_observation.primary_candidates
  （label + score）。**不要输出 XMP，也不要猜桶参数**——桶要等控制器按这个结果判定后，
  才会在 brief 阶段连同锚点一起给出。
- brief: 依据请求包 scene.anchors（该桶已发布参数）、scene.priors（宽离散先验，仅供参考）、
  scene.narrative（语境与注意事项），输出 branch_rationale、preserve、goals(2-5 条可观察目标)、
  avoid、initial_xmp（完整 XMP 文本）。
- plan: 输出 xmp（完整 XMP 文本）或 xmp_path；可选 param_notes、expected_effects。
- evaluate: 输出 style_alignment/branch_alignment/technical_quality/content_preservation
  四个带 verdict 的对象、improvement、decision、evidence[]（anchor+observation）。

约束：
- 只能使用请求包 limits.allowed_keys 里的字段；不要写 rdf:Seq 曲线；不要输出白名单外的键。
- anchors 是桶锚点：取值尽量靠近它；span 越大表示桶内分布越宽，允许偏离越多。
- priors 不是锚点：只在画面与该桶明显不符时参考，不能当成固定目标。
- 场景未命中任何桶时只用 global 兜底；没有锚点的字段（not_evaluable）不要凭感觉写，
  确需按本图调整时在 param_notes 里标 evidence_level=target_adaptive，否则会被守卫拒绝。
- evidence 里若引用 metric，数值必须来自请求包 metrics。
"""


def is_configured() -> bool:
    return bool(os.environ.get("ONLINE_DECODE_VISION_KEY"))


class ApiDriver:
    name = "api"

    def __init__(self, *, url: str | None = None, key: str | None = None,
                 model: str | None = None, timeout: float = REQUEST_TIMEOUT) -> None:
        self.url = url or os.environ.get("ONLINE_DECODE_VISION_URL") or DEFAULT_URL
        self.key = key or os.environ.get("ONLINE_DECODE_VISION_KEY") or ""
        self.model = model or os.environ.get("ONLINE_DECODE_VISION_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        if not self.key:
            raise DriverError(
                Codes.CONFIG_INVALID,
                "未配置 ONLINE_DECODE_VISION_KEY，无法自动跑完整算法；"
                "请改用 decode_start/next/submit 任务编排，或配置外部视觉 API",
            )

    # ---- 内部 ----
    def _image_part(self, path: str | None) -> dict[str, Any] | None:
        if not path or not Path(path).is_file():
            return None
        try:
            from PIL import Image

            with Image.open(path) as image:
                image = image.convert("RGB")
                if max(image.size) > MAX_IMAGE_EDGE:
                    ratio = MAX_IMAGE_EDGE / max(image.size)
                    image = image.resize(
                        (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
                        Image.LANCZOS,
                    )
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=82)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}
        except Exception as exc:
            raise DriverError(Codes.MODEL_RESPONSE_INVALID, f"预览图编码失败：{exc}") from exc

    def _call(self, task: str, request: dict[str, Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"task={task}\n" + json.dumps(request, ensure_ascii=False)},
        ]
        for path in (request.get("images") or {}).values():
            part = self._image_part(path)
            if part:
                content.append(part)

        body = json.dumps({
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            self.url, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DriverError(Codes.MODEL_RESPONSE_TIMEOUT, f"视觉 API 调用失败：{exc}") from exc
        except json.JSONDecodeError as exc:
            raise DriverError(Codes.MODEL_RESPONSE_INVALID, "视觉 API 返回的不是 JSON") from exc

        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DriverError(Codes.MODEL_RESPONSE_INVALID, "视觉 API 响应缺 choices[0].message.content") from exc

        result = _parse_json_object(text)
        if result is None:
            raise DriverError(Codes.MODEL_RESPONSE_INVALID, f"视觉 API 未返回 JSON 对象：{str(text)[:160]}")
        return result

    # ---- Driver 协议 ----
    def scene(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("scene", request)

    def brief(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("brief", request)

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("plan", request)

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("evaluate", request)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None

"""测试用驱动：按队列吐出预设响应，并记录收到的请求。"""

from __future__ import annotations

from typing import Any

from ..errors import Codes
from ..ports import DriverError


class ScriptedDriver:
    name = "scripted"

    def __init__(self, *, brief: dict[str, Any], plans: list[dict[str, Any]] | None = None,
                 evaluations: list[dict[str, Any]] | None = None,
                 scene: dict[str, Any] | None = None) -> None:
        self._brief = brief
        self._scene = scene
        self._plans = list(plans or [])
        self._evaluations = list(evaluations or [])
        self.requests: dict[str, list[dict[str, Any]]] = {
            "scene": [], "brief": [], "plan": [], "evaluate": []
        }

    def scene(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests["scene"].append(request)
        if self._scene is None:
            raise DriverError(Codes.MODEL_RESPONSE_INVALID, "ScriptedDriver 没有提供 scene 响应")
        return self._scene

    def brief(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests["brief"].append(request)
        return self._brief

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests["plan"].append(request)
        if not self._plans:
            raise DriverError(Codes.MODEL_RESPONSE_INVALID, "ScriptedDriver 没有更多 plan 了")
        return self._plans.pop(0)

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests["evaluate"].append(request)
        if not self._evaluations:
            raise DriverError(Codes.MODEL_RESPONSE_INVALID, "ScriptedDriver 没有更多 evaluation 了")
        return self._evaluations.pop(0)

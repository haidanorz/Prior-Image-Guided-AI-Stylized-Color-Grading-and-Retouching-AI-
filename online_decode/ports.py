"""模型侧的端口协议。

为什么控制器不直接调模型：
  * 换模型（agent / API / 本地 VLM）不该动状态机；
  * 首版由宿主 Agent 读写图，`agent_driver` 用文件交换即可满足"确实看过图"；
  * 测试用 `ScriptedDriver` 注入固定响应，整套状态机可以离线跑完。

约定：Driver 只收发 JSON-serializable dict，不碰 LR、不碰文件系统（agent_driver 例外，
它自己负责请求/响应的落盘）。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class DriverError(Exception):
    """驱动层失败（超时 / 文件缺失 / 响应不可解析）。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@runtime_checkable
class Driver(Protocol):
    """三次调用对应三种任务，任务提示必须不同（先创作、后复核）。"""

    name: str

    def scene(self, request: dict[str, Any]) -> dict[str, Any]:
        """只看原图 → scene_observation（桶参数尚未给出，避免先验污染场景判断）。"""

    def brief(self, request: dict[str, Any]) -> dict[str, Any]:
        """看原图 + 命中桶的参数 → creative_brief + initial_xmp。"""

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        """看当前预览 + 历史 → 本轮 adjustments。"""

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        """看原图与当前预览 → 结构化评价 + 决策。"""

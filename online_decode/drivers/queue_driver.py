"""队列驱动：让阻塞式控制器能被 MCP 工具一步步推进。

为什么需要它：
  `SessionController.run()` 是"一路跑到完"的同步循环，AgentDriver 靠文件交换
  把模型决策塞回循环里。但 MCP server 没有"写文件等待"这种交互界面——
  host 只提供工具调用。若把整个 run() 放进一个工具里同步执行，工具会阻塞
  几分钟到几十分钟，且模型根本没机会参与。

做法：控制器放到后台线程，Driver 的三次调用改成「请求入队 + 阻塞等响应」；
MCP 工具 `decode_next` 取请求、`decode_submit` 放回响应。
于是状态机一行都不用改，却变成了可暂停、可跨工具调用续跑的形态。

关键点：
  * 每个任务一个唯一 key，响应按 key 投递，杜绝"答错轮次"；
  * 请求/响应都留盘，MCP 工具挂了也能从 session 目录恢复现场；
  * 超时抛 DriverError(MODEL_RESPONSE_TIMEOUT)，控制器会保持 await 状态，
    不会把会话误判成 failed。
"""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import Codes
from ..ports import DriverError
from ..util import atomic_write_json, ensure_dir, now_iso


@dataclass
class PendingTask:
    key: str
    task: str
    round: int
    request: dict[str, Any]
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "task": self.task,
            "round": self.round,
            "created_at": self.created_at,
            "request": self.request,
        }


class QueueDriver:
    name = "queue"

    def __init__(self, *, wait_timeout: float = 3600.0, io_dir: Path | None = None) -> None:
        self.wait_timeout = wait_timeout
        self.io_dir = ensure_dir(io_dir) if io_dir else None
        self._pending: queue.Queue[PendingTask] = queue.Queue()
        self._slots: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._done = threading.Event()

    # ---- 供控制器线程调用 ----
    def _exchange(self, task: str, request: dict[str, Any]) -> dict[str, Any]:
        key = f"{task}_{int(request.get('round') or 0):02d}_{uuid.uuid4().hex[:6]}"
        slot: queue.Queue = queue.Queue(maxsize=1)
        item = PendingTask(key=key, task=task, round=int(request.get("round") or 0), request=request)
        with self._lock:
            self._slots[key] = slot
        if self.io_dir:
            atomic_write_json(self.io_dir / f"req_{key}.json", item.to_dict())

        self._pending.put(item)
        try:
            answer = slot.get(timeout=self.wait_timeout)
        except queue.Empty as exc:
            raise DriverError(
                Codes.MODEL_RESPONSE_TIMEOUT,
                f"等待 {task} r{item.round:02d} 的响应超时（{self.wait_timeout:.0f}s）",
            ) from exc
        finally:
            with self._lock:
                self._slots.pop(key, None)

        if not isinstance(answer, dict):
            raise DriverError(Codes.MODEL_RESPONSE_INVALID, f"{task} 响应不是 JSON 对象")
        if "__error__" in answer:
            raise DriverError(Codes.MODEL_RESPONSE_TIMEOUT, str(answer["__error__"]))
        if self.io_dir:
            atomic_write_json(self.io_dir / f"resp_{key}.json", answer)
        return answer

    def scene(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._exchange("scene", request)

    def brief(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._exchange("brief", request)

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._exchange("plan", request)

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._exchange("evaluate", request)

    # ---- 供 MCP 工具调用 ----
    def next_task(self, timeout: float = 0.0) -> PendingTask | None:
        try:
            return self._pending.get(timeout=timeout if timeout > 0 else 0.01)
        except queue.Empty:
            return None

    def submit(self, key: str, response: dict[str, Any]) -> bool:
        with self._lock:
            slot = self._slots.get(key)
        if slot is None:
            return False
        try:
            slot.put_nowait(response)
        except queue.Full:
            return False
        return True

    def pending_keys(self) -> list[str]:
        with self._lock:
            return sorted(self._slots)

    def fail_all(self, message: str) -> int:
        """中止所有等待中的任务（供 MCP 的 abort 工具使用）。"""
        with self._lock:
            slots = list(self._slots.values())
        count = 0
        for slot in slots:
            try:
                slot.put_nowait({"__error__": message})
                count += 1
            except queue.Full:
                continue
        return count

    def mark_done(self) -> None:
        self._done.set()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

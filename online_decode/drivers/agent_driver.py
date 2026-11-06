"""agent 驱动：用文件交换让宿主 Agent 读图后写回决策。

为什么首版走这条路：
  * 当前没有独立的视觉 API 凭据；
  * 文件交换天然满足"必须先真的打开过那张图"——模型没法凭空写出响应，
    而且请求/响应两份文件本身就是审计证据。

协议：
  请求写到 <session>/agent_io/req_<task>_<round>.json
  响应从 <session>/agent_io/resp_<task>_<round>.json 读取（等待上限可配）
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..errors import Codes
from ..ports import DriverError
from ..util import atomic_write_json, ensure_dir, read_json


class AgentDriver:
    name = "agent"

    def __init__(self, workspace: str | Path, *, timeout_sec: float = 900.0,
                 poll_sec: float = 0.5, on_request=None, reuse_existing: bool = False) -> None:
        self.workspace = Path(workspace)
        self.timeout_sec = timeout_sec
        self.poll_sec = poll_sec
        self.on_request = on_request
        # resume 时不能把用户已经写好的响应删掉：那会让人白写一次
        self.reuse_existing = reuse_existing

    # ---- 内部 ----
    def _exchange(self, task: str, request: dict[str, Any]) -> dict[str, Any]:
        session_id = request.get("session_id") or "unknown"
        round_no = int(request.get("round") or 0)
        io_dir = ensure_dir(self.workspace / "sessions" / session_id / "agent_io")
        req_path = io_dir / f"req_{task}_{round_no:02d}.json"
        resp_path = io_dir / f"resp_{task}_{round_no:02d}.json"

        atomic_write_json(req_path, request)
        if self.reuse_existing and resp_path.is_file():
            existing = read_json(resp_path)
            if isinstance(existing, dict):
                return existing
        if resp_path.exists():
            resp_path.unlink()

        if callable(self.on_request):
            self.on_request(task, round_no, req_path, resp_path)

        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            data = read_json(resp_path)
            if isinstance(data, dict):
                return data
            time.sleep(self.poll_sec)

        raise DriverError(
            Codes.MODEL_RESPONSE_TIMEOUT,
            f"等待 {resp_path.name} 超时（{self.timeout_sec:.0f}s）。"
            f"请求包在 {req_path}，写回响应后可 resume。",
        )

    # ---- 对外 ----
    def scene(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._exchange("scene", request)

    def brief(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._exchange("brief", request)

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._exchange("plan", request)

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._exchange("evaluate", request)

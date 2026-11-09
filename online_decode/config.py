"""唯一配置入口。

为什么单独一个模块、且不允许从别处读业务参数：
  * `max_rounds` 若能被 job options 或模型响应影响，等于把预算交给调用方；
  * 若各处自行 `int(os.environ.get(...))`，报告里的预算与真实执行会对不上。

约束：
  * `max_rounds` 1–5，默认 5，**越界直接失败**，不做宽容钳制（钳制会静默改变预算）。
  * `stop_on_single_no_progress` 为 None 时按 n 推导：n<=3 为 True，n>=4 为 False。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import Codes, DecodeError
from .util import read_json

ROUND_MIN = 1
ROUND_MAX = 5
DEFAULT_MAX_ROUNDS = 5


@dataclass(frozen=True)
class Config:
    max_rounds: int = DEFAULT_MAX_ROUNDS
    min_rounds: int | None = None
    stop_on_single_no_progress: bool | None = None
    style_strength_default: float = 1.0
    driver: str = "agent"
    workspace: Path = field(default_factory=lambda: Path("."))
    sessions_dir: str = "sessions"
    mcp_repo: str | None = None
    preview_long_edge: int = 1600
    request_timeout_sec: float = 900.0
    job_timeout_sec: float | None = None
    strict_schema: bool = True
    decode_mode: str = "auto"
    # 参数语义知识层（L3_cross_project_reference）：只作提示，不参与数值合成。
    # 关掉它不改变算法结果，只是模型少一份「这个参数管哪段」的参考。
    parameter_knowledge: bool = True
    parameter_knowledge_path: str | None = None

    # ---- 派生 ----
    @property
    def effective_min_rounds(self) -> int:
        """强制最少轮数（2026-11-06 用户决策）。

        首轮参数就是桶中心值，若首轮即收尾，"达到桶中心"会被误当成"最优"，
        第 2 轮的本图收敛永远不会发生。默认至少 2 轮；max_rounds=1 时退化为 1
        （单轮预算的显式场景仍然可用）。
        """
        if self.min_rounds is None:
            return min(2, self.max_rounds)
        return int(self.min_rounds)

    @property
    def effective_stop_on_single_no_progress(self) -> bool:
        if self.stop_on_single_no_progress is not None:
            return bool(self.stop_on_single_no_progress)
        return self.max_rounds <= 3

    @property
    def stop_source(self) -> str:
        return "explicit" if self.stop_on_single_no_progress is not None else "derived"

    def to_report(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "min_rounds": self.effective_min_rounds,
            "stop_on_single_no_progress": self.effective_stop_on_single_no_progress,
            "stop_on_single_no_progress_source": self.stop_source,
            "driver": self.driver,
            "decode_mode": self.decode_mode,
            "preview_long_edge": self.preview_long_edge,
            "parameter_knowledge": self.parameter_knowledge,
        }

    def validate(self) -> None:
        if not isinstance(self.max_rounds, int) or isinstance(self.max_rounds, bool):
            raise _bad("max_rounds 必须是整数")
        if not ROUND_MIN <= self.max_rounds <= ROUND_MAX:
            raise _bad(f"max_rounds 只接受 {ROUND_MIN}–{ROUND_MAX}，收到 {self.max_rounds}")
        if self.min_rounds is not None:
            if not isinstance(self.min_rounds, int) or isinstance(self.min_rounds, bool):
                raise _bad("min_rounds 必须是整数或 null")
            if not ROUND_MIN <= self.min_rounds <= ROUND_MAX:
                raise _bad(f"min_rounds 只接受 {ROUND_MIN}–{ROUND_MAX}，收到 {self.min_rounds}")
            if self.min_rounds > self.max_rounds:
                raise _bad(
                    f"min_rounds({self.min_rounds}) 不能大于 max_rounds({self.max_rounds})"
                )
        if not 0.0 < float(self.style_strength_default) <= 2.0:
            raise _bad("style_strength_default 必须在 (0, 2] 之间")
        if self.driver not in ("agent", "scripted", "api", "queue"):
            raise _bad(f"未知 driver：{self.driver}")
        if not 200 <= int(self.preview_long_edge) <= 4096:
            raise _bad("preview_long_edge 必须在 200–4096 之间")
        if self.decode_mode not in ("auto", "bucket_primary", "global_delta"):
            raise _bad("decode_mode 只能是 auto / bucket_primary / global_delta")
        if self.parameter_knowledge_path is not None:
            if not isinstance(self.parameter_knowledge_path, str) or not self.parameter_knowledge_path.strip():
                raise _bad("parameter_knowledge_path 必须是非空字符串或 null")


def _bad(message: str) -> DecodeError:
    return DecodeError(Codes.CONFIG_INVALID, message, stage="config")


def load(
    path: str | os.PathLike[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """从 JSON 配置加载并合并命令行覆盖项。

    优先级：overrides > 配置文件 > 默认值。配置文件中出现未知键直接报错，
    避免 `max_round` 这类拼写错误被静默忽略（那会让预算变成默认值）。
    """
    data: dict[str, Any] = {}
    if path:
        raw = read_json(path)
        if raw is None:
            raise _bad(f"配置文件不存在：{path}")
        if not isinstance(raw, dict):
            raise _bad(f"配置文件必须是 JSON 对象：{path}")
        data = dict(raw)

    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    known = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = sorted(set(data) - known)
    if unknown:
        raise _bad("配置包含未知键：" + ", ".join(unknown))

    if "workspace" in data:
        data["workspace"] = Path(data["workspace"])

    cfg = replace(Config(), **data)
    cfg.validate()
    return cfg

"""会话状态的唯一真相：session.json。

两条硬规则：
  1. 只追加，不覆盖历史轮次（rounds 里已有条目不会被重写）；
  2. 每次变更整体原子落盘，崩溃后能靠 status + rounds 恢复判断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import atomic_write_json, ensure_dir, now_iso, read_json

STATUSES = (
    "initialized",
    "applying",
    "rendering",
    "evaluating",
    "restoring",
    "exporting",
    "completed",
    "failed",
    "aborted",
)


@dataclass
class SessionState:
    session_id: str
    photo: str
    root: Path
    status: str = "initialized"
    substep: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    # 配置快照（会话开始固化，中途不可变）
    max_rounds: int = 5
    stop_on_single_no_progress: bool = True
    stop_source: str = "derived"
    mode: str = "pilot"
    user_intent: str | None = None
    style_strength: float = 1.0
    fingerprint: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    best: dict[str, Any] | None = None
    stop_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, Any] | None = None

    # ---- 派生 ----
    @property
    def render_count(self) -> int:
        return sum(1 for r in self.rounds if r.get("render"))

    @property
    def path(self) -> Path:
        return self.root / "session.json"

    # ---- 变更 ----
    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def fail(self, error: dict[str, Any]) -> None:
        self.errors.append(error)
        self.status = "failed"

    def set_status(self, status: str, substep: str = "") -> None:
        if status not in STATUSES:
            raise ValueError(f"未知会话状态：{status}")
        self.status = status
        self.substep = substep
        self.save()

    def add_round(self, record: dict[str, Any]) -> None:
        self.rounds.append(record)
        self.save()

    def update_round(self, round_no: int, **patch: Any) -> None:
        for item in self.rounds:
            if item.get("round") == round_no:
                item.update(patch)
                self.save()
                return
        raise KeyError(f"没有第 {round_no} 轮")

    def save(self) -> Path:
        self.updated_at = now_iso()
        return atomic_write_json(self.path, self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_schema_version": "1.0.0",
            "session_id": self.session_id,
            "photo": self.photo,
            "status": self.status,
            "substep": self.substep,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "max_rounds": self.max_rounds,
            "stop_on_single_no_progress": self.stop_on_single_no_progress,
            "stop_on_single_no_progress_source": self.stop_source,
            "mode": self.mode,
            "user_intent": self.user_intent,
            "style_strength": self.style_strength,
            "fingerprint": self.fingerprint,
            "baseline": self.baseline,
            "rounds": self.rounds,
            "best": self.best,
            "stop_reason": self.stop_reason,
            "warnings": self.warnings,
            "errors": self.errors,
            "final": self.final,
        }

    @classmethod
    def load(cls, root: str | Path) -> "SessionState":
        path = Path(root) / "session.json"
        data = read_json(path)
        if not isinstance(data, dict):
            raise FileNotFoundError(f"找不到会话状态：{path}")
        state = cls(
            session_id=data["session_id"],
            photo=data["photo"],
            root=Path(root),
            status=data.get("status", "initialized"),
            substep=data.get("substep", ""),
            created_at=data.get("created_at", now_iso()),
            updated_at=data.get("updated_at", now_iso()),
            max_rounds=int(data.get("max_rounds", 5)),
            stop_on_single_no_progress=bool(data.get("stop_on_single_no_progress", True)),
            stop_source=data.get("stop_on_single_no_progress_source", "derived"),
            mode=data.get("mode", "pilot"),
            user_intent=data.get("user_intent"),
            style_strength=float(data.get("style_strength") or 1.0),
            fingerprint=data.get("fingerprint") or {},
            baseline=data.get("baseline") or {},
            rounds=list(data.get("rounds") or []),
            best=data.get("best"),
            stop_reason=data.get("stop_reason"),
            warnings=list(data.get("warnings") or []),
            errors=list(data.get("errors") or []),
            final=data.get("final"),
        )
        return state


def make_session_id() -> str:
    """会话 ID 由调用方生成，便于 MCP 等外部宿主先拿到 ID 再启动后台线程。"""
    import time
    import uuid

    return f"s_{int(time.time())}_{uuid.uuid4().hex[:6]}"


def new_session(workspace: Path, session_id: str, photo: str, *, max_rounds: int,
                stop_on_single: bool, stop_source: str, mode: str) -> SessionState:
    root = ensure_dir(Path(workspace) / "sessions" / session_id)
    state = SessionState(
        session_id=session_id,
        photo=photo,
        root=root,
        max_rounds=max_rounds,
        stop_on_single_no_progress=stop_on_single,
        stop_source=stop_source,
        mode=mode,
    )
    state.save()
    return state

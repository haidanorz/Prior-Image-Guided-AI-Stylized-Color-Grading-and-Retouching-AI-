"""情节记忆：单次重试、非法响应、分析失败只进这里，不直接成为技能。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .util import append_jsonl, atomic_write_bytes, now_utc_iso, sha256_bytes


@dataclass
class EpisodeStore:
    build_id: str
    path: str
    rows: list[dict] = field(default_factory=list)
    _seq: int = 0
    _written: int = 0

    def add(
        self,
        event_type: str,
        *,
        image_id: str | None,
        context: dict[str, Any],
        action: str,
        result: str,
        feedback: str | None = None,
        evidence_refs: list[str] | None = None,
    ) -> str:
        self._seq += 1
        digest = sha256_bytes(
            f"{self.build_id}|{image_id}|{event_type}|{self._seq}".encode("utf-8")
        )[:16]
        episode_id = f"ep_{digest}"
        self.rows.append(
            {
                "episode_id": episode_id,
                "build_id": self.build_id,
                "image_id": image_id,
                "event_type": event_type,
                "context": context,
                "action": action,
                "result": result,
                "feedback": feedback,
                "evidence_refs": evidence_refs or [],
                "created_at": now_utc_iso(),
            }
        )
        return episode_id

    def write(self) -> None:
        pending = self.rows[self._written :]
        if pending:
            append_jsonl(self.path, pending)
            self._written = len(self.rows)
        elif not os.path.exists(self.path):
            atomic_write_bytes(self.path, b"")

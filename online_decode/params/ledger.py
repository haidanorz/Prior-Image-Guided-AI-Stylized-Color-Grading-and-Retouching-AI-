"""参数台账：意图 → 解析值 → 写入值 三层记录。

为什么不让模型直接产 XMP：
  模型写 XMP 等于把字段白名单、值域、证据等级交给提示词去遵守。台账把"谁决定了
  什么"拆开——模型只给 intent，baseline/guard 负责 resolved/written，
  被拒项也必须留档（否则报告无法解释"为什么这轮没生效"）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EVIDENCE_LEVELS = (
    "runtime_reference",
    "experimental_prior",
    "narrative_only",
    "planner_judgement",
    "target_adaptive",
    "unsupported",
)

# 允许进入首轮的等级（unsupported 永不入）
FIRST_ROUND_ALLOWED = {
    "runtime_reference",
    "experimental_prior",
    "narrative_only",
    "planner_judgement",
    "target_adaptive",
}


@dataclass
class Entry:
    given_key: str
    sdk: str
    intent: Any
    resolved: Any
    written: Any
    evidence_level: str
    ref: str
    reason: str
    role: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "given_key": self.given_key,
            "sdk": self.sdk,
            "intent": self.intent,
            "resolved": self.resolved,
            "written": self.written,
            "evidence_level": self.evidence_level,
            "ref": self.ref,
            "reason": self.reason,
            "role": self.role,
            "notes": self.notes,
        }


@dataclass
class Rejection:
    given_key: str
    value: Any
    code: str
    message: str
    evidence_level: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "given_key": self.given_key,
            "value": self.value,
            "code": self.code,
            "message": self.message,
            "evidence_level": self.evidence_level,
        }


@dataclass
class Ledger:
    round_no: int
    entries: list[Entry] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    auto_added: list[str] = field(default_factory=list)

    @property
    def values(self) -> dict[str, Any]:
        return {e.sdk: e.written for e in self.entries}

    @property
    def is_empty(self) -> bool:
        return not self.entries

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_schema_version": "1.0.0",
            "round": self.round_no,
            "entries": [e.to_dict() for e in self.entries],
            "rejected": [r.to_dict() for r in self.rejected],
            "warnings": self.warnings,
            "auto_added": self.auto_added,
            "written_values": self.values,
        }

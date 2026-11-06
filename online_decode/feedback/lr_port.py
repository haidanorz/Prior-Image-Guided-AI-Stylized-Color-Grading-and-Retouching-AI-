"""Lightroom 往返端口。

控制器不直接拼 job JSON，也不引入 MCP client：
  * 复用 `ai_color_mcp.queue_bridge.QueueBridge`（同进程、纯 Python、已有测试覆盖）；
  * 队列本身就是契约，MCP 工具层只是它的薄封装。

所有动作统一返回 `Outcome`，把插件原始 result 的 `photos[0]` 摊平，
这样上层不需要记住每个动作的字段位置。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..errors import Codes, DecodeError
from ..util import ensure_dir, sha256_bytes


@dataclass
class Outcome:
    ok: bool
    action: str
    job_id: str = ""
    elapsed: float = 0.0
    entry: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    message: str | None = None

    @property
    def settings_hash(self) -> str | None:
        if self.entry.get("settings_hash"):
            return self.entry["settings_hash"]
        # 插件写的是 snapshot；MCP 的 summarize 层会改名成 settings_snapshot，
        # 但控制器直连队列读的是原始结果，两种都得认。
        for key in ("settings_snapshot", "snapshot"):
            snapshot = self.entry.get(key) or {}
            if snapshot.get("settings_hash"):
                return snapshot["settings_hash"]
        for block in ("render", "restore", "export"):
            block_data = self.entry.get(block) or {}
            if block_data.get("settings_hash"):
                return block_data["settings_hash"]
        return None

    @property
    def snapshot(self) -> dict[str, Any]:
        return self.entry.get("settings_snapshot") or self.entry.get("snapshot") or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "job_id": self.job_id,
            "elapsed_sec": round(self.elapsed, 3),
            "error_code": self.error_code,
            "message": self.message,
            "settings_hash": self.settings_hash,
        }


class LrPort(Protocol):
    def capabilities(self) -> dict[str, Any]: ...

    def get_settings(self, photo: str) -> Outcome: ...

    def apply(self, photo: str, xmp_path: str, *, dry_run: bool = False) -> Outcome: ...

    def render_preview(self, photo: str, *, session_id: str, round_no: int,
                       expected_hash: str | None, long_edge: int = 1600) -> Outcome: ...

    def restore(self, photo: str, snapshot_path: str, *,
                expected_current_hash: str | None) -> Outcome: ...

    def export_final(self, photo: str, *, session_id: str,
                     expected_hash: str | None) -> Outcome: ...


# ----------------------------------------------------------------------
# 真实实现
# ----------------------------------------------------------------------


class QueueLrPort:
    name = "queue"

    def __init__(self, mcp_repo: str | Path, *, timeout: float | None = None) -> None:
        repo = Path(mcp_repo).resolve()
        if not (repo / "ai_color_mcp" / "queue_bridge.py").is_file():
            raise DecodeError(
                Codes.CONFIG_INVALID,
                f"mcp_repo 下找不到 ai_color_mcp：{repo}",
                stage="preflight",
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from ai_color_mcp.queue_bridge import (  # noqa: WPS433
            JobOptions,
            QueueBridge,
            discover_queue_dir,
        )

        self._JobOptions = JobOptions
        # QueueBridge 没有无参构造：队列目录必须显式给，否则会表现为
        # "监听器明明开着，job 却没人取"（连到了另一个目录）。
        self.bridge = QueueBridge(discover_queue_dir())
        self.timeout = timeout

    # ---- 内部 ----
    def _run(self, action: str, photo: str, *, extra: dict | None = None,
             payload: dict | None = None, dry_run: bool = False) -> Outcome:
        options = self._JobOptions(dry_run=dry_run, verify=True, extra=extra or {})
        try:
            result, job_id, elapsed = self.bridge.run_action(
                action, [photo], options=options, payload=payload, timeout=self.timeout
            )
        except Exception as exc:  # 桥接层异常（监听器未开 / 超时）
            raise DecodeError(
                Codes.APPLY_STATE_UNKNOWN,
                f"{action} 未拿到结果：{exc}",
                stage=action,
            ) from exc

        photos = result.get("photos") or []
        entry = photos[0] if photos else {}
        code = entry.get("code") or result.get("code")
        return Outcome(
            ok=bool(result.get("ok")) and bool(entry.get("ok")),
            action=result.get("action") or action,
            job_id=result.get("id") or job_id,
            elapsed=elapsed,
            entry=entry,
            error_code=code,
            message=entry.get("message") or result.get("message"),
        )

    # ---- 对外 ----
    def capabilities(self) -> dict[str, Any]:
        info = self.bridge.plugin_capabilities()
        status = self.bridge.self_check()
        return {
            "capabilities": info.get("capabilities") or [],
            "source": info.get("source"),
            "plugin_version": status.get("plugin", {}).get("plugin_version") or info.get("plugin_version"),
            "feedback_fields_version": info.get("feedback_fields_version"),
            "job_schema_versions": status.get("job_schema_versions"),
            "queue_dir": status.get("queue_dir"),
            "writable": status.get("writable"),
            "plugin_ready": status.get("plugin_ready"),
            "missing_capabilities": status.get("missing_capabilities"),
        }

    def get_settings(self, photo: str) -> Outcome:
        return self._run("get_settings", photo)

    def apply(self, photo: str, xmp_path: str, *, dry_run: bool = False) -> Outcome:
        # xmp 属于 job 顶层字段（payload），不是 JobOptions.extra。
        # 放错位置会在 bridge 侧直接抛 "action=apply 必须提供 xmp"。
        return self._run("apply", photo, payload={"xmp": xmp_path}, dry_run=dry_run)

    def render_preview(self, photo: str, *, session_id: str, round_no: int,
                       expected_hash: str | None, long_edge: int = 1600) -> Outcome:
        return self._run(
            "render_preview",
            photo,
            extra={
                "sessionId": session_id,
                "round": round_no,
                "format": "jpg",
                "longEdge": long_edge,
                "quality": 85,
                "colorSpace": "sRGB",
                "stripMetadata": True,
                "expectedSettingsHash": expected_hash,
            },
        )

    def restore(self, photo: str, snapshot_path: str, *,
                expected_current_hash: str | None) -> Outcome:
        return self._run(
            "restore_settings",
            photo,
            extra={"expectedCurrentSettingsHash": expected_current_hash, "force": False},
            payload={"snapshot": snapshot_path},
        )

    def export_final(self, photo: str, *, session_id: str,
                     expected_hash: str | None) -> Outcome:
        return self._run(
            "export_final",
            photo,
            extra={
                "sessionId": session_id,
                "format": "jpg",
                "quality": 95,
                "colorSpace": "sRGB",
                "stripMetadata": False,
                "expectedSettingsHash": expected_hash,
            },
        )


# ----------------------------------------------------------------------
# 测试替身
# ----------------------------------------------------------------------


def quantity(value: Any) -> str:
    """与插件 SnapshotStore.quantize 一致：4 位小数、去尾零、-0 归一。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return "0" if text in ("-0", "") else text
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(quantity(v) for v in value) + "]"
    return str(value)


def settings_hash(settings: dict[str, Any]) -> str:
    parts = [f"{k}={quantity(v)}" for k, v in sorted(settings.items())]
    return "sha256:" + sha256_bytes(";".join(parts).encode("utf-8"))


class FakeLrPort:
    """内存版 Lightroom：可注入哈希漂移、渲染失败、导出失败。

    存在的意义不是"跑通"，而是**故意复现**真机上出现过的坏情况：
    预览哈希不匹配、恢复后字段集合漂移、导出时状态被人改过。
    """

    name = "fake"

    def __init__(self, settings: dict[str, Any], *, workspace: str | Path,
                 fail_render: bool = False, render_hash_drift: bool = False,
                 fail_export: bool = False, export_hash_skipped: bool = False,
                 apply_hash_incomplete: bool = False,
                 unstable_reads: int = 0) -> None:
        self.settings = dict(settings)
        self.root = ensure_dir(Path(workspace) / "fake_lr")
        self.fail_render = fail_render
        self.render_hash_drift = render_hash_drift
        self.fail_export = fail_export
        self.export_hash_skipped = export_hash_skipped
        # 复现真机（刚导入的重 RAW，LR 尚未物化完全 develop 数据）：
        # apply 时刻只读到一个不完整子集，settings_hash_complete=false，
        # 而紧随其后的 get_settings 已读全 —— 两个子集哈希必然对不上。
        self.apply_hash_incomplete = apply_hash_incomplete
        # 复现"读数渐稳"：前 unstable_reads 次 get_settings 返回不完整子集
        # （子集随读取次数逐步变全），之后才稳定为完整读数。
        self.unstable_reads = unstable_reads
        self._reads = 0
        self.calls: list[str] = []
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.renders: dict[str, str] = {}
        self._counter = 0

    def _job(self) -> str:
        self._counter += 1
        return f"fake-{self._counter:04d}"

    def capabilities(self) -> dict[str, Any]:
        return {
            "capabilities": ["apply", "get_settings", "render_preview", "restore_settings", "export_final"],
            "source": "fake",
            "plugin_version": "1.1.0 (build 18)",
            "feedback_fields_version": "1.0.0",
            "job_schema_versions": ["1.0.0", "2.0.0"],
            "queue_dir": str(self.root),
            "writable": True,
            "plugin_ready": True,
            "missing_capabilities": [],
        }

    def get_settings(self, photo: str) -> Outcome:
        self.calls.append("get_settings")
        job = self._job()
        snap_id = f"snap-{job}"
        settings_view = dict(self.settings)
        complete = True
        if self.unstable_reads and self._reads < self.unstable_reads:
            # 模拟异步物化：每次读取少几个字段，逐步变全
            self._reads += 1
            complete = False
            missing = max(1, 4 - self._reads)
            items = sorted(settings_view.items())
            settings_view = dict(items[: max(1, len(items) - missing)])
        payload = {
            "snapshot_id": snap_id,
            "created_at": "2026-11-05T00:00:00Z",
            "photo": photo,
            "settings": settings_view,
            "settings_hash": settings_hash(settings_view),
            "complete": complete,
            "read_error": None,
        }
        path = self.root / f"{snap_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.snapshots[str(path)] = payload
        return Outcome(
            ok=True, action="get_settings", job_id=job,
            entry={
                "settings": dict(settings_view),
                "settings_snapshot": {
                    "snapshot_id": snap_id,
                    "path": str(path),
                    "sha256": "sha256:" + sha256_bytes(path.read_bytes()),
                    "settings_hash": payload["settings_hash"],
                    "complete": complete,
                },
            },
        )

    def apply(self, photo: str, xmp_path: str, *, dry_run: bool = False) -> Outcome:
        self.calls.append("dry_run" if dry_run else "apply")
        job = self._job()
        if dry_run:
            return Outcome(ok=True, action="apply", job_id=job, entry={"wouldApply": 1})
        text = Path(xmp_path).read_text(encoding="utf-8")
        for key, value in _parse_crs(text).items():
            self.settings[key] = value
        reported_hash = settings_hash(self.settings)
        reported_complete = True
        if self.apply_hash_incomplete:
            # 模拟不完整读数：哈希算在少一个字段的子集上（与基线 r00 时刻的行为一致）
            items = sorted(self.settings.items())
            reported_hash = settings_hash(dict(items[:-1]))
            reported_complete = False
        return Outcome(
            ok=True, action="apply", job_id=job,
            entry={"applied": 1, "settings_hash": reported_hash,
                   "settings_hash_complete": reported_complete,
                   "verify": {"checked": 1, "mismatches": 0}},
        )

    def render_preview(self, photo: str, *, session_id: str, round_no: int,
                       expected_hash: str | None, long_edge: int = 1600) -> Outcome:
        self.calls.append("render_preview")
        job = self._job()
        if self.fail_render:
            return Outcome(ok=False, action="render_preview", job_id=job,
                           error_code="RENDER_FAILED", message="注入的渲染失败")

        current = settings_hash(self.settings)
        if expected_hash and expected_hash != current:
            return Outcome(ok=False, action="render_preview", job_id=job,
                           error_code="RENDER_HASH_MISMATCH",
                           message=f"期望 {expected_hash[:16]} 实际 {current[:16]}")

        from PIL import Image

        out_dir = ensure_dir(self.root / "renders" / job)
        path = out_dir / f"r{round_no:02d}.jpg"
        image = Image.new("RGB", (64, 48), color=tuple(_sample_color(self.settings)))
        image.save(path, "JPEG", quality=85)
        self.renders[str(path)] = job

        reported = current
        if self.render_hash_drift:
            reported = "sha256:" + "0" * 64
        return Outcome(
            ok=True, action="render_preview", job_id=job,
            entry={"render": {
                "path": str(path), "format": "jpg", "width": 64, "height": 48,
                "bytes": path.stat().st_size, "sha256": "sha256:" + sha256_bytes(path.read_bytes()),
                "settings_hash": reported, "session_id": session_id, "round": round_no,
            }},
        )

    def restore(self, photo: str, snapshot_path: str, *,
                expected_current_hash: str | None) -> Outcome:
        self.calls.append("restore_settings")
        job = self._job()
        snap = self.snapshots.get(str(snapshot_path))
        if snap is None:
            return Outcome(ok=False, action="restore_settings", job_id=job,
                           error_code="SNAPSHOT_NOT_FOUND", message="快照不存在")
        current = settings_hash(self.settings)
        if expected_current_hash and expected_current_hash != current:
            return Outcome(ok=False, action="restore_settings", job_id=job,
                           error_code="RESTORE_HASH_MISMATCH", message="当前状态已被改动")
        self.settings = dict(snap["settings"])
        return Outcome(
            ok=True, action="restore_settings", job_id=job,
            entry={"restore": {
                "snapshot_id": snap["snapshot_id"], "applied": len(self.settings),
                "settings_hash": settings_hash(self.settings),
                "verify": {"checked": len(self.settings), "mismatches": 0, "unverified": 0, "fields": []},
            }},
        )

    def export_final(self, photo: str, *, session_id: str,
                     expected_hash: str | None) -> Outcome:
        self.calls.append("export_final")
        job = self._job()
        if self.fail_export:
            return Outcome(ok=False, action="export_final", job_id=job,
                           error_code="EXPORT_FAILED", message="注入的导出失败")
        current = settings_hash(self.settings)
        if expected_hash and expected_hash != current:
            return Outcome(ok=False, action="export_final", job_id=job,
                           error_code="EXPORT_HASH_MISMATCH",
                           message=f"期望 {expected_hash[:16]} 实际 {current[:16]}")

        from PIL import Image

        out_dir = ensure_dir(self.root / "exports" / session_id)
        path = out_dir / "final.jpg"
        Image.new("RGB", (640, 480), color=tuple(_sample_color(self.settings))).save(path, "JPEG", quality=95)
        return Outcome(
            ok=True, action="export_final", job_id=job,
            entry={"export": {
                "path": str(path), "format": "jpg", "width": 640, "height": 480,
                "bytes": path.stat().st_size,
                "sha256": None if self.export_hash_skipped else "sha256:" + sha256_bytes(path.read_bytes()),
                "hash_skipped": "file_too_large" if self.export_hash_skipped else None,
                "settings_hash": current, "session_id": session_id,
            }},
        )


def _sample_color(settings: dict[str, Any]) -> list[int]:
    """把设置映射成一个可辨颜色，让连续两轮的预览图不一样（便于指标与测试）。"""
    exposure = float(settings.get("Exposure2012", 0) or 0)
    temp = float(settings.get("Temperature", 5500) or 5500)
    base = int(max(0, min(255, 120 + exposure * 40)))
    warm = int(max(0, min(60, (temp - 5500) / 100)))
    return [min(255, base + warm), base, max(0, base - warm)]


def _parse_crs(text: str) -> dict[str, Any]:
    import re

    out: dict[str, Any] = {}
    for key, raw in re.findall(r'crs:(\w+)="([^"]*)"', text):
        if key in ("PresetType", "Version", "HasSettings", "Name", "Group"):
            continue
        if raw in ("True", "False"):
            out[key] = raw == "True"
        else:
            try:
                out[key] = int(raw)
            except ValueError:
                try:
                    out[key] = float(raw)
                except ValueError:
                    out[key] = raw
    return out

"""单轮执行器：写入 → 快照 → 渲染 → 哈希闸门 → 客观指标。

这一层不碰模型，也不决定"要不要继续"。它只负责把一轮做成**可审计的事实**：
写的是哪份 XMP、LR 返回哪个哈希、预览文件在哪、哈希对不对得上。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..errors import Codes
from ..params import xmp_renderer
from ..util import atomic_write_bytes, ensure_dir, now_iso, sha256_bytes
from .lr_port import LrPort

# LR 对刚导入的重 RAW 会异步物化 develop 数据：连续几次读数的字段集合可能不同。
# 快照不完整时按此节奏重读（实测 2~3 次内稳定）；始终读不全再降级。
STABLE_READ_ATTEMPTS = 4
STABLE_READ_DELAY_SEC = 2.5


def stable_settings(port: LrPort, photo: str, *, attempts: int | None = None,
                    delay: float | None = None):
    """读取设置并等待读数稳定（complete=true）。

    真机（索尼 70MB RAW 刚导入）上 LR 会持续物化 develop 数据，读数从不完整
    逐步变全；直接取第一次读数会把"还在变化的子集哈希"当成权威值。
    始终读不全时返回最后一次读数，由调用方决定是否降级。
    """
    attempts = attempts or STABLE_READ_ATTEMPTS
    delay = STABLE_READ_DELAY_SEC if delay is None else delay
    snapshot = port.get_settings(photo)
    attempt = 1
    while attempt < attempts:
        block = snapshot.snapshot or {}
        if snapshot.settings_hash and block.get("complete", False):
            return snapshot
        time.sleep(delay)
        attempt += 1
        snapshot = port.get_settings(photo)
    return snapshot


def write_round_xmp(round_dir: Path, values: dict[str, Any], *, round_no: int) -> tuple[Path, str]:
    text = xmp_renderer.render(values, label=f"Online decode round {round_no}")
    path = round_dir / f"r{round_no:02d}.xmp"
    atomic_write_bytes(path, text.encode("utf-8"))
    return path, sha256_bytes(text.encode("utf-8"))


def execute(
    port: LrPort,
    *,
    photo: str,
    session_id: str,
    round_no: int,
    values: dict[str, Any],
    round_dir: str | Path,
    long_edge: int = 1600,
    with_metrics: bool = True,
    baseline_preview: str | None = None,
) -> dict[str, Any]:
    from ..eval import metrics as metrics_mod

    directory = ensure_dir(round_dir)
    record: dict[str, Any] = {
        "round": round_no,
        "started_at": now_iso(),
        "writes": len(values),
        "jobs": {},
        "ok": False,
        "error": None,
        "warnings": [],
    }

    xmp_path, xmp_hash = write_round_xmp(directory, values, round_no=round_no)
    record["xmp"] = {"path": str(xmp_path), "sha256": xmp_hash, "values": values}

    # 1) dry-run：非法字段在写入前就被拒
    dry = port.apply(photo, str(xmp_path), dry_run=True)
    record["jobs"]["dry_run"] = dry.job_id
    if not dry.ok:
        record["error"] = _err(Codes.APPLY_FAILED, f"dry-run 失败：{dry.message}", dry.error_code)
        return record

    # 2) 真实写入
    applied = port.apply(photo, str(xmp_path))
    record["jobs"]["apply"] = applied.job_id
    if not applied.ok:
        record["error"] = _err(Codes.APPLY_FAILED, f"apply 失败：{applied.message}", applied.error_code)
        return record

    apply_hash = applied.settings_hash
    apply_complete = bool(applied.entry.get("settings_hash_complete", True))
    record["apply"] = {
        "job_id": applied.job_id,
        "settings_hash": apply_hash,
        "complete": apply_complete,
    }
    if not apply_hash:
        record["warnings"].append(
            "apply 没有返回 settings_hash：无法证明预览对应本轮，只能按 pilot 继续"
        )
    elif not apply_complete:
        record["warnings"].append(
            "apply 返回的 settings_hash 来自不完整读数（settings_hash_complete=false）；"
            "真机在刚导入的重 RAW 上出现过，等快照读全后用快照哈希替代"
        )

    # 3) 本轮快照（恢复最佳轮要用它）；读数不完整时先等它稳定
    snapshot = stable_settings(port, photo)
    record["jobs"]["snapshot"] = snapshot.job_id
    if not snapshot.ok:
        record["error"] = _err(Codes.SNAPSHOT_INCOMPLETE,
                               f"读取本轮设置失败：{snapshot.message}", snapshot.error_code)
        return record
    snap_block = snapshot.snapshot
    record["snapshot"] = snap_block
    snap_complete = bool(snap_block.get("complete", False))
    if not snap_complete:
        record["warnings"].append(
            "快照读数持续不完整（已重试），不能承诺精确恢复到本轮"
        )

    # 4) 渲染预览（带期望哈希，防错轮）
    # 期望值的取法（按权威性递减）：
    #   a. 本轮快照完整 → 用快照哈希（apply 后的完整权威读数）；
    #   b. 快照不完整但 apply 哈希完整 → 用 apply 哈希；
    #   c. 都不完整（读数还在物化）→ 无闸门渲染，pilot 降级并显式留痕。
    snap_hash = snap_block.get("settings_hash")
    if snap_complete and snap_hash:
        expected = snap_hash
    elif apply_hash and apply_complete:
        expected = apply_hash
        record["warnings"].append(
            "本轮快照不完整，渲染期望值退回 apply 哈希"
        )
    else:
        expected = None
        record["warnings"].append(
            "设置读数持续不完整（LR 仍在物化 develop 数据），本轮渲染不带期望哈希"
            "（pilot 降级，哈希闸门关闭）"
        )
    record["state_hash"] = expected or snap_hash

    rendered = port.render_preview(
        photo, session_id=session_id, round_no=round_no,
        expected_hash=expected, long_edge=long_edge,
    )
    record["jobs"]["render"] = rendered.job_id
    if not rendered.ok:
        record["error"] = _err(Codes.RENDER_FAILED,
                               f"预览导出失败：{rendered.message}", rendered.error_code)
        return record

    render_block = rendered.entry.get("render") or {}
    record["render"] = render_block

    # 5) 哈希闸门：这一张图必须就是本轮写出来的那一版
    render_hash = render_block.get("settings_hash")
    if expected and render_hash and render_hash != expected:
        record["error"] = _err(
            Codes.RENDER_HASH_MISMATCH,
            f"预览哈希不等于写入哈希（apply/snapshot {expected[:20]} / render {render_hash[:20]}），"
            "该轮预览作废",
        )
        return record
    if expected and not render_hash:
        record["warnings"].append("预览未返回 settings_hash，无法核对轮次")

    # 6) 客观指标
    if with_metrics:
        try:
            base = baseline_preview or None
            record["metrics"] = metrics_mod.compute(render_block["path"], base)
        except Exception as exc:
            record["warnings"].append(f"指标计算失败（不影响本轮有效）：{exc}")

    record["ok"] = True
    record["finished_at"] = now_iso()
    return record


def _err(code: str, message: str, origin: str | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "origin": origin}

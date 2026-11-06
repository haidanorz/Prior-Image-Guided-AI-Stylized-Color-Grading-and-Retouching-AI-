"""在线个性化解码的 MCP server：把完整算法搬进 MCP 工具层。

设计前提（来自项目里 offline_profile/mcp_server.py 的实测结论）：
  MCP 宿主**不支持 sampling/createMessage**（返回 Method not found），
  server 不能反向请求宿主模型。因此采用任务编排：

      decode_start  →  decode_next（取任务包，含图片路径）
                    →  Agent 用读图能力看图
                    →  decode_submit（回传结构化判断）
                    →  重复直到 done
                    →  decode_report

  算法本体（指纹校验、桶判定、参数守卫、哈希链、n 轮状态机、早停、最佳轮、
  恢复、导出）全部在 server 内运行，宿主只提供视觉判断。控制器跑在后台线程，
  QueueDriver 负责把三次模型调用变成可跨工具调用暂停的请求/响应。

  想要"一句话全自动"必须另配 OpenAI 兼容的视觉 API（宿主模型无法被反向调用），
  当前工具集不伪装能做到这一点。

启动（stdio）：
    python online_decode/mcp_server.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_PACKAGE_PARENT = str(Path(__file__).resolve().parents[1])
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass

from mcp.server.fastmcp import FastMCP  # noqa: E402

from online_decode import contracts  # noqa: E402
from online_decode.config import Config  # noqa: E402
from online_decode.drivers.api_driver import ApiDriver, is_configured  # noqa: E402
from online_decode.drivers.queue_driver import QueueDriver  # noqa: E402
from online_decode.feedback.lr_port import QueueLrPort  # noqa: E402
from online_decode.fingerprint.resolver import verify_package  # noqa: E402
from online_decode.params.fields import Fields  # noqa: E402
from online_decode.session import SessionController  # noqa: E402
from online_decode.state import SessionState, make_session_id  # noqa: E402
from online_decode.util import read_json  # noqa: E402

mcp = FastMCP("online-decode")

ROOT = Path(__file__).resolve().parents[1]
# 默认读仓库内合成示例包；生产使用时用环境变量指向自己的指纹包与插件 MCP 目录。
DEFAULT_PACKAGE = os.environ.get("ONLINE_DECODE_PACKAGE") or str(ROOT / "examples" / "sample_package")
DEFAULT_MCP_REPO = os.environ.get("ONLINE_DECODE_MCP_REPO", "")
DEFAULT_WORKSPACE = os.environ.get("ONLINE_DECODE_WORKSPACE") or str(
    Path.home() / "Pictures" / "Lightroom" / "online_decode_sessions"
)

# 测试可注入：把真 LR 端口换成 FakeLrPort，或把视觉 API 换成 mock
PORT_FACTORY: Callable[[Config], Any] | None = None
API_DRIVER_FACTORY: Callable[[], Any] | None = None


@dataclass
class Handle:
    session_id: str
    cfg: Config
    driver: QueueDriver
    controller: SessionController
    thread: threading.Thread
    photo: str
    package: str
    report: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    aborted: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


SESSIONS: dict[str, Handle] = {}
_LOCK = threading.Lock()


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _error(exc: Exception) -> str:
    return _dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _get(session_id: str) -> Handle:
    with _LOCK:
        handle = SESSIONS.get(session_id)
    if handle is None:
        raise KeyError(f"找不到会话 {session_id}；先调用 decode_start")
    return handle


def _session_root(handle: Handle) -> Path:
    if handle.controller.state:
        return handle.controller.state.root
    # 线程还没跑到建会话，按约定路径推导
    return Path(handle.cfg.workspace) / "sessions" / handle.session_id


def _worker(handle: Handle, intent: str | None) -> None:
    try:
        handle.report = handle.controller.run(
            handle.photo, handle.package,
            user_intent=intent, session_id=handle.session_id,
        )
    except Exception as exc:  # 后台线程异常必须落到 handle，不能静默丢掉
        handle.error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        handle.driver.mark_done()


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------


@mcp.tool()
def decode_capabilities() -> str:
    """环境自检：插件版本、五个动作能力、字段清单数量、默认路径。

    排错第一步。开启会话前先跑一次，确认 Lightroom 自动化监听器在运行。
    """
    try:
        port = QueueLrPort(DEFAULT_MCP_REPO)
        caps = port.capabilities()
        fields = Fields.load()
        return _dump({
            "ok": True,
            "plugin": caps,
            "fields": {"v1": len(fields.v1), "version": fields.raw.get("fields_version")},
            "defaults": {
                "package": DEFAULT_PACKAGE,
                "mcp_repo": DEFAULT_MCP_REPO,
                "workspace": DEFAULT_WORKSPACE,
            },
            "host_note": "MCP 宿主不支持 sampling；模型判断通过 decode_next/submit 回传",
        })
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def decode_contract(task: str) -> str:
    """取某类任务的响应契约（JSON Schema）与填写要点。

    task: scene | brief | plan | evaluate。首次拿到任务包时调用一次即可。
    """
    mapping = {
        "scene": contracts.SCENE,
        "brief": contracts.BRIEF,
        "plan": contracts.PLAN,
        "evaluate": contracts.EVALUATION,
    }
    if task not in mapping:
        return _dump({"ok": False, "error": f"task 必须是 {sorted(mapping)}"})
    name = mapping[task]
    hints = {
        "scene": "只看 original 图判场景：primary_candidates 带分数即可，不要写 XMP。"
                 "桶参数会在下一段 brief 给出，避免先验污染分类。",
        "brief": "看 original 图 + 命中桶的锚点/先验，给出支系判断、preserve/goals/avoid，"
                 "以及一份 initial_xmp 文本（可用 decode_fields 查可写字段与值域）。",
        "plan": "交一份完整 XMP 文本（xmp 字段）或 xmp_path；param_notes 可选。"
                "anchors 是桶锚点，priors 只作参考；写 not_evaluable 字段会被拒。"
                "不要在 XMP 里写 rdf:Seq 曲线，也不要写 tier=ext 字段。",
        "evaluate": "四个 verdict + improvement + decision + evidence[]；"
                    "evidence 里带 metric 时，数值必须与任务包 metrics 对得上（容差 0.02）。",
    }
    return _dump({"ok": True, "task": task, "hint": hints[task], "schema": contracts.load(name)})


@mcp.tool()
def decode_fields() -> str:
    """列出模型可以在 XMP 里写的字段、类型与值域（源自插件 XmpMapping.lua）。

    只需在写第一份 XMP 前查一次；字段名写错会被守卫拒绝并记入 rejected[]。
    """
    try:
        fields = Fields.load()
        out = {}
        for key in sorted(fields.v1):
            spec = fields.spec(key) or {}
            out[key] = {"type": spec.get("type"), "range": spec.get("range"),
                        "group": spec.get("group")}
        return _dump({
            "ok": True,
            "fields_version": fields.raw.get("fields_version"),
            "count": len(out),
            "fields": out,
            "aliases": fields.aliases,
            "note": "tier=ext 字段（grain_amount / vignette_amount / sharpening_amount 等）不可写",
        })
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def decode_verify_package(package: str = "") -> str:
    """校验指纹包完整性（逐文件哈希、fingerprint_hash、package_hash）。"""
    try:
        pkg = verify_package(package or DEFAULT_PACKAGE)
        return _dump({
            "ok": True,
            "fingerprint_id": pkg.fingerprint_id,
            "status": pkg.status,
            "application_ready": pkg.application_ready,
            "buckets": pkg.bucket_ids,
            "files_checked": len(pkg.file_hashes),
        })
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def decode_start(
    photo: str,
    package: str = "",
    workspace: str = "",
    mcp_repo: str = "",
    max_rounds: int = 5,
    intent: str = "",
) -> str:
    """启动一次单图会话（后台线程跑完整算法），立即返回 session_id。

    之后反复调用 decode_next → 看图 → decode_submit，直到 decode_status 显示完成。
    photo 必须是已在 Lightroom catalog 里的照片路径；workspace 必须是 ASCII 路径。
    """
    try:
        cfg = Config(
            max_rounds=max_rounds,
            driver="queue",
            workspace=Path(workspace or DEFAULT_WORKSPACE),
            mcp_repo=mcp_repo or DEFAULT_MCP_REPO,
        )
        cfg.validate()
        port = PORT_FACTORY(cfg) if PORT_FACTORY else QueueLrPort(cfg.mcp_repo)
        io_dir = Path(cfg.workspace) / "mcp_io"
        driver = QueueDriver(io_dir=io_dir)
        controller = SessionController(cfg, port, driver)
        session_id = make_session_id()
        handle = Handle(
            session_id=session_id, cfg=cfg, driver=driver, controller=controller,
            thread=None, photo=photo, package=package or DEFAULT_PACKAGE,  # type: ignore[arg-type]
        )
        handle.thread = threading.Thread(
            target=_worker, args=(handle, intent or None),
            name=f"decode-{session_id}", daemon=True,
        )
        with _LOCK:
            SESSIONS[session_id] = handle
        handle.thread.start()
        return _dump({
            "ok": True,
            "session_id": session_id,
            "max_rounds": cfg.max_rounds,
            "workspace": str(cfg.workspace),
            "next": "调用 decode_next(session_id) 取第一个任务包，再用 decode_contract 查响应格式",
        })
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def decode_next(session_id: str, timeout: float = 20.0) -> str:
    """取下一个待答任务包（含图片路径、指标、历史、填写提示）。

    没有待答任务时返回当前状态；若后台线程已完成，直接提示去取报告。
    """
    try:
        handle = _get(session_id)
        item = handle.driver.next_task(timeout=timeout)
        if item is None:
            return _dump({
                "ok": True, "pending": False, "done": handle.driver.is_done,
                "hint": "暂无待答任务；用 decode_status 查看进度，或稍后重试 decode_next",
                "error": handle.error,
            })

        req = item.request
        brief = req.get("brief") or {}
        limits = req.get("limits") or {}
        packet: dict[str, Any] = {
            "ok": True,
            "pending": True,
            "session_id": session_id,
            "key": item.key,
            "task": item.task,
            "round": item.round,
            "images": req.get("images"),
            "max_rounds": limits.get("max_rounds"),
        }
        if item.task == "scene":
            packet["fingerprint"] = req.get("fingerprint")
            packet["user_intent"] = req.get("user_intent")
            packet["hint"] = ("只看 original 图 → 给 primary_candidates 分数与光线/亮度/逆光；"
                              "不要写 XMP。用 decode_contract('scene') 查必填项。")
        elif item.task == "brief":
            packet["fingerprint"] = req.get("fingerprint")
            packet["scene"] = req.get("scene")
            packet["anchors"] = (req.get("scene") or {}).get("anchors")
            packet["priors"] = (req.get("scene") or {}).get("priors")
            packet["baseline_settings"] = req.get("baseline_settings")
            packet["user_intent"] = req.get("user_intent")
            packet["style_strength"] = req.get("style_strength")
            packet["hint"] = ("看 original 图 + anchors（桶锚点）/priors（宽离散先验）→ "
                              "branch_rationale、preserve/goals/avoid，并产出 initial_xmp。"
                              "锚点尽量贴近，span 越大允许偏离越多；priors 只作参考，不是固定目标。"
                              "用 decode_fields 查可写字段，用 decode_contract('brief') 查必填项。")
        elif item.task == "plan":
            packet["goals"] = brief.get("goals")
            packet["avoid"] = brief.get("avoid")
            packet["history"] = req.get("history")
            packet["last_evaluation"] = req.get("last_evaluation")
            packet["hint"] = ("看 current 预览 → 交一份完整 XMP 文本（xmp 字段）。"
                              "前两轮贴近锚点（band_until_round）；第 3 轮起可按本图自由调整；"
                              "末轮只做收敛或回退。")
        else:
            packet["goals"] = brief.get("goals")
            packet["avoid"] = brief.get("avoid")
            packet["metrics"] = req.get("metrics")
            packet["adjustments"] = req.get("adjustments")
            packet["history"] = req.get("history")
            packet["hint"] = ("对照 original 与 current 预览 → 四个 verdict + improvement + "
                              "decision + evidence[]。evidence 带 metric 时数值必须来自本包 metrics。")

        packet["response_schema_tool"] = f"decode_contract(task='{item.task}')"
        return _dump(packet)
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def decode_submit(session_id: str, key: str, response: dict) -> str:
    """把模型判断回传给后台算法，推进到下一轮或收尾。

    key 必须来自 decode_next 的返回；答错 key 会被拒绝，避免把第 2 轮的答案塞进第 1 轮。
    """
    try:
        handle = _get(session_id)
        if not handle.driver.submit(key, response):
            return _dump({
                "ok": False,
                "error": "任务 key 无效或已答过；调用 decode_next 取最新的 key",
                "pending_keys": handle.driver.pending_keys(),
            })
        return _dump({"ok": True, "submitted": key, "next": "调用 decode_next(session_id)"})
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def decode_status(session_id: str) -> str:
    """查看会话进度：状态、已用轮次、最佳轮、停止原因、待答任务、报告是否就绪。"""
    try:
        handle = _get(session_id)
        state = handle.controller.state
        payload: dict[str, Any] = {
            "ok": True,
            "session_id": session_id,
            "thread_alive": handle.thread.is_alive() if handle.thread else False,
            "driver_done": handle.driver.is_done,
            "report_ready": handle.report is not None,
            "error": handle.error,
            "aborted": handle.aborted,
            "pending_keys": handle.driver.pending_keys(),
        }
        if state:
            payload.update({
                "status": state.status,
                "substep": state.substep,
                "max_rounds": state.max_rounds,
                "rounds_used": len(state.rounds),
                "best_round": (state.best or {}).get("round"),
                "stop_reason": state.stop_reason,
                "warnings": state.warnings,
                "session_path": str(state.root),
            })
        return _dump(payload)
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def decode_report(session_id: str) -> str:
    """取完整 decode_report.json（含每轮参数、预览、评价、哈希链、成片信息）。"""
    try:
        handle = _get(session_id)
        report = handle.report or read_json(_session_root(handle) / "decode_report.json")
        if report is None:
            return _dump({"ok": False, "error": "报告尚未生成；先看 decode_status"})
        return _dump(report)
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def decode_abort(session_id: str, reason: str = "用户中止") -> str:
    """中止会话：让等待中的模型调用超时退出，控制器保持可恢复状态。"""
    try:
        handle = _get(session_id)
        count = handle.driver.fail_all(reason)
        handle.aborted = True
        return _dump({"ok": True, "unblocked": count,
                      "hint": "会话不会被标成 failed；可用 CLI resume 从落盘状态续跑"})
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def decode_run(
    photo: str,
    package: str = "",
    workspace: str = "",
    mcp_repo: str = "",
    max_rounds: int = 5,
    intent: str = "",
) -> str:
    """一键自动跑完整算法（需要外部视觉 API，同步阻塞数分钟）。

    前提：环境变量 ONLINE_DECODE_VISION_KEY（以及可选 URL / MODEL）已配置。
    未配置时明确报错——MCP 宿主不支持 sampling，server 无法反向
    调用宿主模型，所以默认请走 decode_start/next/submit 任务编排。
    """
    if not is_configured() and API_DRIVER_FACTORY is None:
        return _dump({
            "ok": False,
            "error": "未配置 ONLINE_DECODE_VISION_KEY，无法一键自动运行",
            "reason": "MCP 宿主不支持 sampling/createMessage，server 不能反向调用宿主模型",
            "alternative": "用 decode_start → decode_next → 看图 → decode_submit 的任务编排路径",
        })
    try:
        cfg = Config(
            max_rounds=max_rounds, driver="api",
            workspace=Path(workspace or DEFAULT_WORKSPACE),
            mcp_repo=mcp_repo or DEFAULT_MCP_REPO,
        )
        cfg.validate()
        port = PORT_FACTORY(cfg) if PORT_FACTORY else QueueLrPort(cfg.mcp_repo)
        driver = API_DRIVER_FACTORY() if API_DRIVER_FACTORY else ApiDriver()
        controller = SessionController(cfg, port, driver)
        report = controller.run(photo, package or DEFAULT_PACKAGE, user_intent=intent or None)
        return _dump(report)
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def decode_sessions() -> str:
    """列出本 MCP 进程里已启动的会话（便于跨轮对话找回 session_id）。"""
    with _LOCK:
        items = [
            {"session_id": sid, "photo": h.photo, "done": h.driver.is_done,
             "report_ready": h.report is not None}
            for sid, h in SESSIONS.items()
        ]
    return _dump({"ok": True, "count": len(items), "sessions": items})


if __name__ == "__main__":
    mcp.run()

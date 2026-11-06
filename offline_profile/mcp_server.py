"""MCP 工具集：由宿主 agent/子代理完成看图，Python 负责校验与聚合。

不实现 sampling——当前 MCP 宿主 对 `sampling/createMessage` 返回
`Method not found`，所以采用任务编排：

    profile_start → profile_tasks → agent 看图 → profile_submit → profile_finalize

启动（stdio）：
    python offline_profile/mcp_server.py
"""

from __future__ import annotations

import json
import os
import sys

_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from offline_profile.agent_flow import (  # noqa: E402
    DEFAULT_AGENT_MODEL,
    DEFAULT_AGENT_VERSION,
    build_status,
    finalize_build,
    list_tasks,
    start_build,
    submit_batch,
)

mcp = FastMCP("offline-profile")


def _dump(payload) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _error(exc: Exception) -> str:
    return _dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


@mcp.tool()
def profile_start(
    input_dir: str,
    build_root: str,
    limit: int = 0,
    overwrite: bool = False,
    agent_model: str = DEFAULT_AGENT_MODEL,
    agent_version: str = DEFAULT_AGENT_VERSION,
) -> str:
    """初始化一次 agent 建档：校验预处理输入，生成逐图任务与 prompt。

    limit=0 表示全部 ready 图片；agent_model/agent_version 会写进 model_info。
    返回任务数、前三个任务和后续调用方式。
    """
    try:
        return _dump(
            start_build(
                input_dir,
                build_root,
                limit=limit,
                overwrite=overwrite,
                agent_model=agent_model,
                agent_version=agent_version,
            )
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def profile_tasks(build_root: str, status: str = "open", limit: int = 3) -> str:
    """取待分析任务。status: open（pending+invalid）/ pending / invalid / submitted / all。

    每个任务给出 image_id、预览路径、特征路径、prompt 路径与 input_fingerprint。
    单次最多返回 10 个，避免上下文过大。
    """
    try:
        return _dump(list_tasks(build_root, status=status, limit=limit))
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def profile_submit(build_root: str, submissions: list[dict], replace: bool = False) -> str:
    """提交单图或小批分析 JSON。

    submissions 形如 [{"image_id": "...", "analysis": {...}}]；analysis 只需
    scene_observation / visual_observation / edit_hypotheses / quality /
    local_adjustments_suspected / overall_impression / warnings，身份字段由本工具注入。
    单次最多 10 张；被拒绝的条目会返回具体 Schema/语义错误，可修正后重提。
    """
    try:
        return _dump(submit_batch(build_root, submissions, replace=replace))
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def profile_finalize(build_root: str) -> str:
    """收口：对已提交的分析做分桶、训练/留出划分，写 manifest/report，并锁定构建。"""
    try:
        return _dump(finalize_build(build_root))
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def profile_status(build_root: str) -> str:
    """查看任务计数与构建状态。"""
    try:
        return _dump(build_status(build_root))
    except Exception as exc:
        return _error(exc)


if __name__ == "__main__":
    mcp.run()

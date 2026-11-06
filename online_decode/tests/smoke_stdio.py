"""online-decode MCP 的 stdio 冒烟测试：真的起一个 server，列工具、调只读工具。

单元测试证明不了"stdio 协议跑得通、stdout 没被日志污染"。这条链路出错时
症状是客户端直接断开却不报原因，所以留一个可手动跑的脚本。

    python online_decode/tests/smoke_stdio.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).resolve().parents[1] / "mcp_server.py"
EXPECTED = {
    "decode_capabilities",
    "decode_run",
    "decode_contract",
    "decode_fields",
    "decode_verify_package",
    "decode_start",
    "decode_next",
    "decode_submit",
    "decode_status",
    "decode_report",
    "decode_abort",
    "decode_sessions",
}


async def main() -> int:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print("tools:", len(names), names)

            def text_of(result) -> str:
                return "\n".join(
                    c.text for c in result.content if getattr(c, "type", "") == "text"
                )

            info = json.loads(text_of(await session.call_tool("decode_capabilities", {})))
            print("plugin:", (info.get("plugin") or {}).get("plugin_version"),
                  "| plugin_ready:", (info.get("plugin") or {}).get("plugin_ready"),
                  "| fields:", info.get("fields"))

            fields = json.loads(text_of(await session.call_tool("decode_fields", {})))
            print("writable fields:", fields.get("count"))

            contract = json.loads(text_of(await session.call_tool("decode_contract", {"task": "plan"})))
            print("plan schema id:", (contract.get("schema") or {}).get("$id"))

            sessions = json.loads(text_of(await session.call_tool("decode_sessions", {})))
            print("sessions:", sessions.get("count"))

    if set(names) != EXPECTED:
        print("工具集合不符：", set(names) ^ EXPECTED, file=sys.stderr)
        return 1
    if not info.get("ok") or fields.get("count") != 65 or not contract.get("ok"):
        print("只读工具返回异常", file=sys.stderr)
        return 2
    print("[ok] stdio 链路通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

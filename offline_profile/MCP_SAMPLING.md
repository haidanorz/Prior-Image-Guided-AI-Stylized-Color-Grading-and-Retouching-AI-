# MCP Sampling 探针：失败记录

日期：2026-11-04

## 结论

MCP 宿主 **没有实现 MCP `sampling/createMessage`**。服务端发起反向请求时返回：

```json
{"ok": false, "error": "McpError: Method not found"}
```

- `probe_image`（带图片）与 `probe_text`（纯文本）返回**同一个错误**；
- 图片路径与文件已验证存在（`profile_build/probe/probe.png`，349 字节）；
- 失败发生在 host 处理 sampling 请求的方法分发阶段，图片根本没走到模型。

因此“服务端请求 MCP 宿主 调模型”的路线在当前版本不可用，探针工具已从
`mcp_server.py` 移除。

## 保留物

- `profile_build/probe/probe.png`：左红右蓝的测试图；
- MCP 客户端配置里的旧 `offline-profile-probe` 条目已移除；
- 配置备份：`mcp.json.bak-<时间>`。

## 当前方案

改用任务编排：`profile_start → profile_tasks → agent/子代理看图 → profile_submit →
profile_finalize`。详见 `MCP_AGENT.md`。

# MCP 任务编排（当前可用方案）

MCP 宿主 对 MCP `sampling/createMessage` 返回 `Method not found`，所以不能用
“服务端反向请求 host 调模型”。改用任务编排：**Python 出任务，agent/子代理看图，提交 JSON，
Python 校验并收口**。

## 1. 工具

| 工具 | 作用 |
| --- | --- |
| `profile_start(input_dir, build_root, limit=0, overwrite=false)` | 校验输入、冻结配置、生成任务与逐图 prompt |
| `profile_tasks(build_root, status="open", limit=3)` | 取任务（含预览路径、prompt 路径、fingerprint），单次最多 10 个 |
| `profile_submit(build_root, submissions, replace=false)` | 提交 1～10 份分析 JSON，逐条校验并落盘 |
| `profile_finalize(build_root)` | 分桶、10 折划分、写 manifest/report，锁定构建 |
| `profile_status(build_root)` | 查看计数与状态 |

构建目录新增：

```text
build_root/
├── mcp/
│   ├── state.json              # 输入/配置/agent 身份/近重复组
│   ├── tasks.jsonl             # 逐图任务状态
│   └── prompts/<image_id>.txt  # 每张图的完整分析提示词
├── analyses/raw|normalized/    # 提交后写入
├── assignments.jsonl
├── evaluation_split.json
├── manifest.jsonl
└── run_report.json
```

## 2. 在 MCP 宿主里怎么跑

1. 连接器管理 → 自定义连接器 → 找到 `offline-profile` → **信任/启用**；
2. **新开会话**（工具 schema 在会话启动时注入）；
3. 对 agent 说：

> 用 offline-profile 给 `预处理输出` 建档，build_root 用 `profile_build/demo`，
> 先 limit=5。每批取 3 个任务，读 prompt 和预览图，按 Schema 返回分析 JSON，
> 用 profile_submit 提交；全部提交后 profile_finalize，并跑 verify。

agent 会：

1. `profile_start(...)` → 拿到任务数与前三个任务；
2. `profile_tasks(build_root, limit=3)` → 拿到 `image_id`、`preview_path`、`prompt_path`、`input_fingerprint`；
3. 读 prompt 文件、打开预览图，生成分析；
4. `profile_submit(build_root, [{"image_id": ..., "analysis": {...}}])`；
5. 重复 2–4，直到 `profile_tasks` 返回 `remaining=0`；
6. `profile_finalize(build_root)`，再运行 `python -m offline_profile verify --build-root ...`。

## 3. 提交的 JSON 长什么样

`analysis` 只需观察与假设，身份字段由 `profile_submit` 注入：

```json
{
  "image_id": "img_...",
  "analysis": {
    "scene_observation": {
      "primary_candidates": [{"label": "natural_landscape", "score": 0.85}],
      "lighting": "daylight",
      "subject": "a coastal headland",
      "brightness": "mid",
      "backlit": false,
      "backlit_evidence": "front-lit grass, no rim light",
      "environment_notes": "open coast"
    },
    "visual_observation": {
      "contrast": "neutral", "shadow_detail": "high", "highlight_rolloff": "high",
      "black_level": "neutral", "warmth": "neutral", "saturation_global": "high",
      "saturation_skin": "unknown", "saturation_sky": "neutral", "saturation_foliage": "high",
      "shadow_tint_strength": "low", "highlight_tint_strength": "low",
      "grain_visibility": "neutral", "vignette_visibility": "neutral",
      "sharpness_impression": "high", "clarity_impression": "high"
    },
    "narrative_observation": {
      "mood": "安静、开阔，带一点旅途中的疏离感",
      "atmosphere": "海风把草地压向一侧，画面节奏缓慢，天空留出大量呼吸空间。",
      "light_quality": "薄云把日光打散，阴影柔和，海面只剩低对比的亮带。",
      "sense_of_place": "开阔的海岸高地，远处海平线把空间推向无限。",
      "seasonal_cues": "草色偏干，花朵零散，像是夏末或初秋。",
      "composition_rhythm": "前景草坡压低视线，礁石与海平线形成两条缓慢的水平节奏。",
      "keywords": ["开阔", "海风", "薄雾", "干草"],
      "basis": "低对比的海面、被风吹斜的草、浅色天空与远处模糊的海平线"
    },
    "edit_hypotheses": {
      "highlights": {"estimate": -12, "range": [-20, -4], "confidence": 0.6, "evidence": "bright cloud tops retain texture"},
      "vibrance": {"estimate": 10, "range": [4, 16], "confidence": 0.6, "evidence": "greens are lifted without sky shift"}
    },
    "quality": {"overall": "good", "issues": [], "confidence": 0.8},
    "local_adjustments_suspected": [],
    "overall_impression": "A fresh coastal scene with controlled highlights.",
    "warnings": []
  }
}
```

非法键、越界值、缺证据、区间不含估计会被整条拒绝，返回具体错误；修正后用
`replace=true` 重提。

## 4. 子代理任务模板

父 agent 取 3 个任务后，可给每个子代理一段：

```text
你只分析一张照片，不要做聚合。
1. 读取 <prompt_path> 的完整提示词并严格遵守。
2. 打开 <preview_path> 查看图片。
3. 只返回一个 JSON 对象，包含 scene_observation / visual_observation /
   edit_hypotheses / quality / local_adjustments_suspected /
   overall_impression / warnings。
4. 不要包含 image_id、source_sha256、input_fingerprint、model_info，父代理会注入。
5. 不要输出 markdown 或解释。
```

父 agent 汇总后调用 `profile_submit`；子代理不必直接调 MCP 工具。

## 5. 检查点

构建未 finalize 时，可随时只读试算分桶与划分，不写文件：

```bash
python -m offline_profile checkpoint --build-root "profile_build/demo_full"
```

输出 submitted/pending/invalid、core/associate/unassigned、各桶 n_core 与
`full/provisional/exception_group`、`split_status` 和 split_hash。建议每 15 张看一次。

## 6. 批量与限制

- 单次 `profile_tasks` 最多 10 个，`profile_submit` 最多 10 份；建议每批 3～5 张。
- 建议每批 3～5 张，用子代理并行分析、父代理串行提交，避免上下文爆炸。
- 每张图都要读 6.8 KB 的 prompt 和 1600px 预览，token 成本主要在这里。
- `agent_in_the_loop` 会写进报告：分析由会话生成，不是固定模型端点；换会话不保证一致。
- 构建一旦 `finalize` 就锁定；要重跑请换 build_root 或 `overwrite=true`。

## 7. 为什么不用 sampling

`MCP_SAMPLING.md` 记录了探针结果：`probe_text` 与 `probe_image` 都返回
`McpError: Method not found`。这是 host 未实现该协议方法，不是图片或路径问题；
因此任何“服务端请求 host 调模型”的写法在当前版本都走不通。

## 8. 与 CLI 的关系

- `python -m offline_profile run --adapter replay`：回放已冻结的响应集；
- `python -m offline_profile run --adapter mock`：合成响应，供契约测试；
- MCP 任务编排：agent 现场看图，适合 5～69 张的人工辅助建档；
- 三者共用同一套 Schema、规范化、分桶、划分与 `verify`。

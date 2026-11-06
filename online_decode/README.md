# online_decode · 在线个性化解码

输入一张新图 + 离线指纹包，走 n 轮 Lightroom 真实反馈闭环后出片。

架构与边界见 `session.py`、`params/guard.py` 的注释与本文第 5 节；本文只讲怎么用、代码在哪、还差什么。

---

## 1. 当前状态

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 配置入口、字段清单、错误码、状态文件 | **已完成** |
| M1 | 插件 / MCP 反馈接口 | 已完成（build 18，真机 AC-10 通过） |
| M2 | 单图 n 轮闭环（agent 驱动，模型交 XMP）+ 中断续跑 | **已完成** |
| M3 | 真机联调（真实 LR + 真实读数） | **已完成** |
| M4 | JSON Schema 契约 + Golden 回放 | **已完成** |
| M5 | MCP 工具化：完整算法进工具层 + 一键 API 路径 | **已完成**，12 个工具，stdio 冒烟通过 |
| M6 | 顺序批处理 | 未开始 |

> 默认 `max_rounds=5`（范围仍 1–5）。**强制最少轮数 `min_rounds` 默认 2**（2026-11-06 决策）：
> 首轮只认 `abort`，`stop_and_select` / 模型请求停止都会被执行到第 2 轮——首轮交的是桶中心
> 参数（已是"解"），一轮即停会把"达到桶中心"误判成"本图最优"。`max_rounds=1` 时退化为 1。
> 早停判据按 n 自动推导：n ≤ 3 为「单轮无改善即停」，n ≥ 4 为「连续两轮无改善才停」。
> 生效值写进 `session.json` 与 `decode_report.json`，报告里的 `source=derived` 记录这一推导。
> 长预算档的「连续两轮无改善才停」已在 `eval/selector.py` 实现并有测试固定。
> 强制轮次内 `plan.stop=true` 被解释为「保持上轮参数再验证一次」，而不是跳过该轮。

---

## 2. 在 MCP 宿主里启用（推荐）

在 MCP 客户端的配置文件里注册 `online-decode`（与另外两个 server 并存）：

```json
"online-decode": {
  "command": "python",
  "args": ["/ABS/PATH/TO/REPO/online_decode/mcp_server.py"]
}
```

在连接器管理页**信任**它，然后**新开一个会话**（工具 schema 在会话启动时注入）。
先跑 `decode_capabilities` 确认插件就绪，再按任务编排推进：

```
decode_start(photo, max_rounds=5)
   → decode_next()            取任务包（含图片路径 / 指标 / 历史）
   → 用宿主的读图能力看 original / current 预览
   → decode_submit(key, 响应)  回传结构化判断
   → 重复直到 decode_status 显示完成
   → decode_report()          取完整报告
decode_contract(task)  查该类任务的 JSON Schema
decode_fields()        查 65 个可写 XMP 字段与值域
```

**完整算法都在 MCP server 里**：指纹校验、桶判定、参数守卫、哈希链、n 轮状态机、
早停、最佳轮、恢复、导出。宿主只提供视觉判断。控制器跑在后台线程，
`QueueDriver` 把三次模型调用变成可跨工具暂停的请求/响应。

> 为什么不是"一句话全自动"：MCP 宿主 **不支持 sampling**，
> server 不能反向调用宿主模型（项目里 offline_profile 已实测）。
> 若确实需要一键跑完，配置 `ONLINE_DECODE_VISION_KEY`（可选 URL / MODEL）
> 后用 `decode_run(photo)`，它会走 OpenAI 兼容的视觉 API。未配置时该工具明确报错，
> 不伪装成能自动跑。该调用同步阻塞数分钟。

---

## 3. 命令行快速开始

```bash
# 0) 字段清单：每次改 XmpMapping.lua 后重新生成，并会与插件声明数对数（65）
python -m online_decode.tools.dump_fields
python -m online_decode.tools.dump_fields --check   # CI 用

# 1) 校验指纹包
python -m online_decode verify-package --package examples/sample_package

# 2) 跑一次会话（agent 驱动：控制器写请求包，宿主 Agent 读图后写回响应）
python -m online_decode run \
    --photo "D:/photos/IMG_0001.jpg" \
    --package examples/sample_package \
    --max-rounds 5 \
    --mcp-repo "/ABS/PATH/TO/AI-XMP-Color-Importer/mcp" \
    --intent "保留天空灰蓝层次"

# 3) 查看会话
python -m online_decode inspect --session sessions/s_1762...

# 4) 等模型超时后接着跑（已完成的轮次不重做，已写好的响应会被复用）
python -m online_decode resume --session sessions/s_1762... --mcp-repo ".../mcp"

# 5) 回归基线
python -m online_decode.tools.record_golden --check
python -m online_decode.tools.dump_fields --check
```

运行时会打印请求包路径，例如：

```
[agent] brief r00 请求包：sessions/s_xxx/agent_io/req_brief_00.json
        写回响应：sessions/s_xxx/agent_io/resp_brief_00.json
```

把响应写回同名 `resp_*.json` 即可继续；超时后会话不会作废，可 `resume`（见 §7）。

---

## 4. 代码地图

```
online_decode/
├── config.py            唯一配置入口（max_rounds 默认 5，范围 1–5，越界即失败）
├── errors.py            稳定错误码 + 可执行建议
├── state.py             session.json 读写（原子替换，只追加）
├── ports.py             Driver 协议（brief / plan / evaluate）
├── session.py           状态机：预检 → 基线 → brief → n 轮 → 收尾
├── report.py            decode_report.json
├── cli.py               命令行（run / resume / inspect / verify-package）
├── mcp_server.py        MCP 工具层（12 个工具；后台线程跑控制器）
├── contracts.py         JSON Schema 校验（严格模式默认开）
├── schema/              model_request / brief / plan / evaluation 四份契约
├── drivers/             agent（文件交换）/ queue（MCP 编排）/ api（外部 VLM）/ scripted（测试）
├── fingerprint/         包校验、桶判定、CreativeBrief
├── params/              字段清单、台账、基线锚定、守卫、XMP 渲染
├── feedback/            LR 端口（真 / 假）、单轮执行器
├── eval/                客观指标、证据核验、早停与选轮
├── tools/dump_fields.py 从 XmpMapping.lua 生成字段清单
├── tools/record_golden.py 录制 / 比对回归轨迹
├── data/                fields_manifest.json（产物，勿手改）
└── tests/               114 项 + golden/cases.json + smoke_stdio.py，无需 Lightroom
```

依赖方向单向：`session → (fingerprint | params | feedback | eval | report) → errors`。

---

## 5. 三条不可让步的约束

1. **模型交 XMP，控制器做闸门**。每轮产出是一份 `model.xmp`（原样存档），
   解析 → 守卫校验 → 规范化成 `rNN.xmp` → apply。写入 LR 的是后者，两者哈希都进报告。
   原因：Lightroom 对不认识的属性名静默丢弃，原样 apply 会得到"没报错也没变化"。
2. **哈希链**：`apply.settings_hash == render.settings_hash`，不等则该轮预览作废、不评价、不继续。
3. **轮次预算只认 render**：apply 重试、dry-run、响应重做都不消耗轮次；n 只在配置里，模型改不了。
4. **参考带宽按桶内分散度放宽**：

   ```text
   allowance = max(|ref_value|, min(span × 1.0, 25), 5) × (0.5 + clamp(strength, 0.3, 2.0))
   ```

   `span = p75 − p25`。桶参数越分散，允许偏离中心越多；上限 25 防止极端 span 把约束放大到失效。
   **带宽只在前两轮生效**（`band_until_round=2`）：首轮定方向、次轮收敛；第 3 轮起允许按本图自由调整，
   仅在该值原本会超出带宽时记一条 `notes`，不再拒绝。
   当前包里只有 `natural saturation`（span 20）与 `natural clarity`（span 15）被放宽，
   其余锚点仍由 `|value|` 主导。`wide_dispersion` 参数不进 references，因此没有带宽，只作先验。

---

## 6. 模型侧的四个响应

### scene（桶为主结构专用，第一段）

只回 `scene_observation`：`primary_candidates`（带分数）+ 光线/亮度/逆光。
**这一轮不写 XMP、也看不到桶参数**，避免先验污染场景判断。控制器拿它跑
`decide_bucket`（与离线同一个函数），结果落到 `scene.json` 供续跑复用。

### brief（第二段，命中桶之后）

请求里多了 `scene.anchors`（该桶 published 参数，含 role/span）、`scene.priors`
（宽离散先验）与 `scene.narrative`。必填：`branch_rationale`、`preserve`、
`goals`(2–5)、`avoid`、`initial_xmp`。
`goals` 必须能用眼睛复核，写了"更有氛围感"会被 `brief.validate` 拒掉。

### plan（第 2 轮起）

```json
{"xmp": "<x:xmpmeta ... crs:Highlights2012=\"-12\" ...>",
 "param_notes": {"Highlights2012": {"evidence_level": "runtime_reference",
                                    "ref": "fingerprint.global.highlights",
                                    "reason": "云层高光压一档"}},
 "expected_effects": ["云层层次恢复"]}
```

- `xmp` 或 `xmp_path` 二选一；文本原样存档为 `rounds/rNN/model.xmp`。
- `param_notes` 可选，用于补逐项理由与证据等级；缺省记 `planner_judgement` + `ref=model:xmp`。
- 兼容旧写法 `adjustments[]`（会走同一套守卫），但**以 XMP 为主**。
- **指纹未覆盖（not_evaluable）的字段可自主写**（2026-11-06 起）：解码者可按本图
  调整 HSL / 白平衡 / Whites / Dehaze 等指纹未担保的参数，证据等级逐项记在台账
  entry 上；不再要求 `target_adaptive` 通道，也不额外告警。仍被拒的只有：
  字段不在清单、`tier != v1`（接口写不进）、自标 `unsupported`。
- 曲线（`rdf:Seq`）首版解析不了，会明确报错而不是静默丢掉。

### evaluate

必填四个 verdict + `improvement` + `decision` + `evidence[]`。
`evidence[].anchor` 只能用受控方位词；带 `metric` 的必须与 `metrics.json` 对得上（容差 0.02），
否则该评价作废、允许重做一次且不消耗轮次。

---

## 7. 产物

```
sessions/<session_id>/
├── session.json            唯一状态真相
├── creative_brief.json
├── decode_report.json
├── agent_io/               req_/resp_ 请求响应（审计用）
├── rounds/
│   ├── r00_original.jpg
│   └── r01/{plan.json,param_ledger.json,model.xmp,r01.xmp,snapshot.json,metrics.json,evaluation.json,preview.jpg}
└── final/{final.jpg,best_settings.snapshot.json,restore.result.json}
```

`param_ledger.json` 里的 `rejected[]` 解释"这轮为什么少调了某项"，不要跳过它。

---

## 8. 已知限制

- **CLI 只有 agent 驱动**；MCP 层提供 queue（任务编排）与 api（一键，需 key）两种驱动。
- **workspace 必须 ASCII**（XMP 产物路径）；源照片路径含中文只发 warning，真机已验证可定位。
- **曲线不由模型设定**。`ToneCurvePV2012` 会被守卫拒；恢复已有曲线走 `restore_settings`。
- **tier=ext 字段不可写**：`grain_amount / vignette_amount / sharpening_amount` 会被判 `FIELD_UNSUPPORTED`。

---

## 9. 测试

```bash
python -m unittest discover -s online_decode/tests -t .
```

114 项，覆盖：配置越界、守卫七道闸、温度 shift 锚定、桶判定与回退、XMP 解析与拒绝、
预览哈希漂移、评价不可核验、首轮全拒、单轮无改善早停、5 轮预算、agent 文件交换超时、契约 schema、中断续跑、Golden 回放、MCP 工具链、mock 视觉 API。

**故意复现的两个坑**（真机上出现过）：
- `FakeLrPort(render_hash_drift=True)` → 预览哈希不匹配，该轮作废；
- 评价里写一个不存在的 `metric` → 必须被 `verify_evidence` 拦下。

---

## 10. 还没做（按优先级）

1. **顺序批处理**：多图排队、失败隔离、汇总报告（LR 写权限下不并行）。
2. **合同完善**：`session.json` / `decode_report.json` 的正式 schema。
3. **真机复测**：指纹包升级为 application_ready 后，用不同支系的图各跑一次
   （海岸冷调 / 夜景城市），验证 AC-02 的真实个性化差异。

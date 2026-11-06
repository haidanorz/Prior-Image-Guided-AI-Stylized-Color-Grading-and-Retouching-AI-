# 离线建档后处理（offline_profile）

消费预处理阶段的 `manifest.jsonl`、`previews/` 与 `features/`，完成：

1. 单图视觉分析（场景观察、视觉观察、编辑假设）；
2. 响应规范化与非法输出隔离；
3. 场景分桶（core / associate / unassigned）；
4. 按桶分层、近重复组同侧的训练/留出划分。

分析协议、分桶与统计口径见本文第 3 节与代码注释。
当前已包含：统计聚合、桶参数表、10 折交叉验证、桶级叙事、指纹包；
仍不包含：XMP 批量应用、LR 真实渲染、`dist/` 发布判定；
§6.5 的“低置信度/场景冲突独立复核”也留到接入真实模型后实现（Mock 无法产生独立第二意见）。

## 1. 当前范围与硬边界

- 只读取预处理产物中 `status=ready` 的记录，不修改任何输入。
- 支持 **Mock 与 Replay** 两种离线适配器：从 `--mock-dir` 读取 `<image_id>.json`；
  Replay 额外要求 `_model_info.json`，用于回放 MCP 宿主会话或真实模型冻结的响应。
  真实模型端点尚未接入，接入时实现 `analysis/adapter.py` 的同一协议即可。
- 构建状态一律为草稿：所有报告写 `validation_scope=analysis_consistency`、
  `application_ready=false`，不会生成 XMP、不会调用 Lightroom。
- 不生成任何“已校准概率”、LR 参数或局部蒙版；`edit_hypotheses` 只是可见证据支持的假设。

## 2. 运行

```bash
python -m offline_profile run ^
  --input "预处理输出" ^
  --build-root "profile_build" ^
  --mock-dir "mock_responses"
```

```bash
python -m offline_profile verify --build-root "profile_build"
```

参数：

| 参数 | 说明 |
| --- | --- |
| `--input` | 预处理输出目录（只读） |
| `--build-root` | 构建输出目录，不得等于或位于输入目录内 |
| `--mock-dir` | 响应目录，文件名 `<image_id>.json`（mock / replay 共用） |
| `--adapter` | `mock` 或 `replay`，覆盖配置 |
| `--config` | 可选 JSON 配置，未知键直接报错 |
| `--overwrite` | 允许覆盖配置/版本不一致的旧构建 |
| `--strict` | 覆盖率不足或存在 invalid/error 时退出码 1 |

Mock 响应文件可以是完整分析 JSON，也可以是 `{"raw_text": "..."}` 包装；后者用于测试
非法格式、Markdown 围栏、重试与错误隔离。

### Replay：回放 MCP 宿主 或真实模型的冻结响应

```bash
python -m offline_profile run ^
  --input "预处理输出" ^
  --build-root "profile_build" ^
  --mock-dir "frozen_responses" ^
  --adapter replay
```

目录内除 `<image_id>.json` 外必须有 `_model_info.json`：

```json
{"provider": "agent", "model": "agent-vision", "version": "2026-11-04", "is_mock": false}
```

`provider=agent` 会在报告里写入 `agent_in_the_loop` 警告：响应由会话生成，
不是固定模型端点；可依赖已保存的原始响应回放，但换会话不保证一致。批量或需要重复
构建时应使用固定模型端点。

## 3. 单图分析契约

规范化后写入 `analyses/normalized/<image_id>.json`，顶层字段：

- 版本与身份：`analysis_schema_version`、`image_id`、`source_sha256`、`input_fingerprint`、
  `prompt_version`、`model_info`。
- `scene_observation`：候选场景（分数归一化到 1）、光照、主体、亮度、逆光与证据、环境备注。
- `visual_observation`：15 个固定序数属性（`very_low | low | neutral | high | very_high | unknown`）。
- `narrative_observation`：同一次响应内的中文意境层（mood/atmosphere/light_quality/
  sense_of_place/seasonal_cues/composition_rhythm/keywords/basis），只描述可见线索，
  不得写地点、身份或数值参数。分析 Schema 1.1.0 / prompt v2 起为必填。
- `edit_hypotheses`：只允许参数登记表中的键，每项 `estimate / range / confidence / evidence`。
- `quality`、`local_adjustments_suspected`、`overall_impression`、`warnings`。

硬约束：

- 结构由 `schema/analysis.schema.json` 校验；键、枚举、范围、区间包含关系在
  `analysis/normalize.py` 校验。越界、缺证据、ID 不匹配、NaN 一律判非法，不 clamp。
- 候选分数同标签去重后归一化；亮度以客观亮度 P50 映射为准，模型改写会被覆盖并记警告。
- 原始响应保存在 `analyses/raw/<image_id>.json`，包含每次尝试的原文与错误，便于审计。

## 4. 参数登记表

`schema/parameters.json` 首版登记：

- 17 个编辑参数：`exposure_ev`、`contrast`、`highlights`、`shadows`、`whites`、`blacks`、
  `temperature_shift`、`tint_shift`、`vibrance`、`saturation`、`clarity`、`dehaze`、
  `sharpening_amount`、`grain_amount`、`vignette_amount`、`shadow_tint_hue`、`highlight_tint_hue`。
- 每个参数含单位、合法范围、误差归一化尺度 `scale`、实务零带 `practical_zero_band`、
  显著性阈值 `significance_threshold` 与方向（signed / positive / circular）。
- 15 个视觉属性，统一使用六档序数枚举。
- 改任何一项都必须提升 `registry_version`，否则统计口径会悄悄变化。

## 5. 分桶与划分

- 成员等级：`top_score >= 0.65` 且 `margin >= 0.15` 为 core；`>= 0.45` 为 associate；
  其余 unassigned；`other` 永不成为正式桶。
- 桶规模：core ≥ 8 为 full，5–7 为 provisional，< 5 为 exception_group。
- 默认 `validation_mode=kfold`（10 折交叉验证）：全部 core 参与估计，不再扣掉 20% 留出；
  按 `SHA256(split_seed|fold|成员)` 排序后贪心均衡分折，`n_core` 小于折数时按单位数封顶。
- `validation_mode=holdout` 保留旧路径：`n_core >= 10` 且留出后训练 ≥ 8 才划分，
  留出数为 `max(1, round(n_core × 0.20))`。
- 两种模式下，同一近重复组必须同折/同侧；跨桶的近重复组会阻断相关桶的划分（保守处理）。
- 保存 `evaluation_split.json`，含 `validation_mode`、折/留出成员、近重复单位与 `split_hash`
  （`split_hash_of()` 由构建与 verify 共用，避免口径漂移）。

## 6. 输出结构

```text
build_root/
├── analyses/
│   ├── raw/<image_id>.json          # 每次尝试的原始响应与错误
│   └── normalized/<image_id>.json   # 通过 Schema 与语义校验的分析
├── cache/<cache_key>.json           # 模型响应缓存
├── assignments.jsonl                # 每张有效分析图的分桶记录
├── evaluation_split.json            # 按桶的 K 折（或留出）划分
├── episodes.jsonl                   # 重试、非法响应、失败（只追加）
├── manifest.jsonl                   # 每个 ready 输入一行
├── run_report.json                  # 计数、覆盖率、版本、指纹
└── config.resolved.json             # 最终配置 + config_hash + 版本
```

## 7. 测试

```bash
python -m unittest discover -s offline_profile/tests -t . -v
```

覆盖：配置校验、Schema/语义规范化、非法输出隔离、候选归一化、亮度覆盖、
分桶阈值、划分确定性/近重复同侧/跨桶阻断、缓存复用、构建状态门禁、CLI 与 verify。

## 8. MCP 接入

MCP 宿主 未实现 `sampling/createMessage`（探针返回 `Method not found`），
因此使用任务编排：`profile_start → profile_tasks → agent/子代理看图 → profile_submit →
profile_finalize`。工具说明见 `MCP_AGENT.md`，失败记录见 `MCP_SAMPLING.md`。

## 9. 统计与档案合成

构建 finalize 后，用已提交的规范化分析生成“桶参数表 + 全局兜底”：

```bash
python -m offline_profile.profile_compose "profile_build/demo_full"
```

输出：

```text
global_signature.json     # 跨桶稳定参数，角色是 fallback
profile.json              # profile_shape + bucket_parameters + cv_eval + merge 校验
buckets/<id>/stats.json   # 桶内稳健统计（全部 core）
buckets/<id>/delta.json   # 相对全局的增量（审计用，可为空）
buckets/<id>/profile.json # 桶风格指纹：parameters + 证据哈希 + profile_hash
buckets/index.json        # 各桶 profile_hash 索引
```

`profile_compose` 会先写 `qa/analysis_coverage.json` 并检查 `edit_hypotheses` 覆盖度：
均值低于 `analysis_min_mean_hypotheses` 或关键参数覆盖低于 `analysis_min_key_coverage` 时
直接阻止 compose/package（可用 `--allow-underfilled` 仅作调试）。这是为了防止叙事字段
挤掉数值假设，产出空的风格指纹。

当前状态语义：档案 `status=draft`；桶参数逐项 `published / neutral / wide_dispersion / withheld`；
`application_ready=false`。

`cv_eval`（10 折，默认）逐桶逐参数给出 out-of-fold 的 `mae_global_only`、
`mae_bucket_only`、`relative_improvement` 与 Bootstrap 90% 区间。命中全局参数的
对比基线是 global，其余参数的基线是**跨桶池化中位数**（`baseline_kind=pooled`）；
每折重算桶心与基线，不重复稳定性门禁。

`cv_eval` 与 `holdout_eval` 都是探索性证据：未按 `scale_p` 归一化、未做 P_bucket 全集门槛，
`gate_applied=false`，不能直接作为 §13 的发布判据。发布前仍需补齐归一化误差与整体门槛。

### 桶为主结构（`profile_shape=bucket_primary`，默认）

每个桶直接给出**绝对参数**，不依赖全局零点；global 退化为兜底：

```text
bucket.parameters[key] = {value, unit, p25, p75, span, n, consistency, cv_vs_pooled, status}
status: published（可直接作锚点）/ neutral（值在零带内，不调）
        / wide_dispersion（只作先验，进 bucket.priors）
        / withheld（证据不足或留出更差，附 reason）
```

运行时规则：

```text
命中桶 -> 用 bucket.parameters
桶内缺该键 -> 用 global（role=global_fallback）
两边都没有 -> not_evaluable，跳过该滑块
未命中桶 -> 只用 global
永远不要把一个桶的参数叠加到全局上
```

`profile_shape=global_plus_delta` 保留旧行为（global + delta），用于历史对比。
`fingerprint.json` 的 `shape` / `global_role` / `buckets[].parameters` / `buckets[].priors`
`/ buckets[].not_evaluable` 由该开关决定；`fingerprint_schema_version=1.1.0`。

### 小样本桶诊断（`diagnostic_cv`）

provisional / exception 桶会在 `evaluation_split.json` 里拿到 `diagnostic_folds`（每折 1 张，
最多 10 折），compose 时对它们跑只读诊断，写进 `profile.diagnostic_cv`：

- `used_in_baseline=false`、`publishable=false`：不进入 `stats`，不影响全局基线，不产生 delta；
- 逐参数给出 `bucket_central`、`candidate_delta`、out-of-fold MAE 与 Bootstrap 区间；
- 额外给出 `fold_central_min/max/span`：折间跨度若达到一个档位（5 个 index），说明中位数
  被单张图左右，结论不可用。

这些数字只用于判断“是否值得补样本”，不能写进指纹包，也不能据此发布。

加权中位数在累计权重恰好等于一半时取两侧中点。
`bucket_primary` 下，单桶证据直接成为该桶的绝对参数（不传播给别的桶）；
`global_plus_delta` 下，单桶证据永不进入全局基线或桶增量。

## 10. 特征指纹包（离线编码器最终产物）

```bash
python -m offline_profile package ^
  --build-root "profile_build/demo_full" ^
  --out-dir "build/package" --overwrite
```

包内只放编码结果与证据，不含 XMP、原图或单图分析原文：

```text
fingerprint.json          # 运行时指纹：global + per-bucket delta + runtime_rule
global_signature.json     # 全局证据
profile.json              # 档案索引
buckets/index.json
buckets/<id>/{profile,delta,stats}.json
package_manifest.json     # 逐文件 SHA-256 + package_hash
README.md
```

`fingerprint.json` 带 `fingerprint_id`/`fingerprint_hash`；包带 `package_id`/`package_hash`，
状态为 `draft`、`application_ready=false`。包内桶参数仍为 draft，求解器只应读取
`fingerprint.json`，其余文件用于证据与审计。

## 11. 桶级语言描述

数值档案之外，每个桶可挂一份 `narrative.json`，只描述语境、光线、色彩、构图与情绪，
不复制任何数值。流程：

```bash
# 1. 生成逐图描述与归纳 prompt（只取训练侧 core）
python -m offline_profile narrative-collect --build-root "profile_build/demo_full" --bucket natural_landscape

# 2. 父代理读 prompt + descriptions.jsonl，写 narrative JSON
# 3. 校验并写入
python -m offline_profile narrative-submit --build-root "profile_build/demo_full" ^
  --bucket natural_landscape --file "narrative_drafts/natural_landscape.json"

# 4. 重算桶指纹并重新打包
python -m offline_profile.profile_compose "profile_build/demo_full"
python -m offline_profile package --build-root "profile_build/demo_full" ^
  --out-dir "build/package" --overwrite
```

`narrative-collect` 会优先使用 v1.1 分析里的 `narrative_observation`；旧 v1.0 分析没有该字段时，
退回到 `subject/environment_notes/overall_impression/edit evidence` 的抽取。

`narrative.json` 带 `narrative_hash`，并通过 `buckets/<id>/profile.json.narrative_ref` 进入
桶指纹；`fingerprint.json` 的桶条目里也有 `narrative` 引用。**全局指纹不添加任何描述性字段**。
描述只使用训练成员，`evidence_refs` 必须是训练成员 image_id；正文不得出现数值参数。

## 12. Pilot XMP（仅试看，不属于指纹包）

```bash
python -m offline_profile xmp-export ^
  --build-root "profile_build/demo_full" ^
  --out-dir "profile_build/demo_full/xmp_pilot"
```

生成 `pilot_global.xmp`、`pilot_natural_landscape.xmp`、`pilot_urban_architecture.xmp`
与 `xmp_manifest.json`。桶预设 = 全局兜底，再用该桶 `published` 参数覆盖；
`wide_dispersion` 参数会标为 `pilot_only`（`--strict` 可排除）。
色温/色调/分离色调没有可靠映射，保持 As Shot 并跳过。

所有输出 `status=pilot_only`、`application_ready=false`，不修改 `profile.json`；
只适合 2–3 张测试图回看，不能批量套用或当作已发布档案。

## 13. 依赖

```bash
python -m pip install -r requirements.txt
```

Python ≥ 3.10，`jsonschema` ≥ 4.0。本层不依赖 numpy / Pillow，也不访问网络。

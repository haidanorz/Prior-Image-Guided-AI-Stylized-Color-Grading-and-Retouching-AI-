**简体中文** | [English](README.en.md)

# 基于先验图片的 AI 自动风格化调色修图

## 这个项目是干什么的

用相机拍完照片，往往还需要用 Photoshop 或 Lightroom 逐张调色，才能得到理想的视觉效果——这是一个费时费力的过程。

而这些照片里有大量场景重复、调色思路可以复用的片子，例如漫展的 cosplay 场照、不同地方出游的风景照。

本项目提供基于 AI 的方案：从先前的作品集中提取可复用的调色风格，自动适配到新的照片上，从而免去大量照片调色的工作量。

技术上说，它从一组**已调色的参考成片**中提取可审计的调色先验，再对一张**新照片**走多轮 Lightroom 真实渲染反馈，收敛到该照片自己的最优 Develop 设置。

适用于「有一批自己满意的成片，希望把这种处理习惯迁移到新照片」的场景，也适用于需要把调色决策留下证据链的工程环境。

---

## 为什么用语言 / 多模态 AI 处理纯视觉任务

目前 GitHub 和期刊论文上已经有不少基于 CV 模型和深度学习的自动化调色方案。本项目选择另一条路径，依据是作者的一个观点：

如摄影调色这类依靠师徒关系口口相传并训练的技能载体是言语和视觉的知识，而非大量的参数统计和数学拟合，且调色修图本身就是一个基于主观审美的benchmark，不存在一致性标准，因此作者认为这类以言语为基础的技能习得可以利用大模型本身强大的语言能力和学习记忆库而非各类CV算法来实现

此外，基于多模态大模型实现的这个方案还有另一个好处，即本方案的模型接口可以切换为任意模型的API（如Chatgpt、DeepSeek等），随着模型能力的进步本方案的效果也许可以不断的提升，对比于传统CV模型的架构难以升级，只能微调参数，本模型可以直接升级核心能力模块。

---

## 快速上手

最短路径，只依赖 Python；真正出片才需要 Lightroom。

```bash
# 1) 取代码、装依赖（Python >= 3.10）
git clone <repo-url>
cd <repo-dir>
python -m pip install -r requirements.txt

# 2) 校验仓库自带的示例指纹包
python -m online_decode verify-package --package examples/sample_package

# 3) 跑测试，确认环境正常（不需要 Lightroom）
python -m unittest discover -s online_decode/tests -t .
```

前三步约一分钟。想真正跑一张图，把 `--photo` 换成你自己的照片：

```bash
python -m online_decode run \
  --photo "D:/photos/IMG_0001.jpg" \
  --package examples/sample_package \
  --workspace "D:/decode_sessions" \
  --max-rounds 5 \
  --intent "保留天空层次"
```

会话启动后会打印 `agent_io/req_*.json` 的路径，按提示写回同名 `resp_*.json` 即可推进；  
中断后用 `python -m online_decode resume --session <session_id>` 继续，已完成的轮次不会重做。

想换成自己的风格集，则依次跑 `offline_preprocess` → `offline_profile` →  
`offline_profile package`，完整命令与参数见 [第 3 节](#3-快速开始)；真机出片所需的  
Lightroom 插件见 [第 4 节](#4-真机反馈依赖)。

---

## 1. 它由三段组成

```text
参考成片 ──► offline_preprocess ──► 预览 + 客观特征 + 清单
                                          │
                                          ▼
                              offline_profile（编码）
                     单图分析 → 场景分桶 → 稳健统计
                                          │
                                          ▼
                              指纹包（draft，可校验哈希）
                                          │
新照片 ────────────────────────────────────┤
                                          ▼
                              online_decode（解码）
              scene 判桶 → brief 定目标 → n 轮 plan/evaluate
                    → 选最佳轮 → 恢复并导出
```

| 阶段       | 包                    | 输入                         | 输出                                     |
| -------- | -------------------- | -------------------------- | -------------------------------------- |
| 预处理      | `offline_preprocess` | 成片目录                       | sRGB 标准预览、客观特征 JSON、manifest/errors 清单 |
| 编码（离线建档） | `offline_profile`    | 预处理产物 + 单图分析响应             | 全局签名、场景桶参数、桶叙事、指纹包                     |
| 解码（在线出片） | `online_decode`      | 新照片 + 指纹包 + Lightroom 反馈端口 | 逐轮 XMP、评价、最佳轮快照、成片与完整报告                |

### 编码：先验是怎么来的

1. 每张参考图由 Agent 看图，产出结构化的场景观察、视觉观察与编辑假设；响应先规范化再入库。
2. 按场景分桶（`natural_landscape` / `urban_architecture` / `street` / `indoor` / `still_life`），核心样本进桶，边缘样本只作关联。
3. 在同一桶内做稳健统计：中位数、四分位距、支持度、符号一致性；跨桶稳定的项升级为全局签名，桶内特有的项作为条件增量（delta），离散过大的项只记为 `prior_only`，不参与硬约束。
4. 桶级叙事由该桶的描述集合归纳而来，供解码时理解「这个桶的语言」，而不是机械套参数。
5. 导出指纹包：逐文件 SHA-256 + package_hash，任何文件被改动都会在校验时失败。

包内结构：

```text
sample_package/
├── package_manifest.json      # 逐文件哈希 + package_hash
├── fingerprint.json           # 解码端唯一需要读取的文件
├── global_signature.json      # 全局签名
├── profile.json               # 构建元信息
├── README.md
└── buckets/<bucket_id>/
    ├── profile.json           # 桶参数（published / wide_dispersion / withheld）
    ├── delta.json             # 相对全局的增量
    ├── stats.json             # 分位与离散度
    └── narrative.json         # 桶级叙事
```

### 解码：闭环怎么走

1. `scene`：只给原图，判场景桶；这一轮看不到桶参数，避免先验污染分类。
2. `brief`：给出该桶锚点、宽离散先验与叙事，模型交 `preserve / goals / avoid / initial_xmp`。
3. `plan`：第 2 轮起交完整 XMP；控制器做字段白名单、值域、参考带宽与类型校验，拒绝项写进台账。
4. `evaluate`：每轮由 Lightroom 真实导出预览，模型按可核验指标给 verdict 与 improvement；无改善达到阈值即停。
5. 选最佳轮（不是最后一轮），恢复快照并导出原尺寸成片；全程哈希链闭合才认一轮。

---

## 2. 目录

```text
.
├── offline_preprocess/     # 预处理：色彩管理、缩放、特征、pHash、校验
├── offline_profile/        # 离线建档：分析协议、分桶、统计、叙事、指纹包
├── online_decode/          # 在线解码：状态机、参数守卫、XMP、LR 反馈、MCP 工具
├── examples/
│   ├── make_sample_package.py   # 生成合成示例指纹包
│   └── sample_package/          # 已生成的示例包（全合成数据）
├── requirements.txt
├── requirements-mcp.txt
└── LICENSE
```

各模块的详细说明见其目录内 `README.md`。

---

## 3. 快速开始

```bash
python -m pip install -r requirements.txt      # 核心依赖：numpy / Pillow / jsonschema
```

### 3.1 校验示例指纹包

```bash
python -m online_decode verify-package --package examples/sample_package
```

### 3.2 跑测试（不需要 Lightroom）

```bash
python -m unittest discover -s offline_preprocess/tests -t .
python -m unittest discover -s offline_profile/tests -t .
python -m unittest discover -s online_decode/tests -t .
```

共 230 项。`online_decode` 用假 LR 端口与脚本驱动覆盖状态机、守卫、哈希链、早停、恢复与 Golden 回放。

### 3.3 编码自己的参考集

```bash
# 1) 预处理：预览 + 客观特征 + 清单
python -m offline_preprocess \
  --input-dir /path/to/finished_photos \
  --output-dir ./build/preprocess

# 2) 逐图分析 → 分桶 → 统计 → 建档
#    --mock-dir 指向已冻结的逐图分析响应（格式见 offline_profile/README.md 的单图分析契约）
python -m offline_profile \
  --input ./build/preprocess \
  --build-root ./build/profile \
  --mock-dir ./responses

# 3) 校验并导出指纹包
python -m offline_profile verify --build-root ./build/profile
python -m offline_profile package --build-root ./build/profile --out-dir ./build/package
```

若不使用 `--mock-dir`，可通过 `offline_profile/mcp_server.py` 的任务编排工具  
（`profile_start` → `profile_tasks` → `profile_submit` → `profile_finalize`）让 Agent 逐图提交分析，  
再走上面的 `verify` / `package`。

### 3.4 解码一张新照片

```bash
python -m online_decode run \
  --photo "D:/photos/IMG_0001.jpg" \
  --package examples/sample_package \
  --workspace "D:/decode_sessions" \
  --max-rounds 5 \
  --intent "保留天空层次"
```

产物在 `workspace/sessions/<session_id>/`：`session.json`（唯一状态真相）、`rounds/rNN/`（模型 XMP、校验后 XMP、台账、快照、指标、评价、预览）、`final/`（成片与最佳快照）。

### 3.5 在 MCP 宿主里使用

```bash
python -m pip install -r requirements-mcp.txt
```

在 MCP 客户端里注册两个 server（按实际绝对路径填写）：

```json
{
  "mcpServers": {
    "offline-profile": { "command": "python", "args": ["/ABS/PATH/TO/REPO/offline_profile/mcp_server.py"] },
    "online-decode":  { "command": "python", "args": ["/ABS/PATH/TO/REPO/online_decode/mcp_server.py"] }
  }
}
```

在线解码的环境变量：

| 变量                         | 作用                                                     |
| -------------------------- | ------------------------------------------------------ |
| `ONLINE_DECODE_PACKAGE`    | 默认指纹包路径（缺省为 `examples/sample_package`）                 |
| `ONLINE_DECODE_MCP_REPO`   | 插件 MCP 目录（真机反馈必需）                                      |
| `ONLINE_DECODE_WORKSPACE`  | 会话目录（缺省 `~/Pictures/Lightroom/online_decode_sessions`） |
| `ONLINE_DECODE_VISION_KEY` | 可选；配置后 `decode_run` 可走 OpenAI 兼容视觉 API                 |

---

## 4. 真机反馈依赖

在线解码的 `evaluate` 阶段需要 Lightroom 的真实导出与读数，由配套插件提供：

- 仓库：[AI-generated-xmp-file-interface-plugin-for-lightroom-cc](https://github.com/haidanorz/AI-generated-xmp-file-interface-plugin-for-lightroom-cc)
- 组件：Lightroom Classic 插件（写 Develop 设置 / 渲染预览 / 恢复快照 / 导出）+ 本地 MCP 接口
- 没有插件时可以跑全部单元测试与 Golden 回放（假端口），但不能产出真实成片

---

## 5. 边界与诚实声明

- 指纹包默认 `status=draft`、`application_ready=false`；在线解码只应将其作为 **pilot 先验**，不声称复现摄影师的稳定风格。
- 解码不保证可复现：同一张图两次运行可能走不同轮次，这是反馈闭环的预期行为，不是缺陷。
- 参数写入有硬闸门：字段不在清单、类型不符、超出参考带宽、接口写不进的项一律拒绝，并逐条记录原因。
- 曲线与局部蒙版不在模型可控范围内；需要时走快照恢复。
- 所有图像处理只读输入、只写输出目录；预览默认剥离 EXIF/GPS。

## 6. 许可

MIT，见 [LICENSE](LICENSE)。

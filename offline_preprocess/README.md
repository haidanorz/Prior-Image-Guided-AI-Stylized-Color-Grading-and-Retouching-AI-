# 离线建档预处理（offline_preprocess）

把目录中的**已完成调色成片**转换为统一、可复核、可重复生成的：

- `previews/<image_id>.jpg`：sRGB、8-bit RGB、长边 ≤ 1600 的标准预览（不含 EXIF/GPS）
- `features/<image_id>.json`：客观图像特征（亮度、RGB、Lab、色相、饱和度、主色、空间分布、锐度、pHash）
- `manifest.jsonl` / `errors.jsonl` / `near_duplicates.json` / `run_report.json` / `config.resolved.json`

设计说明见本文与代码注释；需求文档不随本仓库发布。
本模块是全新独立实现，不读取、不修改、不依赖任何旧 Demo、旧插件、旧队列、旧 XMP 或旧风格档案。

## 1. 环境要求

| 依赖 | 版本 | 说明 |
| --- | --- | --- |
| Python | ≥ 3.10 | Windows 10/11 |
| numpy | ≥ 1.24 | 统计与颜色计算 |
| Pillow | ≥ 10.0 | 解码、缩放、JPEG 编码、ICC |
| jsonschema | ≥ 4.0 | 可选，`verify` 与测试用于 Schema 校验 |

> **Pillow 必须带 littlecms**。PyPI 官方 wheel 自带；部分 conda 构建没有。
> 可用 `python -c "from PIL import features; print(features.check('littlecms2'))"` 检查。
> 没有 littlecms 时：
> - 无 ICC 的图和 ICC 恰为标准 sRGB 的图仍可处理（后者走可证明的恒等转换）；
> - 内嵌 AdobeRGB / Display P3 等需要真实色彩转换的图会明确报 `ICC_ENGINE_UNAVAILABLE`，
>   **不会**静默按 sRGB 解释。

安装：

```bash
python -m pip install -r requirements.txt
```

国内网络若直连 PyPI 超时，可临时用镜像：

```bash
python -m pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple
```

已验证环境：

- Python 3.13.14 + numpy 2.5.4 + Pillow 13.0.0（littlecms=True）：完整 36 项测试通过；
- Python 3.11.7 + numpy 1.26.4 + Pillow 10.2.0（littlecms=False）：35 项通过、1 项跳过
  （非 sRGB ICC 的真实转换无法在该环境执行）。

## 2. 运行

```bash
python -m offline_preprocess ^
  --input-dir "/path/to/finished_photos" ^
  --output-dir "./build/preprocess"
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--config my.json` | 覆盖默认配置（未知键会直接报错） |
| `--no-recursive` | 只扫描顶层目录 |
| `--overwrite` | 忽略既有清单，全部重新处理 |
| `--fail-on-error` | 存在 error 记录时退出码为 1 |
| `--quiet` / `--json` | 静默 / JSON 摘要 |

输出目录与输入目录**不得相同**；输出目录若位于输入目录内，扫描时会自动排除它，避免把生成图当输入。

### 校验输出

```bash
python -m offline_preprocess verify --output-dir "./build/preprocess"
```

只读检查：清单计数闭环、预览可解码、尺寸/ICC/哈希一致、特征 JSON 通过 Schema、
直方图与主色比例归一、无 NaN/Infinity、`duplicate_of` 不悬空。

## 3. 输出结构

```text
output_dir/
├── previews/<image_id>.jpg
├── features/<image_id>.json
├── manifest.jsonl          # 每个输入一行
├── errors.jsonl            # 只含 error
├── near_duplicates.json    # 仅提示复核，不删除、不合并
├── run_report.json         # 计数、错误码分布、软件版本
└── config.resolved.json    # 最终配置 + config_hash
```

`image_id = "img_" + sha256(原文件字节)[:16]`，改名或移动不改变 ID。
产物文件名全部 ASCII；原始中文路径只保存在清单里，UTF-8 不转义。

每张输入在 `manifest.jsonl` 中恰好一条记录，状态为：

- `ready`：预览与特征生成成功
- `duplicate_exact`：与更早出现的文件字节完全相同，不重复生成产物
- `skipped`：`unsupported_alpha`（含 Alpha）或 `too_small`（长边 < 512）
- `error`：解码 / ICC / 写入 / 校验失败，带稳定错误码与阶段

## 4. 处理口径（要点）

- 原图只读：运行前后 SHA-256 不变；所有产物先写临时文件再原子替换。
- 顺序：哈希 → 完全重复 → 解码 → EXIF 方向 → ICC 到 sRGB → 缩放 → 特征 → JPEG 编码 → 回读校验。
- 统计全部来自**缩放后、JPEG 编码前**的内存图像；预览也由同一数组编码，不从 JPEG 反读。
- 缩放：`scale = min(1, L / max(W, H))`，只缩不放；长边定点为 L，短边 `floor(x + 0.5)`；
  每图最多一次 Lanczos 重采样，不裁剪、不补边、不拉伸。
- 预览固定 JPEG q=90、8-bit RGB、写入 sRGB ICC、不写 EXIF/GPS。
- 近重复：64 位 DCT pHash，Hamming ≤ 5 只写报告，`review_required=true`。

## 5. 续跑与幂等

- 默认不覆盖有效结果：只有源哈希、`preprocess_version`、`feature_schema_version`、
  `config_hash` 全部一致，且预览哈希匹配、特征 JSON 可解析时，才跳过。
  预览的可解码性在写入时已完整解码校验；续跑不再重复解码，`verify` 命令会再做一次深校验。
- 配置/版本变化时拒绝混用旧特征，要求新输出目录或 `--overwrite`。
- 预览或 JSON 被截断后再次运行，会识别为不完整并重新生成。

## 6. 测试

```bash
python -m unittest discover -s offline_preprocess/tests -t . -v
```

测试自建合成语料（不依赖旧训练集），覆盖需求文档 §12 与 AC-01 ~ AC-10：
中文/空格/大小写扩展名、EXIF 方向、ICC、灰度、16-bit、Alpha、损坏/截断/伪扩展名、
过小、完全重复/改名重复/近重复、纯黑纯白纯红、灰阶/低饱和/高饱和、极端长宽比、多帧。

## 7. 明确不做

RAW 显影、AI 看图/分桶/风格结论、XMP/preset 生成、Lightroom 调用、网络请求、
按质量自动淘汰、自动删除或合并重复原图。

## 8. 目录

```text
offline_preprocess/
├── cli.py / __main__.py     命令行入口
├── config.py                默认配置、校验、config_hash
├── scan.py                  枚举与输出目录排除
├── imaging.py               解码、方向、ICC
├── geometry.py              尺寸公式与单次重采样
├── color.py                 sRGB/Lab/HSV 数学
├── features.py              特征组装
├── palette.py               Lab k-means 主色
├── similarity.py            pHash
├── pipeline.py              单图流水线与产物校验
├── report.py                清单/错误/近重复/报告
├── verify.py                输出完整性校验
├── schema/feature.schema.json
├── data/srgb.icc            预览写入的标准 sRGB 配置
└── tests/                   合成语料与自动化测试
```

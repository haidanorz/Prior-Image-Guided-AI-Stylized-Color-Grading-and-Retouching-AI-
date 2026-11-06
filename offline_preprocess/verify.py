"""输出目录完整性校验（只读）。

用于交接前自检与 AC-08 的「截断后必须识别为不完整」验证。
"""

from __future__ import annotations

import json
import os
from typing import Any

from PIL import Image

from .errors import Status
from .pipeline import validate_preview_and_feature
from .schema_utils import has_jsonschema, load_feature_schema, validate_feature_payload
from .util import read_json, read_jsonl


def verify_output(output_dir: str) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []

    def add(code: str, message: str, image_id: str | None = None) -> None:
        problems.append({"code": code, "image_id": image_id, "message": message})

    required = ["manifest.jsonl", "errors.jsonl", "near_duplicates.json", "run_report.json", "config.resolved.json"]
    for name in required:
        if not os.path.isfile(os.path.join(output_dir, name)):
            add("MISSING_FILE", f"缺少必需文件：{name}")
    if problems:
        return {"ok": False, "checked": 0, "problems": problems, "output_dir": output_dir}

    resolved = read_json(os.path.join(output_dir, "config.resolved.json"))
    cfg = resolved.get("config", {})
    cfg_hash = resolved.get("config_hash")
    manifest = read_jsonl(os.path.join(output_dir, "manifest.jsonl"))
    errors = read_jsonl(os.path.join(output_dir, "errors.jsonl"))
    report = read_json(os.path.join(output_dir, "run_report.json"))
    near = read_json(os.path.join(output_dir, "near_duplicates.json"))

    ids = {row.get("image_id") for row in manifest if row.get("image_id")}
    counts = {status: 0 for status in Status.ALL}
    ready_records = []
    for row in manifest:
        status = row.get("status")
        if status in counts:
            counts[status] += 1
        else:
            add("BAD_STATUS", f"未知状态：{status!r}", row.get("image_id"))
        if row.get("config_hash") != cfg_hash:
            add("CONFIG_HASH_MISMATCH", "清单记录的 config_hash 与 config.resolved.json 不一致", row.get("image_id"))
        if status == Status.READY:
            ready_records.append(row)
        if status == Status.DUPLICATE_EXACT and row.get("duplicate_of") not in ids:
            add("DANGLING_DUPLICATE", f"duplicate_of 不存在：{row.get('duplicate_of')}", row.get("image_id"))

    if sum(counts.values()) != len(manifest):
        add("COUNT_CLOSED", "状态计数之和与清单行数不一致")
    if report.get("manifest_record_count") != len(manifest):
        add("REPORT_COUNT", "run_report.manifest_record_count 与清单行数不一致")
    if report.get("status_counts") != counts:
        add("REPORT_STATUS", "run_report.status_counts 与清单统计不一致")
    if len(errors) != counts[Status.ERROR]:
        add("ERROR_COUNT", "errors.jsonl 行数与 error 状态数不一致")

    schema = load_feature_schema()
    schema_ready = schema is not None and has_jsonschema()
    schema_checked = 0
    for row in ready_records:
        image_id = row.get("image_id")
        if not validate_preview_and_feature(row, output_dir, deep=True):
            add("ARTIFACT_INVALID", "预览或特征文件缺失/损坏/哈希不符", image_id)
            continue

        feature_path = os.path.join(output_dir, row["feature_path"])
        try:
            payload = read_json(feature_path)
        except Exception as exc:
            add("FEATURE_UNREADABLE", f"特征 JSON 不可读：{exc}", image_id)
            continue

        if schema_ready:
            schema_checked += 1
            for message in validate_feature_payload(payload, schema):
                add("SCHEMA_INVALID", message, image_id)

        _check_semantics(payload, cfg, output_dir, row, add)

    if near.get("review_required") is not True:
        add("NEAR_DUP_FLAG", "near_duplicates.json 必须显式 review_required=true")
    if not report.get("software"):
        add("REPORT_SOFTWARE", "run_report 缺少软件版本信息")

    return {
        "ok": not problems,
        "checked": len(manifest),
        "ready": len(ready_records),
        "schema_validated": schema_checked,
        "problems": problems,
        "output_dir": output_dir,
    }


def _check_semantics(payload: dict, cfg: dict, output_dir: str, row: dict, add) -> None:
    image_id = row.get("image_id")
    geometry = payload.get("geometry", {})
    preview_path = os.path.join(output_dir, row["preview_path"])
    try:
        with Image.open(preview_path) as im:
            size = im.size
    except Exception:
        return
    if size != (geometry.get("output_width"), geometry.get("output_height")):
        add("SIZE_MISMATCH", f"预览尺寸 {size} 与特征记录不一致", image_id)
    long_edge = max(size)
    if long_edge > int(cfg.get("long_edge", 1600)):
        add("LONG_EDGE", f"预览长边 {long_edge} 超过配置上限", image_id)

    hist = payload.get("luminance", {}).get("histogram", {})
    counts = hist.get("counts") or []
    if hist.get("bins") != int(cfg.get("luminance_bins", 64)) or len(counts) != int(cfg.get("luminance_bins", 64)):
        add("HIST_BINS", "亮度直方图 bin 数与配置不一致", image_id)
    elif counts and abs(sum(counts) - 1.0) > 1e-3:
        add("HIST_SUM", f"亮度直方图之和为 {sum(counts):.6f}", image_id)

    hue = payload.get("hue", {})
    if hue.get("bins") != int(cfg.get("hue_bins", 36)) or len(hue.get("histogram") or []) != int(cfg.get("hue_bins", 36)):
        add("HUE_BINS", "色相直方图 bin 数与配置不一致", image_id)
    else:
        hue_sum = float(sum(hue.get("histogram") or []))
        if hue_sum > 0 and abs(hue_sum - 1.0) > 1e-3:
            add("HUE_SUM", "色相直方图之和不为 1（且非全零）", image_id)

    palette = payload.get("palette") or []
    if palette and abs(sum(item.get("ratio", 0) for item in palette) - 1.0) > 1e-3:
        add("PALETTE_SUM", "主色比例之和不为 1", image_id)

    try:
        json.dumps(payload, allow_nan=False)
    except ValueError:
        add("NON_FINITE", "特征 JSON 含 NaN/Infinity", image_id)

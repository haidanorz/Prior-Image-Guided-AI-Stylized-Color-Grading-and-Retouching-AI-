"""输入校验：只消费预处理产物中 status=ready 的记录，并冻结输入指纹。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .contract import load_parameters, load_taxonomy
from .errors import ErrorCode, FatalCode, FatalError
from .util import canonical_hash, canonical_json, read_json, read_jsonl, sha256_bytes, sha256_file
from .version import ANALYSIS_SCHEMA_VERSION


@dataclass(frozen=True)
class IntakeImage:
    image_id: str
    source_path: str
    source_sha256: str
    preview_path: str
    preview_rel: str
    feature_path: str
    feature_rel: str
    preview_sha256: str
    feature_sha256: str
    feature: dict
    record: dict

    @property
    def luminance_p50(self) -> float:
        try:
            return float(self.feature["luminance"]["percentiles"]["p50"])
        except (KeyError, TypeError, ValueError):
            return 0.0


@dataclass
class IntakeResult:
    input_dir: str
    images: list[IntakeImage]
    manifest_total: int
    non_ready: int
    fingerprint: str
    versions: dict
    preprocess_config_hash: str
    near_duplicate_groups: list[list[str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready_count(self) -> int:
        return len(self.images)


def load_input(input_dir: str) -> IntakeResult:
    input_dir = os.path.abspath(input_dir)
    manifest_path = os.path.join(input_dir, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        raise FatalError(FatalCode.INPUT_INVALID, f"缺少 manifest.jsonl：{input_dir}")

    records = read_jsonl(manifest_path)
    if not records:
        raise FatalError(FatalCode.INPUT_MANIFEST_EMPTY, "manifest.jsonl 没有任何记录")

    ready_records = [row for row in records if row.get("status") == "ready"]
    if not ready_records:
        raise FatalError(FatalCode.INPUT_NO_READY, "没有 status=ready 的输入，无法开始建档")

    versions = {
        "preprocess_version": _uniform(ready_records, "preprocess_version"),
        "feature_schema_version": _uniform(ready_records, "feature_schema_version"),
    }
    preprocess_config_hash = _uniform(ready_records, "config_hash")

    images: list[IntakeImage] = []
    seen_ids: set[str] = set()
    for row in ready_records:
        image = _load_image(input_dir, row)
        if image.image_id in seen_ids:
            raise FatalError(FatalCode.INPUT_INVALID, f"输入中出现重复 image_id：{image.image_id}")
        seen_ids.add(image.image_id)
        images.append(image)

    groups, near_warnings = _load_near_duplicates(input_dir, seen_ids)

    fingerprint = _dataset_fingerprint(images, versions, preprocess_config_hash)
    warnings = list(near_warnings)
    non_ready = len(records) - len(ready_records)
    if non_ready:
        warnings.append(f"manifest 中有 {non_ready} 条非 ready 记录，已排除")

    return IntakeResult(
        input_dir=input_dir,
        images=sorted(images, key=lambda item: item.image_id),
        manifest_total=len(records),
        non_ready=non_ready,
        fingerprint=fingerprint,
        versions=versions,
        preprocess_config_hash=preprocess_config_hash,
        near_duplicate_groups=groups,
        warnings=warnings,
    )


def image_fingerprint(image: IntakeImage, *, analysis_config_hash: str, prompt_version: str, model_info: dict, schema_version: str) -> str:
    payload = {
        "image_id": image.image_id,
        "source_sha256": image.source_sha256,
        "preview_sha256": image.preview_sha256,
        "feature_sha256": image.feature_sha256,
        "preprocess_version": image.record.get("preprocess_version"),
        "feature_schema_version": image.record.get("feature_schema_version"),
        "preprocess_config_hash": image.record.get("config_hash"),
        "analysis_config_hash": analysis_config_hash,
        "prompt_version": prompt_version,
        "analysis_schema_version": schema_version,
        "taxonomy_version": load_taxonomy()["taxonomy_version"],
        "parameter_registry_version": load_parameters()["registry_version"],
        "model_info": model_info,
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _uniform(records: list[dict], key: str) -> str:
    values = {str(row.get(key)) for row in records}
    if "None" in values:
        raise FatalError(FatalCode.INPUT_INVALID, f"输入记录缺少 {key}，无法确认版本")
    if len(values) != 1:
        raise FatalError(
            FatalCode.INPUT_INVALID,
            f"输入记录中的 {key} 不唯一：{sorted(values)}；禁止拼接不同版本的分析输入",
        )
    return values.pop()


def _load_image(input_dir: str, row: dict) -> IntakeImage:
    image_id = row.get("image_id")
    if not isinstance(image_id, str) or not image_id:
        raise FatalError(FatalCode.INPUT_RECORD_INVALID, "ready 记录缺少 image_id")
    for key in ("source_path", "source_sha256", "preview_path", "feature_path", "preview_sha256"):
        if not row.get(key):
            raise FatalError(FatalCode.INPUT_RECORD_INVALID, f"{image_id} 缺少字段 {key}")

    preview_path = _inside(input_dir, row["preview_path"], image_id, "preview_path")
    feature_path = _inside(input_dir, row["feature_path"], image_id, "feature_path")
    if not os.path.isfile(preview_path):
        raise FatalError(FatalCode.INPUT_INVALID, f"{image_id} 预览不存在：{preview_path}")
    if not os.path.isfile(feature_path):
        raise FatalError(FatalCode.INPUT_INVALID, f"{image_id} 特征文件不存在：{feature_path}")

    preview_sha = sha256_file(preview_path)
    if preview_sha != row["preview_sha256"]:
        raise FatalError(
            FatalCode.INPUT_INVALID,
            f"{image_id} 预览哈希与 manifest 不一致，输入可能被修改：{preview_sha} != {row['preview_sha256']}",
        )

    try:
        feature = read_json(feature_path)
    except Exception as exc:
        raise FatalError(FatalCode.INPUT_INVALID, f"{image_id} 特征 JSON 不可解析：{exc}") from exc
    if feature.get("image_id") != image_id:
        raise FatalError(FatalCode.INPUT_INVALID, f"{image_id} 特征 JSON 的 image_id 不匹配")
    if feature.get("source_sha256") != row.get("source_sha256"):
        raise FatalError(FatalCode.INPUT_INVALID, f"{image_id} 特征 JSON 的 source_sha256 不匹配")
    if not os.path.isfile(row["source_path"]):
        raise FatalError(FatalCode.INPUT_INVALID, f"{image_id} 原始文件不存在：{row['source_path']}")

    return IntakeImage(
        image_id=image_id,
        source_path=row["source_path"],
        source_sha256=row["source_sha256"],
        preview_path=preview_path,
        preview_rel=str(row["preview_path"]).replace("\\", "/"),
        feature_path=feature_path,
        feature_rel=str(row["feature_path"]).replace("\\", "/"),
        preview_sha256=preview_sha,
        feature_sha256=sha256_file(feature_path),
        feature=feature,
        record=row,
    )


def _inside(root: str, relative: str, image_id: str, field_name: str) -> str:
    candidate = os.path.abspath(os.path.join(root, relative))
    try:
        if os.path.normcase(os.path.commonpath([root, candidate])) != os.path.normcase(root):
            raise ValueError
    except ValueError:
        raise FatalError(
            FatalCode.INPUT_INVALID, f"{image_id} 的 {field_name} 越出输入目录：{relative}"
        ) from None
    return candidate


def _load_near_duplicates(input_dir: str, ready_ids: set[str]) -> tuple[list[list[str]], list[str]]:
    path = os.path.join(input_dir, "near_duplicates.json")
    if not os.path.isfile(path):
        return [], ["输入缺少 near_duplicates.json，划分将不考虑近重复约束"]
    try:
        payload = read_json(path)
    except Exception as exc:
        return [], [f"near_duplicates.json 不可解析，忽略近重复约束：{exc}"]

    groups: list[list[str]] = []
    for group in payload.get("groups", []):
        members = [item.get("image_id") for item in group.get("members", []) if item.get("image_id")]
        members = sorted({m for m in members if m in ready_ids})
        if len(members) >= 2:
            groups.append(members)
    return groups, []


def _dataset_fingerprint(images: list[IntakeImage], versions: dict, preprocess_config_hash: str) -> str:
    payload = {
        "images": [
            {
                "image_id": item.image_id,
                "source_sha256": item.source_sha256,
                "preview_sha256": item.preview_sha256,
                "feature_sha256": item.feature_sha256,
            }
            for item in sorted(images, key=lambda x: x.image_id)
        ],
        "versions": versions,
        "preprocess_config_hash": preprocess_config_hash,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "taxonomy_version": load_taxonomy()["taxonomy_version"],
        "parameter_registry_version": load_parameters()["registry_version"],
    }
    return canonical_hash(payload)

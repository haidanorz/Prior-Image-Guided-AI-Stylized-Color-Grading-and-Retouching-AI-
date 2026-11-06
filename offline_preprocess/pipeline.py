"""单图处理流水线与产物写入。

每张图严格按需求文档 §5 的顺序执行：
枚举/哈希 → 完全重复判定 → 解码 → 方向 → ICC → 缩放 → 特征 → JPEG 编码 → 写入与校验。
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image

from . import features as features_mod
from .errors import ErrorCode, PreprocessError, SkipImage, Stage
from .geometry import compute_target_size, geometry_record, resize_once
from .imaging import load_standardized, srgb_icc_bytes
from .util import atomic_write_bytes, atomic_write_json, now_utc_iso, sha256_bytes, sha256_file
from .version import FEATURE_SCHEMA_VERSION, PREPROCESS_VERSION, STATISTICS_SOURCE


@dataclass
class ProcessOutcome:
    record: dict[str, Any]
    phash: str | None = None
    error_stage: str | None = None


def process_image(
    *,
    source_path: str,
    image_id: str,
    source_sha256: str,
    source_size: int,
    cfg: dict[str, Any],
    cfg_hash: str,
    output_dir: str,
    duplicate_of: str | None,
) -> ProcessOutcome:
    """处理单张图片；任何单图失败都收敛成结构化记录，不抛给批处理层。"""
    base_record = _base_record(image_id, source_path, source_sha256, source_size, cfg_hash)

    if duplicate_of is not None:
        record = dict(base_record)
        record.update(
            {
                "status": "duplicate_exact",
                "duplicate_of": duplicate_of,
                "skip_reason": None,
                "error_code": None,
                "preview_path": None,
                "feature_path": None,
                "preview_sha256": None,
                "warnings": [],
            }
        )
        return ProcessOutcome(record=record)

    try:
        return _process_canonical(
            source_path=source_path,
            image_id=image_id,
            source_size=source_size,
            cfg=cfg,
            cfg_hash=cfg_hash,
            output_dir=output_dir,
            base_record=base_record,
        )
    except SkipImage as skip:
        record = dict(base_record)
        record.update(
            {
                "status": "skipped",
                "duplicate_of": None,
                "skip_reason": skip.reason,
                "error_code": None,
                "preview_path": None,
                "feature_path": None,
                "preview_sha256": None,
                "warnings": [],
            }
        )
        return ProcessOutcome(record=record)
    except PreprocessError as exc:
        record = dict(base_record)
        record.update(
            {
                "status": "error",
                "duplicate_of": None,
                "skip_reason": None,
                "error_code": exc.code,
                "preview_path": None,
                "feature_path": None,
                "preview_sha256": None,
                "error_stage": exc.stage,
                "error_message": exc.message,
                "warnings": [],
            }
        )
        return ProcessOutcome(record=record, error_stage=exc.stage)
    except Exception as exc:  # 兜底：绝不让单图异常终止整批
        record = dict(base_record)
        record.update(
            {
                "status": "error",
                "duplicate_of": None,
                "skip_reason": None,
                "error_code": "INTERNAL_ERROR",
                "preview_path": None,
                "feature_path": None,
                "preview_sha256": None,
                "error_stage": Stage.FEATURES,
                "error_message": f"未预期异常：{type(exc).__name__}: {exc}",
                "warnings": [],
            }
        )
        return ProcessOutcome(record=record, error_stage=Stage.FEATURES)


def _base_record(
    image_id: str, source_path: str, source_sha256: str, source_size: int, cfg_hash: str
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "source_size_bytes": int(source_size),
        "preprocess_version": PREPROCESS_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config_hash": cfg_hash,
        "processed_at": now_utc_iso(),
    }


def _process_canonical(
    *,
    source_path: str,
    image_id: str,
    source_size: int,
    cfg: dict[str, Any],
    cfg_hash: str,
    output_dir: str,
    base_record: dict[str, Any],
) -> ProcessOutcome:
    standardized = load_standardized(source_path, cfg)
    target = compute_target_size(standardized.width, standardized.height, int(cfg["long_edge"]))
    geometry = geometry_record(
        decoded=standardized.decoded_size,
        oriented=standardized.oriented_size,
        target=target,
        long_edge=int(cfg["long_edge"]),
        orientation=standardized.orientation,
        orientation_applied=standardized.orientation_applied,
    )

    image = Image.fromarray(standardized.pixels)
    resized = resize_once(image, target)
    if resized is not image:
        standardized.pixels = _as_rgb_array(resized)

    feature_groups, feature_warnings = features_mod.compute_features(
        standardized.pixels, cfg, standardized.exif_summary
    )
    warnings = _dedupe(standardized.warnings + ([target.warning] if target.warning else []) + feature_warnings)

    feature_payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "image_id": image_id,
        "source_sha256": base_record["source_sha256"],
        "config_hash": cfg_hash,
        "statistics_source": STATISTICS_SOURCE,
        "source": {
            "file_name": os.path.basename(source_path),
            "format": standardized.format,
            "size_bytes": int(source_size),
            "image_mode": standardized.original_mode,
            "was_grayscale": standardized.was_grayscale,
        },
        "geometry": geometry,
        "color_management": {
            "output_color_space": "sRGB",
            "icc_present": standardized.icc_present,
            "icc_description": standardized.icc_description,
            "assumed_srgb": standardized.assumed_srgb,
            "grayscale_converted": standardized.was_grayscale,
            "conversion_path": standardized.color_path,
        },
        "warnings": warnings,
        **feature_groups,
    }

    preview_rel = f"previews/{image_id}.jpg"
    feature_rel = f"features/{image_id}.json"
    preview_path = os.path.join(output_dir, "previews", f"{image_id}.jpg")
    feature_path = os.path.join(output_dir, "features", f"{image_id}.json")

    jpeg_bytes = _encode_jpeg(standardized.pixels, int(cfg["jpeg_quality"]))
    try:
        atomic_write_bytes(preview_path, jpeg_bytes)
        atomic_write_json(feature_path, feature_payload)
    except OSError as exc:
        _remove_quietly(preview_path, feature_path)
        raise PreprocessError(ErrorCode.WRITE_FAILED, Stage.WRITE, f"写入产物失败：{exc}") from exc

    try:
        preview_sha = _verify_artifacts(preview_path, feature_path, image_id, base_record["source_sha256"], target)
    except PreprocessError:
        _remove_quietly(preview_path, feature_path)
        raise

    record = dict(base_record)
    record.update(
        {
            "status": "ready",
            "duplicate_of": None,
            "skip_reason": None,
            "error_code": None,
            "preview_path": preview_rel,
            "feature_path": feature_rel,
            "preview_sha256": preview_sha,
            "warnings": warnings,
        }
    )
    return ProcessOutcome(record=record, phash=feature_groups["similarity"]["phash"])


def _as_rgb_array(image: Image.Image):
    import numpy as np

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _encode_jpeg(pixels, quality: int) -> bytes:
    try:
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(
            buffer,
            format="JPEG",
            quality=int(quality),
            icc_profile=srgb_icc_bytes(),
            optimize=False,
            progressive=False,
        )
        return buffer.getvalue()
    except Exception as exc:
        raise PreprocessError(ErrorCode.ENCODE_FAILED, Stage.ENCODE, f"JPEG 编码失败：{exc}") from exc


def _verify_artifacts(
    preview_path: str, feature_path: str, image_id: str, source_sha256: str, target
) -> str:
    """回读校验：预览可解码、尺寸正确、带 sRGB ICC、无 Orientation；JSON 可解析。"""
    try:
        with Image.open(preview_path) as im:
            im.load()
            if im.format != "JPEG":
                raise PreprocessError(ErrorCode.VERIFY_FAILED, Stage.VERIFY, f"预览格式不是 JPEG：{im.format}")
            if im.mode != "RGB":
                raise PreprocessError(ErrorCode.VERIFY_FAILED, Stage.VERIFY, f"预览不是 8-bit RGB：{im.mode}")
            if im.size != (target.width, target.height):
                raise PreprocessError(
                    ErrorCode.VERIFY_FAILED,
                    Stage.VERIFY,
                    f"预览尺寸不符：{im.size} != {(target.width, target.height)}",
                )
            if not im.info.get("icc_profile"):
                raise PreprocessError(ErrorCode.VERIFY_FAILED, Stage.VERIFY, "预览缺少 ICC")
            orientation = im.getexif().get(274, 1) if hasattr(im, "getexif") else 1
            if int(orientation) != 1:
                raise PreprocessError(ErrorCode.VERIFY_FAILED, Stage.VERIFY, "预览仍带 Orientation 标记")
    except PreprocessError:
        raise
    except Exception as exc:
        raise PreprocessError(ErrorCode.VERIFY_FAILED, Stage.VERIFY, f"预览不可读：{exc}") from exc

    try:
        with open(feature_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        raise PreprocessError(ErrorCode.VERIFY_FAILED, Stage.VERIFY, f"特征 JSON 不可解析：{exc}") from exc

    if payload.get("image_id") != image_id:
        raise PreprocessError(ErrorCode.VERIFY_FAILED, Stage.VERIFY, "特征 JSON 的 image_id 不匹配")
    if payload.get("source_sha256") != source_sha256:
        raise PreprocessError(ErrorCode.VERIFY_FAILED, Stage.VERIFY, "特征 JSON 的 source_sha256 不匹配")

    return sha256_file(preview_path)


def _remove_quietly(*paths: str) -> None:
    for path in paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def validate_preview_and_feature(record: dict, output_dir: str, *, deep: bool = False) -> bool:
    """产物完整性检查；不通过就必须重新生成。

    默认浅校验：预览哈希 + 特征 JSON 可解析 + 关键字段一致。哈希匹配即证明文件与
    写入时逐字节相同，而写入时已经解码验证过格式/尺寸/ICC，因此续跑不必再解码。
    deep=True 时额外完整解码预览（供 verify 审计命令使用）。
    """
    preview_rel = record.get("preview_path")
    feature_rel = record.get("feature_path")
    if not preview_rel or not feature_rel:
        return False
    preview_path = os.path.join(output_dir, preview_rel)
    feature_path = os.path.join(output_dir, feature_rel)
    if not os.path.isfile(preview_path) or not os.path.isfile(feature_path):
        return False
    expected = record.get("preview_sha256")
    if not expected or sha256_file(preview_path) != expected:
        return False
    if deep:
        try:
            with Image.open(preview_path) as im:
                im.verify()
            with Image.open(preview_path) as im:
                im.load()
                if im.format != "JPEG" or im.mode != "RGB":
                    return False
                if not im.info.get("icc_profile"):
                    return False
        except Exception:
            return False
    try:
        with open(feature_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    return (
        payload.get("image_id") == record.get("image_id")
        and payload.get("source_sha256") == record.get("source_sha256")
        and payload.get("config_hash") == record.get("config_hash")
    )

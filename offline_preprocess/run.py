"""批处理编排：扫描 → 哈希 → 完全重复 → 逐图处理 → 汇总产物。"""

from __future__ import annotations

import os
import platform
import time
from typing import Any, Callable

import numpy as np
import PIL
from PIL import features as pil_features

from .config import config_hash, load_config_file, resolve_config, write_resolved_config
from .errors import ErrorCode, FatalCode, FatalError, Stage, Status
from .pipeline import process_image, validate_preview_and_feature
from .report import build_near_duplicates, build_run_report, write_errors, write_manifest, write_near_duplicates, write_run_report
from .scan import normalize_path, scan_inputs
from .util import now_utc_iso, read_jsonl, sha256_file
from .version import FEATURE_SCHEMA_VERSION, PREPROCESS_VERSION

ProgressFn = Callable[[str], None]


def run(
    input_dir: str,
    output_dir: str,
    *,
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    started_at = now_utc_iso()
    clock = time.perf_counter()
    log = progress or (lambda _message: None)

    input_dir = normalize_path(input_dir)
    output_dir = normalize_path(output_dir)

    user_config = load_config_file(config_path) if config_path else {}
    cfg = resolve_config(user_config, overrides)
    cfg_hash = config_hash(cfg)

    _validate_directories(input_dir, output_dir)
    os.makedirs(os.path.join(output_dir, "previews"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)
    existing_rows = _load_existing_manifest(output_dir, cfg, cfg_hash, log)
    write_resolved_config(os.path.join(output_dir, "config.resolved.json"), cfg, cfg_hash)
    existing_by_path = {row.get("source_path"): row for row in existing_rows if row.get("source_path")}

    files, scan_issues = scan_inputs(input_dir, bool(cfg["recursive"]), exclude_dirs=[output_dir])
    for issue in scan_issues:
        log(f"[scan] 无法读取：{issue['path']}：{issue['message']}")

    entries: list[dict[str, Any]] = []
    id_registry: dict[str, str] = {}
    for path in files:
        entries.append(_hash_entry(path, id_registry, cfg_hash))

    _assign_duplicates(entries)

    phash_by_id: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    reused = 0
    reprocessed = 0

    canonical_entries = [e for e in entries if e.get("duplicate_of") is None]
    duplicate_entries = [e for e in entries if e.get("duplicate_of") is not None]

    # 先处理有产物的主体，再补重复记录，最后按输入顺序重排清单
    for entry in canonical_entries:
        if entry.get("record") is not None:  # 哈希/ID 阶段就失败
            records.append(entry["record"])
            reprocessed += 1
            log(f"[error] {os.path.basename(entry['path'])}：{entry['record'].get('error_code')}")
            continue

        old = existing_by_path.get(entry["path"])
        if _can_reuse(old, entry, cfg_hash, output_dir):
            records.append(old)
            reused += 1
            if old.get("status") == Status.READY:
                phash = _read_phash(output_dir, old)
                if phash:
                    phash_by_id[old["image_id"]] = phash
            log(f"[reuse] {entry['image_id']} {os.path.basename(entry['path'])} ({old.get('status')})")
            continue

        outcome = process_image(
            source_path=entry["path"],
            image_id=entry["image_id"],
            source_sha256=entry["sha256"],
            source_size=entry["size"],
            cfg=cfg,
            cfg_hash=cfg_hash,
            output_dir=output_dir,
            duplicate_of=None,
        )
        records.append(outcome.record)
        reprocessed += 1
        if outcome.phash:
            phash_by_id[entry["image_id"]] = outcome.phash
        record = outcome.record
        detail = record.get("skip_reason") or record.get("error_code") or "ok"
        log(f"[{record['status']}] {entry['image_id']} {os.path.basename(entry['path'])} ({detail})")

    for entry in duplicate_entries:
        if entry.get("record") is not None:
            records.append(entry["record"])
            reprocessed += 1
            continue
        old = existing_by_path.get(entry["path"])
        if (
            old
            and old.get("status") == Status.DUPLICATE_EXACT
            and old.get("source_sha256") == entry["sha256"]
            and old.get("duplicate_of") == entry["duplicate_of"]
            and _versions_match(old, cfg_hash)
        ):
            records.append(old)
            reused += 1
            log(f"[reuse] {entry['image_id']} duplicate_exact of {entry['duplicate_of']}")
        else:
            outcome = process_image(
                source_path=entry["path"],
                image_id=entry["image_id"],
                source_sha256=entry["sha256"],
                source_size=entry["size"],
                cfg=cfg,
                cfg_hash=cfg_hash,
                output_dir=output_dir,
                duplicate_of=entry["duplicate_of"],
            )
            records.append(outcome.record)
            reprocessed += 1
            log(f"[duplicate_exact] {entry['image_id']} -> {entry['duplicate_of']}")

    records.sort(key=lambda rec: (str(rec.get("source_path", "")).lower(), str(rec.get("source_path", ""))))

    near_duplicates = (
        build_near_duplicates(phash_by_id, int(cfg["near_duplicate_hamming"]))
        if cfg["near_duplicate_enabled"]
        else {"threshold": int(cfg["near_duplicate_hamming"]), "groups": [], "review_required": True}
    )

    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    errors_path = os.path.join(output_dir, "errors.jsonl")
    near_path = os.path.join(output_dir, "near_duplicates.json")
    report_path = os.path.join(output_dir, "run_report.json")

    write_manifest(manifest_path, records)
    write_errors(errors_path, records)
    write_near_duplicates(near_path, near_duplicates)

    current_paths = {entry["path"] for entry in entries}
    stale = [row for row in existing_rows if row.get("source_path") not in current_paths]

    finished_at = now_utc_iso()
    report = build_run_report(
        started_at=started_at,
        finished_at=finished_at,
        duration=round(time.perf_counter() - clock, 3),
        input_dir=input_dir,
        output_dir=output_dir,
        records=records,
        near_duplicates=near_duplicates,
        cfg_hash=cfg_hash,
        cfg=cfg,
        reused=reused,
        reprocessed=reprocessed,
        stale_records=len(stale),
        scan_issues=len(scan_issues),
    )
    report["software"] = {
        "python": platform.python_version(),
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "littlecms": bool(pil_features.check("littlecms2")),
        "platform": platform.platform(),
    }
    write_run_report(report_path, report)

    return {
        "report": report,
        "manifest_path": manifest_path,
        "errors_path": errors_path,
        "near_duplicates_path": near_path,
        "report_path": report_path,
        "config_path": os.path.join(output_dir, "config.resolved.json"),
        "output_dir": output_dir,
    }


def _validate_directories(input_dir: str, output_dir: str) -> None:
    if not os.path.isdir(input_dir):
        raise FatalError(FatalCode.INPUT_DIR_INVALID, f"输入目录不存在或不是目录：{input_dir}")
    if os.path.normcase(input_dir) == os.path.normcase(output_dir):
        raise FatalError(FatalCode.OUTPUT_DIR_INVALID, "output_dir 不得等于 input_dir")


def _load_existing_manifest(output_dir: str, cfg: dict, cfg_hash: str, log: ProgressFn) -> list[dict]:
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        _ensure_output_not_foreign(output_dir)
        return []

    if cfg["overwrite"]:
        log("[overwrite] 忽略既有清单，全部重新处理")
        return []

    rows = read_jsonl(manifest_path)
    if not rows:
        return []

    incompatible = [row for row in rows if not _versions_match(row, cfg_hash)]
    if incompatible and not cfg["overwrite"]:
        raise FatalError(
            FatalCode.OUTPUT_STATE_INCOMPATIBLE,
            "输出目录已有旧结果，但 config_hash / preprocess_version / feature_schema_version 与本次不一致；"
            "请使用新的输出目录，或显式加 --overwrite 重新处理（旧特征不会与新特征混用）。",
        )
    if incompatible:
        log(f"[warn] 检测到 {len(incompatible)} 条不兼容旧记录，--overwrite 将全部重新处理")
        return []
    return rows


def _ensure_output_not_foreign(output_dir: str) -> None:
    try:
        entries = os.listdir(output_dir)
    except OSError as exc:
        raise FatalError(FatalCode.OUTPUT_DIR_INVALID, f"无法读取输出目录：{output_dir}：{exc}") from exc
    if not entries:
        return
    known_dirs = {"previews", "features"}
    foreign = [name for name in entries if name not in known_dirs]
    has_artifacts = False
    for name in known_dirs & set(entries):
        if os.listdir(os.path.join(output_dir, name)):
            has_artifacts = True
    if foreign or has_artifacts:
        raise FatalError(
            FatalCode.OUTPUT_DIR_INVALID,
            f"输出目录非空且没有 manifest.jsonl，拒绝写入以免覆盖无关文件：{output_dir}",
        )


def _versions_match(row: dict, cfg_hash: str) -> bool:
    return (
        row.get("config_hash") == cfg_hash
        and row.get("preprocess_version") == PREPROCESS_VERSION
        and row.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
    )


def _can_reuse(old: dict | None, entry: dict, cfg_hash: str, output_dir: str) -> bool:
    if not old or not _versions_match(old, cfg_hash):
        return False
    if old.get("source_sha256") != entry["sha256"]:
        return False
    status = old.get("status")
    if status == Status.READY:
        return validate_preview_and_feature(old, output_dir)
    if status == Status.SKIPPED:
        return bool(old.get("skip_reason"))
    return False


def _read_phash(output_dir: str, record: dict) -> str | None:
    import json

    feature_rel = record.get("feature_path")
    if not feature_rel:
        return None
    try:
        with open(os.path.join(output_dir, feature_rel), "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("similarity", {}).get("phash")
    except Exception:
        return None


def _hash_entry(path: str, id_registry: dict[str, str], cfg_hash: str) -> dict[str, Any]:
    try:
        size = os.path.getsize(path)
        digest = sha256_file(path)
    except OSError as exc:
        return {
            "path": path,
            "record": _file_error_record(path, cfg_hash, ErrorCode.READ_FAILED, Stage.HASH, str(exc)),
        }

    image_id = f"img_{digest[:16]}"
    existing_full = id_registry.get(image_id)
    if existing_full is not None and existing_full != digest:
        return {
            "path": path,
            "record": _file_error_record(
                path,
                cfg_hash,
                ErrorCode.ID_COLLISION,
                Stage.HASH,
                f"image_id 前缀冲突：{image_id} 已对应另一个完整哈希，拒绝覆盖",
            ),
        }
    id_registry[image_id] = digest

    return {
        "path": path,
        "sha256": digest,
        "size": size,
        "image_id": image_id,
        "duplicate_of": None,
        "record": None,
    }


def _file_error_record(path: str, cfg_hash: str, code: str, stage: str, message: str) -> dict[str, Any]:
    return {
        "image_id": None,
        "source_path": path,
        "source_sha256": None,
        "source_size_bytes": None,
        "preprocess_version": PREPROCESS_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config_hash": cfg_hash,
        "processed_at": now_utc_iso(),
        "status": Status.ERROR,
        "duplicate_of": None,
        "skip_reason": None,
        "error_code": code,
        "error_stage": stage,
        "error_message": message,
        "preview_path": None,
        "feature_path": None,
        "preview_sha256": None,
        "warnings": [],
    }


def _assign_duplicates(entries: list[dict]) -> None:
    by_sha: dict[str, dict] = {}
    for entry in entries:
        if entry.get("record") is not None or not entry.get("sha256"):
            continue
        first = by_sha.get(entry["sha256"])
        if first is None:
            by_sha[entry["sha256"]] = entry
        else:
            entry["duplicate_of"] = first["image_id"]

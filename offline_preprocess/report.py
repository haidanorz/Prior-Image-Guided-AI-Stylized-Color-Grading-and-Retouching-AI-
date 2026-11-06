"""清单、错误、近重复与运行报告的生成。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .errors import Status
from .similarity import PHASH_ALGORITHM, hamming_distance
from .util import atomic_write_bytes, atomic_write_json, dump_json_text, now_utc_iso
from .version import FEATURE_SCHEMA_VERSION, PREPROCESS_VERSION

MANIFEST_KEYS = (
    "image_id",
    "status",
    "source_path",
    "source_sha256",
    "source_size_bytes",
    "preview_path",
    "feature_path",
    "preview_sha256",
    "duplicate_of",
    "skip_reason",
    "error_code",
    "warnings",
    "preprocess_version",
    "feature_schema_version",
    "config_hash",
    "processed_at",
)


def write_manifest(path: str, records: list[dict]) -> None:
    lines = []
    for record in records:
        row = {key: record.get(key) for key in MANIFEST_KEYS}
        row["warnings"] = list(record.get("warnings") or [])
        lines.append(dump_json_text(row, pretty=False).rstrip("\n"))
    atomic_write_bytes(path, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))


def write_errors(path: str, records: list[dict]) -> None:
    lines = []
    for record in records:
        if record.get("status") != Status.ERROR:
            continue
        row = {
            "image_id": record.get("image_id"),
            "source_path": record.get("source_path"),
            "stage": record.get("error_stage"),
            "code": record.get("error_code"),
            "message": record.get("error_message"),
            "timestamp": record.get("processed_at"),
        }
        lines.append(dump_json_text(row, pretty=False).rstrip("\n"))
    atomic_write_bytes(path, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))


def write_near_duplicates(path: str, payload: dict) -> None:
    atomic_write_json(path, payload)


def write_run_report(path: str, payload: dict) -> None:
    atomic_write_json(path, payload)


def build_near_duplicates(phash_by_id: dict[str, str], threshold: int) -> dict:
    """按 Hamming 距离 <= threshold 做连通分组；只提示复核，不做任何处置。"""
    ids = sorted(phash_by_id)
    parent = {image_id: image_id for image_id in ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pairs: list[dict] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            distance = hamming_distance(phash_by_id[a], phash_by_id[b])
            if distance <= threshold:
                union(a, b)
                pairs.append({"a": a, "b": b, "hamming": distance})

    grouped: dict[str, list[str]] = {}
    for image_id in ids:
        grouped.setdefault(find(image_id), []).append(image_id)

    groups = []
    index = 0
    for members in sorted(grouped.values(), key=lambda m: m[0]):
        if len(members) < 2:
            continue
        index += 1
        member_set = set(members)
        groups.append(
            {
                "group_id": f"nd_{index:04d}",
                "members": [{"image_id": m, "phash": phash_by_id[m]} for m in members],
                "pairs": [p for p in pairs if p["a"] in member_set and p["b"] in member_set],
            }
        )

    return {
        "generated_at": now_utc_iso(),
        "algorithm": PHASH_ALGORITHM,
        "threshold": int(threshold),
        "review_required": True,
        "note": "近重复仅为人工复核线索，不代表内容相同，不自动删除、不自动合并、不据此淘汰照片。",
        "groups": groups,
    }


def build_run_report(
    *,
    started_at: str,
    finished_at: str,
    duration: float,
    input_dir: str,
    output_dir: str,
    records: list[dict],
    near_duplicates: dict,
    cfg_hash: str,
    cfg: dict,
    reused: int,
    reprocessed: int,
    stale_records: int,
    scan_issues: int,
) -> dict:
    status_counts = {status: 0 for status in Status.ALL}
    warning_counter: Counter[str] = Counter()
    error_code_counter: Counter[str] = Counter()
    stage_counter: Counter[str] = Counter()

    for record in records:
        status = record.get("status")
        if status in status_counts:
            status_counts[status] += 1
        for warning in record.get("warnings") or []:
            warning_counter[warning] += 1
        if status == Status.ERROR:
            error_code_counter[str(record.get("error_code"))] += 1
            stage_counter[str(record.get("error_stage"))] += 1

    total = len(records)
    closed = sum(status_counts.values())
    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "input_file_count": total,
        "manifest_record_count": total,
        "status_counts": status_counts,
        "counts_consistent": closed == total,
        "warning_count": int(sum(warning_counter.values())),
        "warning_counts": dict(sorted(warning_counter.items())),
        "error_code_counts": dict(sorted(error_code_counter.items())),
        "error_stage_counts": dict(sorted(stage_counter.items())),
        "near_duplicate_group_count": len(near_duplicates.get("groups", [])),
        "near_duplicate_threshold": int(near_duplicates.get("threshold", cfg["near_duplicate_hamming"])),
        "config_hash": cfg_hash,
        "preprocess_version": PREPROCESS_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "resume": {
            "reused": reused,
            "reprocessed": reprocessed,
            "stale_manifest_records": stale_records,
            "scan_issue_count": scan_issues,
        },
        "resolved_config": cfg,
    }

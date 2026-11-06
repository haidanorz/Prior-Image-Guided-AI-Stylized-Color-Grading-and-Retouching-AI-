"""离线编码器的最终产物：特征指纹包。

包内只放“编码结果与证据”，不放 XMP、原图、单图分析原文：
- fingerprint.json：运行时消费的紧凑指纹（global + bucket delta）
- global_signature.json / profile.json / buckets/*.json：证据与审计
- package_manifest.json：逐文件 SHA-256 与 package_hash
- README.md：边界与校验方法
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .util import atomic_write_bytes, now_utc_iso

PACKAGE_SCHEMA_VERSION = "1.1.0"
FINGERPRINT_SCHEMA_VERSION = "1.1.0"

_REQUIRED_ROOT_FILES = ("global_signature.json", "profile.json")
_REQUIRED_BUCKET_FILES = ("profile.json", "delta.json", "stats.json")
_OPTIONAL_BUCKET_FILES = ("narrative.json", "descriptions.jsonl")


def export_fingerprint_package(build_root: str, out_dir: str, *, overwrite: bool = False) -> dict:
    build_root = os.path.abspath(build_root)
    out_dir = os.path.abspath(out_dir)
    manifest_path = os.path.join(out_dir, "package_manifest.json")
    if os.path.exists(manifest_path) and not overwrite:
        raise FileExistsError(f"包已存在，使用 overwrite=True 重新生成：{manifest_path}")
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not os.path.isfile(manifest_path) and not overwrite:
        raise FileExistsError(f"输出目录非空且不是指纹包：{out_dir}")

    profile = _read_json(os.path.join(build_root, "profile.json"))
    signature = _read_json(os.path.join(build_root, "global_signature.json"))
    fingerprint = _build_fingerprint(profile, signature, build_root)
    fingerprint_bytes = _json_bytes(fingerprint)

    os.makedirs(out_dir, exist_ok=True)
    files: list[dict] = []

    def add(rel: str, data: bytes) -> None:
        atomic_write_bytes(os.path.join(out_dir, rel), data)
        files.append(
            {"path": rel.replace(os.sep, "/"), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        )

    add("fingerprint.json", fingerprint_bytes)
    for name in _REQUIRED_ROOT_FILES:
        path = os.path.join(build_root, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"缺少必需文件：{path}")
        add(name, _read_bytes(path))

    index_path = os.path.join(build_root, "buckets", "index.json")
    if os.path.isfile(index_path):
        add("buckets/index.json", _read_bytes(index_path))

    for bucket_id in sorted(fingerprint["buckets"]):
        for name in _REQUIRED_BUCKET_FILES + _OPTIONAL_BUCKET_FILES:
            rel = f"buckets/{bucket_id}/{name}"
            path = os.path.join(build_root, rel)
            if os.path.isfile(path):
                add(rel, _read_bytes(path))

    readme = _readme(fingerprint)
    add("README.md", readme.encode("utf-8"))

    files.sort(key=lambda item: item["path"])
    package_hash = _canonical_hash(
        {
            "package_schema_version": PACKAGE_SCHEMA_VERSION,
            "fingerprint_hash": fingerprint["fingerprint_hash"],
            "files": files,
        }
    )
    manifest = {
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "package_id": f"fpkg_{package_hash[:16]}",
        "package_hash": package_hash,
        "fingerprint_id": fingerprint["fingerprint_id"],
        "fingerprint_hash": fingerprint["fingerprint_hash"],
        "created_at": now_utc_iso(),
        "status": "draft",
        "application_ready": False,
        "validation_scope": profile.get("validation_scope"),
        "source_build": {
            "build_id": profile.get("build_id"),
            "profile_id": profile.get("profile_id"),
            "input_fingerprint": profile.get("input_fingerprint"),
            "config_hash": profile.get("config_hash"),
            "response_set_hash": profile.get("response_set_hash"),
            "aggregate_hash": profile.get("aggregate_hash"),
        },
        "counts": {
            "files": len(files),
            "buckets": len(fingerprint["buckets"]),
            "global_parameters": len(fingerprint["global"]),
            "bucket_parameters": sum(len(item.get("parameters") or {}) for item in fingerprint["buckets"].values()),
            "insufficient_parameters": len(fingerprint["insufficient_evidence"]),
        },
        "files": files,
        "notes": [
            "这是离线编码器的特征指纹包，不包含 XMP、原图或单图分析原文。",
            "status=draft / application_ready=false：桶参数仍为 draft，未通过 §13 发布门槛。",
            "shape=bucket_primary：桶参数是主路径，global 仅作未命中桶的兜底。",
            "在线解码器只应读取 fingerprint.json；其余文件用于证据与审计。",
        ],
    }
    manifest_bytes = _json_bytes(manifest)
    atomic_write_bytes(manifest_path, manifest_bytes)

    return manifest


def _build_fingerprint(profile: dict, signature: dict, build_root: str) -> dict:
    global_params = {}
    for key, item in signature.get("parameters", {}).items():
        global_params[key] = {
            "central": item.get("central"),
            "unit": item.get("unit"),
            "kind": item.get("kind"),
            "cross_scene_invariant": item.get("cross_scene_invariant"),
            "evidence_buckets": item.get("evidence_buckets", []),
            "sign_agreement": item.get("sign_agreement"),
            "weighted_consistency": item.get("weighted_consistency"),
        }

    from .contract import load_parameters

    registry_keys = [item["key"] for item in load_parameters()["edit_parameters"]]
    shape = profile.get("profile_shape", "bucket_primary")
    bucket_parameters = profile.get("bucket_parameters") or {}

    buckets = {}
    for bucket_id, bucket in sorted(profile.get("buckets", {}).items()):
        raw_params = bucket_parameters.get(bucket_id, {})
        published = {
            key: {
                "value": item.get("value"),
                "unit": item.get("unit"),
                "kind": item.get("kind"),
                "p25": item.get("p25"),
                "p75": item.get("p75"),
                "span": item.get("span"),
                "n": item.get("n"),
                "consistency": item.get("consistency"),
                "status": item.get("status"),
                "apply_policy": item.get("apply_policy"),
            }
            for key, item in raw_params.items()
            if item.get("status") == "published"
        }
        priors = {
            key: {
                "value": item.get("value"),
                "unit": item.get("unit"),
                "span": item.get("span"),
                "p25": item.get("p25"),
                "p75": item.get("p75"),
                "n": item.get("n"),
                "consistency": item.get("consistency"),
                "cv_vs_pooled": item.get("cv_vs_pooled"),
                "status": item.get("status"),
                "apply_policy": "prior_only",
            }
            for key, item in raw_params.items()
            if item.get("status") == "wide_dispersion"
        }
        not_evaluable = [
            key
            for key in registry_keys
            if key not in published and key not in priors and key not in global_params
        ]
        buckets[bucket_id] = {
            "bucket_status": bucket.get("bucket_status"),
            "n_core": bucket.get("n_core"),
            "n_train": bucket.get("n_train"),
            "n_holdout": bucket.get("n_holdout"),
            "parameters": published,
            "priors": priors,
            "withheld_keys": [
                {"key": key, "reason": item.get("reason")}
                for key, item in raw_params.items()
                if item.get("status") == "withheld"
            ],
            "not_evaluable": not_evaluable,
            "delta": {
                key: item.get("delta")
                for key, item in bucket.get("deltas", {}).items()
            },
            "delta_units": {
                key: item.get("unit")
                for key, item in bucket.get("deltas", {}).items()
            },
            "unresolved_keys": [item.get("key") for item in bucket.get("unresolved", [])],
        }
        bucket_profile_path = os.path.join(build_root, "buckets", bucket_id, "profile.json")
        if os.path.isfile(bucket_profile_path):
            with open(bucket_profile_path, encoding="utf-8") as handle:
                bucket_profile = json.load(handle)
            buckets[bucket_id]["profile_hash"] = bucket_profile.get("profile_hash")
            if bucket_profile.get("narrative_ref"):
                buckets[bucket_id]["narrative"] = bucket_profile["narrative_ref"]

    payload = {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "status": "draft",
        "application_ready": False,
        "validation_scope": profile.get("validation_scope"),
        "source": {
            "build_id": profile.get("build_id"),
            "profile_id": profile.get("profile_id"),
            "aggregate_hash": profile.get("aggregate_hash"),
            "taxonomy_version": signature.get("taxonomy_version"),
            "registry_version": signature.get("registry_version"),
            "input_fingerprint": profile.get("input_fingerprint"),
            "config_hash": profile.get("config_hash"),
            "response_set_hash": profile.get("response_set_hash"),
        },
        "shape": shape,
        "global_role": "fallback" if shape == "bucket_primary" else "baseline",
        "runtime_rule": (
            "scene_primary -> bucket parameters (status=published); "
            "priors -> advisory only (wide dispersion), never a hard anchor; "
            "key missing in both bucket and global -> not_evaluable, skip the slider; "
            "unknown/exception scene -> global only; never blend two buckets"
            if shape == "bucket_primary"
            else "scene_primary -> global + bucket delta; unknown/exception -> global only; never add a full bucket again"
        ),
        "global": global_params,
        "buckets": buckets,
        "insufficient_evidence": [
            {"key": item.get("key"), "reason": item.get("reason")}
            for item in signature.get("insufficient_evidence", [])
        ],
        "scene_dependent_parameters": signature.get("scene_dependent_parameters", []),
        "invariant_parameters": signature.get("invariant_parameters", []),
        "constraints": signature.get("constraints", []),
        "local_adjustment_policy": signature.get("local_adjustment_policy"),
        "warning": (
            "draft: bucket parameters are candidates, not validated for application"
            if shape == "bucket_primary"
            else "draft: bucket deltas are candidates, not validated for application"
        ),
    }
    fingerprint_hash = _canonical_hash(payload)
    payload["fingerprint_hash"] = fingerprint_hash
    payload["fingerprint_id"] = f"fp_{fingerprint_hash[:16]}"
    return payload


def _readme(fingerprint: dict) -> str:
    shape = fingerprint.get("shape", "global_plus_delta")
    if shape == "bucket_primary":
        rule = (
            "Runtime rule: classify the target scene first, then use the bucket parameters;\n"
            "`global` is a fallback for keys the bucket does not carry, or for unknown scenes.\n"
            "Never blend two buckets."
        )
    else:
        rule = (
            "Runtime rule: classify the target scene, then use `global + bucket delta`;\n"
            "if no bucket matches, use `global` only. Never add a full bucket profile on top."
        )
    return f"""# Feature fingerprint package

This directory is the offline encoder output. It contains no XMP, no source images,
and no per-image analysis text.

- `fingerprint.json`: runtime fingerprint (bucket parameters; global as fallback)
- `global_signature.json`, `profile.json`, `buckets/*.json`: evidence for audit
- `package_manifest.json`: per-file SHA-256 and package hash

Status: draft / application_ready=false
Fingerprint schema: {FINGERPRINT_SCHEMA_VERSION}
Fingerprint id: {fingerprint['fingerprint_id']}
Profile shape: {shape}

{rule}

Integrity:
1. Recompute SHA-256 for each file in `package_manifest.json`.
2. Recompute package_hash from `files` + `fingerprint_hash`.
3. Recompute `fingerprint_hash` from `fingerprint.json` with the hash field removed.
"""


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8")


def _canonical_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()



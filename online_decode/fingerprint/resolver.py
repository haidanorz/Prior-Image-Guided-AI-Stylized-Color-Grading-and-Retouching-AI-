"""指纹包校验与桶判定。

两条不可让步：
  1. 包哈希不过就不进 LR 阶段 —— 指纹是风格判断的唯一依据，被改过就失去意义。
  2. 桶判定不交给模型：模型只提供带分数的候选，阈值判定复用离线同一函数
     （`offline_profile.bucketing.assign_scene`），否则同一份档案会被解释成两套流派。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import Codes, DecodeError
from ..util import canonical_json, sha256_bytes, sha256_file

PACKAGE_SCHEMA = "1.0.0"


@dataclass
class PackageInfo:
    path: Path
    manifest: dict[str, Any]
    fingerprint: dict[str, Any]
    fingerprint_id: str
    fingerprint_hash: str
    status: str
    application_ready: bool
    schema_version: str
    bucket_ids: list[str] = field(default_factory=list)
    file_hashes: dict[str, str] = field(default_factory=dict)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise DecodeError(Codes.PACKAGE_INVALID, f"{path.name} 不是合法 JSON：{exc}", stage="preflight")


def verify_package(path: str | Path) -> PackageInfo:
    root = Path(path)
    if not root.is_dir():
        raise DecodeError(Codes.PACKAGE_MISSING, f"指纹包目录不存在：{root}", stage="preflight")

    manifest = _read_json(root / "package_manifest.json")
    fingerprint = _read_json(root / "fingerprint.json")
    if not isinstance(manifest, dict) or not isinstance(fingerprint, dict):
        raise DecodeError(
            Codes.PACKAGE_MISSING,
            "指纹包缺 package_manifest.json 或 fingerprint.json",
            stage="preflight",
        )

    # 1) 逐文件哈希
    file_hashes: dict[str, str] = {}
    for item in manifest.get("files") or []:
        rel = item.get("path")
        want = item.get("sha256")
        if not rel or not want:
            raise DecodeError(Codes.PACKAGE_INVALID, "manifest.files 条目缺 path/sha256", stage="preflight")
        target = root / rel
        if not target.is_file():
            raise DecodeError(Codes.PACKAGE_HASH_MISMATCH, f"包内文件缺失：{rel}", stage="preflight")
        got = sha256_file(target)
        if got != want:
            raise DecodeError(
                Codes.PACKAGE_HASH_MISMATCH,
                f"{rel} 哈希不符（manifest {want[:12]} / 实际 {got[:12]}）",
                stage="preflight",
            )
        file_hashes[rel] = got

    # 2) 指纹自哈希（去掉 fingerprint_hash / fingerprint_id 后规范化）
    payload = {k: v for k, v in fingerprint.items() if k not in ("fingerprint_hash", "fingerprint_id")}
    recomputed = sha256_bytes(canonical_json(payload).encode("utf-8"))
    if fingerprint.get("fingerprint_hash") and fingerprint["fingerprint_hash"] != recomputed:
        raise DecodeError(
            Codes.PACKAGE_HASH_MISMATCH,
            "fingerprint.json 自身哈希不符，文件可能被改动",
            stage="preflight",
            detail={"want": fingerprint.get("fingerprint_hash"), "got": recomputed},
        )

    # 3) 包哈希
    files = sorted(manifest.get("files") or [], key=lambda item: item.get("path", ""))
    package_payload = {
        "package_schema_version": manifest.get("package_schema_version", PACKAGE_SCHEMA),
        "fingerprint_hash": fingerprint.get("fingerprint_hash"),
        "files": files,
    }
    package_hash = sha256_bytes(canonical_json(package_payload).encode("utf-8"))
    if manifest.get("package_hash") and manifest["package_hash"] != package_hash:
        raise DecodeError(
            Codes.PACKAGE_HASH_MISMATCH,
            "package_hash 复算不符（manifest 与文件清单不一致）",
            stage="preflight",
            detail={"want": manifest.get("package_hash"), "got": package_hash},
        )

    return PackageInfo(
        path=root,
        manifest=manifest,
        fingerprint=fingerprint,
        fingerprint_id=fingerprint.get("fingerprint_id") or "",
        fingerprint_hash=fingerprint.get("fingerprint_hash") or "",
        status=fingerprint.get("status") or "unknown",
        application_ready=bool(fingerprint.get("application_ready")),
        schema_version=fingerprint.get("fingerprint_schema_version") or "",
        bucket_ids=sorted((fingerprint.get("buckets") or {}).keys()),
        file_hashes=file_hashes,
    )


def load_narrative(pkg: PackageInfo, bucket_id: str) -> dict[str, Any] | None:
    meta = ((pkg.fingerprint.get("buckets") or {}).get(bucket_id) or {}).get("narrative") or {}
    rel = meta.get("file")
    if not rel:
        return None
    path = pkg.path / rel
    data = _read_json(path)
    if not isinstance(data, dict):
        raise DecodeError(Codes.PACKAGE_INVALID, f"桶叙事缺失或不可解析：{rel}", stage="preflight")

    declared = meta.get("narrative_hash") or data.get("narrative_hash")
    if declared:
        payload = {k: v for k, v in data.items() if k != "narrative_hash"}
        got = sha256_bytes(canonical_json(payload).encode("utf-8"))
        if got != declared:
            raise DecodeError(
                Codes.PACKAGE_HASH_MISMATCH,
                f"{rel} 的叙事哈希不符（声明 {declared[:12]} / 复算 {got[:12]}）",
                stage="preflight",
            )
    data["_file"] = rel
    return data


def decide_bucket(
    pkg: PackageInfo,
    observation: dict[str, Any],
) -> dict[str, Any]:
    """模型给候选分，这里用离线同一套阈值判定 core/associate/unassigned。"""
    from offline_profile.bucketing import assign_scene
    from offline_profile.config import DEFAULTS

    candidates = (observation or {}).get("primary_candidates") or []
    if not isinstance(candidates, list) or not candidates:
        raise DecodeError(
            Codes.MODEL_RESPONSE_INVALID,
            "scene_observation.primary_candidates 为空，无法判定场景桶",
            stage="brief",
        )

    analysis = {
        "image_id": (observation or {}).get("image_id") or "target",
        "scene_observation": {
            "primary_candidates": candidates,
            "lighting": (observation or {}).get("lighting") or "unknown",
            "brightness": (observation or {}).get("brightness") or "unknown",
            "backlit": bool((observation or {}).get("backlit", False)),
        },
    }
    assignment = assign_scene(analysis, DEFAULTS)

    bucket_id = assignment.get("bucket_id")
    known = bucket_id in (pkg.fingerprint.get("buckets") or {})
    reliable = assignment.get("level") == "core" and known

    decision = {
        "level": assignment.get("level"),
        "candidate_bucket": assignment.get("candidate_bucket"),
        "bucket_id": bucket_id if reliable else None,
        "top_score": assignment.get("top_score"),
        "margin": assignment.get("margin"),
        "reason": assignment.get("reason"),
        "thresholds": {
            "core_top_score": DEFAULTS["core_top_score"],
            "core_margin": DEFAULTS["core_margin"],
            "associate_top_score": DEFAULTS["associate_top_score"],
        },
        "candidates": assignment.get("candidates"),
        "fallback": None if reliable else "global",
        "fallback_reason": None if reliable else _fallback_reason(assignment, known),
        "taxonomy_version": assignment.get("taxonomy_version"),
    }
    return decision


def _fallback_reason(assignment: dict[str, Any], known: bool) -> str:
    if assignment.get("level") == "core" and not known:
        return "bucket_not_in_package"
    if assignment.get("level") == "associate":
        # 沿用离线判定的具体原因（分数不足 / 领先幅度不足），别把它压成一个模糊词
        return assignment.get("reason") or "associate_only"
    if assignment.get("candidate_bucket") == "other":
        return "other_label"
    return assignment.get("reason") or "not_reliable"


def references(pkg: PackageInfo, bucket_id: str | None) -> dict[str, dict[str, Any]]:
    """整理参数参考。

    shape=bucket_primary：桶参数是主路径，global 只在桶缺项时兜底；
    shape=global_plus_delta（旧）：global + 桶 delta。
    """
    from ..params.fields import LOGICAL_TO_SDK

    out: dict[str, dict[str, Any]] = {}
    shape = pkg.fingerprint.get("shape", "global_plus_delta")
    bucket = ((pkg.fingerprint.get("buckets") or {}).get(bucket_id) or {}) if bucket_id else {}

    if shape == "bucket_primary" and bucket_id:
        for logical, spec in (bucket.get("parameters") or {}).items():
            sdk = LOGICAL_TO_SDK.get(logical)
            if not sdk or spec.get("value") is None:
                continue
            out[sdk] = {
                "logical": logical,
                "value": float(spec["value"]),
                "evidence_level": "runtime_reference",
                "ref": f"fingerprint.buckets.{bucket_id}.parameters.{logical}",
                "unit": spec.get("unit"),
                "consistency": spec.get("consistency"),
                "span": spec.get("span"),
                "role": "bucket",
                "status": spec.get("status"),
            }

    for logical, spec in (pkg.fingerprint.get("global") or {}).items():
        sdk = LOGICAL_TO_SDK.get(logical)
        if not sdk or spec.get("central") is None:
            continue
        if sdk in out:
            continue
        out[sdk] = {
            "logical": logical,
            "value": float(spec["central"]),
            "evidence_level": "runtime_reference",
            "ref": f"fingerprint.global.{logical}",
            "unit": spec.get("unit"),
            "consistency": spec.get("weighted_consistency"),
            "span": (
                round(float(spec["p75"]) - float(spec["p25"]), 4)
                if spec.get("p25") is not None and spec.get("p75") is not None
                else None
            ),
            "role": "global_fallback" if shape == "bucket_primary" else "global",
        }

    if shape != "bucket_primary" and bucket_id:
        for logical, value in (bucket.get("delta") or {}).items():
            sdk = LOGICAL_TO_SDK.get(logical)
            if not sdk or value is None:
                continue
            out[sdk] = {
                "logical": logical,
                "value": float(value),
                "evidence_level": "runtime_reference",
                "ref": f"fingerprint.buckets.{bucket_id}.delta.{logical}",
                "unit": (bucket.get("delta_units") or {}).get(logical),
                "role": "bucket_delta",
            }

    not_evaluable = set(bucket.get("not_evaluable") or [])
    for item in pkg.fingerprint.get("insufficient_evidence") or []:
        logical = item.get("key")
        sdk = LOGICAL_TO_SDK.get(logical)
        if not sdk or sdk in out:
            continue
        out[sdk] = {
            "logical": logical,
            "value": None,
            "evidence_level": "experimental_prior",
            "ref": f"fingerprint.insufficient_evidence.{logical}",
            "note": item.get("reason"),
            "not_evaluable": logical in not_evaluable,
        }
    return out


def priors(pkg: PackageInfo, bucket_id: str | None) -> dict[str, dict[str, Any]]:
    """宽离散参数的先验：只给模型作参考，不作锚点、不参与带宽检查。"""
    if not bucket_id:
        return {}
    bucket = (pkg.fingerprint.get("buckets") or {}).get(bucket_id) or {}
    out: dict[str, dict[str, Any]] = {}
    for logical, spec in (bucket.get("priors") or {}).items():
        if spec.get("value") is None:
            continue
        out[logical] = {
            "value": spec.get("value"),
            "unit": spec.get("unit"),
            "span": spec.get("span"),
            "p25": spec.get("p25"),
            "p75": spec.get("p75"),
            "n": spec.get("n"),
            "consistency": spec.get("consistency"),
            "note": "wide_dispersion: advisory only, do not treat as a fixed target",
        }
    return out

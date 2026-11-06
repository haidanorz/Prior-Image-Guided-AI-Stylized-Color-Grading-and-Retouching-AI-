"""Pilot XMP 导出：把 global + bucket delta 映射成 Lightroom 预设。

这不是正式应用层：
- 不修改 profile.json；
- 不把 application_ready 置为 true；
- 不稳定但可看的桶参数（contrast/clarity）作为 pilot_only 单独标注；
- 未映射的参数（色温/色调/分离色调）明确跳过。
"""

from __future__ import annotations

import hashlib
import json
import os
from xml.sax.saxutils import escape

from .util import atomic_write_bytes, atomic_write_json, now_utc_iso

MAPPING_VERSION = "xmp-pilot-v1"
XMP_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"

# key: (Lightroom setting, minimum, maximum)
NUMERIC_MAP = {
    "exposure_ev": ("Exposure2012", -5.0, 5.0),
    "contrast": ("Contrast2012", -100.0, 100.0),
    "highlights": ("Highlights2012", -100.0, 100.0),
    "shadows": ("Shadows2012", -100.0, 100.0),
    "whites": ("Whites2012", -100.0, 100.0),
    "blacks": ("Blacks2012", -100.0, 100.0),
    "vibrance": ("Vibrance", -100.0, 100.0),
    "saturation": ("Saturation", -100.0, 100.0),
    "clarity": ("Clarity2012", -100.0, 100.0),
    "dehaze": ("Dehaze", -100.0, 100.0),
    "sharpening_amount": ("Sharpness", 0.0, 150.0),
    "grain_amount": ("GrainAmount", 0.0, 100.0),
    "vignette_amount": ("PostCropVignetteAmount", -100.0, 100.0),
}

SKIPPED = [
    {"key": "temperature_shift", "reason": "relative index, not Kelvin; requires WB calibration"},
    {"key": "tint_shift", "reason": "relative index, not a calibrated tint value"},
    {"key": "shadow_tint_hue", "reason": "hue only; no defensible saturation/strength mapping"},
    {"key": "highlight_tint_hue", "reason": "hue only; no defensible saturation/strength mapping"},
]


def export_pilot_xmp(build_root: str, out_dir: str, *, include_unstable: bool = True) -> dict:
    build_root = os.path.abspath(build_root)
    out_dir = os.path.abspath(out_dir)
    profile_path = os.path.join(build_root, "profile.json")
    signature_path = os.path.join(build_root, "global_signature.json")
    with open(profile_path, encoding="utf-8") as handle:
        profile = json.load(handle)
    with open(signature_path, encoding="utf-8") as handle:
        signature = json.load(handle)

    global_params = signature.get("parameters", {})
    buckets = profile.get("buckets", {})
    shape = profile.get("profile_shape", "global_plus_delta")
    presets = []

    presets.append(
        _make_preset(
            out_dir,
            scope="global",
            label="Offline Pilot - Global",
            values=_base_values(global_params),
            candidate_params=[],
            pilot_only_params=[],
            source_ref={
                "global_signature_file": "global_signature.json",
                "global_signature_hash": profile.get("global_signature", {}).get("file_hash"),
                "profile_aggregate_hash": profile.get("aggregate_hash"),
            },
        )
    )

    for bucket_id in sorted(buckets):
        bucket = buckets[bucket_id]
        if bucket.get("status") != "draft":
            continue
        values = _base_values(global_params)
        candidate = []
        pilot_only = []
        if shape == "bucket_primary":
            params = (profile.get("bucket_parameters") or {}).get(bucket_id, {})
            for key, item in params.items():
                if key not in NUMERIC_MAP or item.get("value") is None:
                    continue
                if item.get("status") == "published":
                    values[key] = float(item["value"])
                    candidate.append(key)
                elif item.get("status") == "wide_dispersion" and include_unstable:
                    values[key] = float(item["value"])
                    pilot_only.append(key)
        else:
            for key, item in bucket.get("deltas", {}).items():
                if key not in NUMERIC_MAP:
                    continue
                values[key] = values.get(key, 0.0) + float(item["delta"])
                candidate.append(key)
        if shape != "bucket_primary" and include_unstable:
            stats_path = os.path.join(build_root, "buckets", bucket_id, "stats.json")
            if os.path.isfile(stats_path):
                with open(stats_path, encoding="utf-8") as handle:
                    stats = json.load(handle)
                for key in ("contrast", "clarity"):
                    if key in values or key in candidate:
                        continue
                    param = stats.get("params", {}).get(key)
                    if param and param.get("central") is not None:
                        values[key] = float(param["central"])
                        pilot_only.append(key)
        bucket_profile_path = os.path.join(build_root, "buckets", bucket_id, "profile.json")
        bucket_ref = None
        if os.path.isfile(bucket_profile_path):
            with open(bucket_profile_path, encoding="utf-8") as handle:
                bucket_profile = json.load(handle)
            bucket_ref = {
                "bucket_profile_file": f"buckets/{bucket_id}/profile.json",
                "bucket_profile_hash": bucket_profile.get("profile_hash"),
            }
        presets.append(
            _make_preset(
                out_dir,
                scope=bucket_id,
                label=f"Offline Pilot - {bucket_id}",
                values=values,
                candidate_params=candidate,
                pilot_only_params=pilot_only,
                source_ref=bucket_ref,
            )
        )

    manifest = {
        "status": "pilot_only",
        "application_ready": False,
        "validation_scope": profile.get("validation_scope"),
        "build_id": profile.get("build_id"),
        "profile_status": profile.get("status"),
        "source_profile": "profile.json",
        "source_profile_sha256": _sha256_file(profile_path),
        "source_global_signature_sha256": _sha256_file(signature_path),
        "mapping_version": MAPPING_VERSION,
        "generated_at": now_utc_iso(),
        "presets": presets,
        "skipped": SKIPPED,
        "warnings": [
            "pilot_only：未通过 §13 留出/Bootstrap 发布门槛，不能作为正式应用配置。",
            "contrast/clarity 在桶预设中来自不稳定或单桶证据，仅用于试看。",
            "色温/色调/分离色调未映射，应用后白平衡仍为 As Shot。",
            "请只用于 2-3 张测试图回看，不要批量套用或分发。",
        ],
    }
    os.makedirs(out_dir, exist_ok=True)
    atomic_write_json(os.path.join(out_dir, "xmp_manifest.json"), manifest)
    _write_readme(out_dir, manifest)
    return manifest


def _base_values(global_params: dict) -> dict:
    values = {}
    for key, spec in global_params.items():
        if key in NUMERIC_MAP and spec.get("central") is not None:
            values[key] = float(spec["central"])
    return values


def _make_preset(
    out_dir: str,
    *,
    scope: str,
    label: str,
    values: dict,
    candidate_params: list[str],
    pilot_only_params: list[str],
    source_ref: dict | None = None,
) -> dict:
    clamped = {}
    clamp_notes = []
    for key, setting in NUMERIC_MAP.items():
        setting_name, low, high = setting
        value = float(values.get(key, 0.0))
        if value < low:
            clamp_notes.append(f"{key}:{value}->{low}")
            value = low
        if value > high:
            clamp_notes.append(f"{key}:{value}->{high}")
            value = high
        clamped[key] = value

    attrs = {
        "crs:PresetType": "Normal",
        "crs:Version": "15.0",
        "crs:ProcessVersion": "15.4",
        "crs:WhiteBalance": "As Shot",
        "crs:HasSettings": "True",
    }
    for key, value in clamped.items():
        attrs[f"crs:{NUMERIC_MAP[key][0]}"] = _fmt(value)

    filename = f"pilot_{scope}.xmp"
    xml = _render_xmp(label, attrs)
    path = os.path.join(out_dir, filename)
    atomic_write_bytes(path, xml.encode("utf-8"))
    return {
        "file": filename,
        "scope": scope,
        "candidate_params": sorted(candidate_params),
        "pilot_only_params": sorted(pilot_only_params),
        "values": {NUMERIC_MAP[key][0]: round(value, 2) for key, value in clamped.items()},
        "clamped": clamp_notes,
        "source_ref": source_ref or {},
        "sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
    }


def _render_xmp(label: str, attrs: dict) -> str:
    lines = [
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>',
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="prior-color-profile pilot">',
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        '  <rdf:Description rdf:about=""',
        f'    xmlns:crs="{XMP_NS}"',
    ]
    for key, value in attrs.items():
        lines.append(f'    {key}="{escape(str(value), {chr(34): "&quot;"})}"')
    lines.extend(
        [
            "  >",
            "   <crs:Name>",
            "    <rdf:Alt>",
            f'     <rdf:li xml:lang="x-default">{escape(label)}</rdf:li>',
            "    </rdf:Alt>",
            "   </crs:Name>",
            "   <crs:Group>",
            "    <rdf:Alt>",
            '     <rdf:li xml:lang="x-default">Offline Profile Pilot</rdf:li>',
            "    </rdf:Alt>",
            "   </crs:Group>",
            "  </rdf:Description>",
            " </rdf:RDF>",
            "</x:xmpmeta>",
            '<?xpacket end="w"?>',
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_readme(out_dir: str, manifest: dict) -> None:
    lines = [
        "# Pilot XMP（仅试看）",
        "",
        f"- build: `{manifest['build_id']}`",
        f"- mapping: `{MAPPING_VERSION}`",
        "- 状态：pilot_only，application_ready=false",
        "",
        "## 文件",
        "",
    ]
    for preset in manifest["presets"]:
        lines.append(f"- `{preset['file']}`：{preset['scope']}")
    lines.extend(
        [
            "",
            "## 导入",
            "",
            "Lightroom Classic → 预设 → 导入预设，选择本目录的 `.xmp`；",
            "把对应预设套到 2–3 张同场景测试图上回看。",
            "",
            "## 注意",
            "",
            "- 桶预设里的 `Contrast2012`/`Clarity2012` 来自不稳定统计，仅用于试看；",
            "- 白平衡未映射，保持 As Shot；色温/色调/分离色调没有可靠映射，已跳过；",
            "- 不要批量套用或分发；正式应用前必须补齐 §13 验证。",
            "",
        ]
    )
    atomic_write_bytes(os.path.join(out_dir, "README.md"), "\n".join(lines).encode("utf-8"))

# -*- coding: utf-8 -*-
"""生成一个完全合成的示例指纹包（examples/sample_package）。

用途：让仓库克隆后无需任何私有数据即可校验指纹包、跑单元测试与演示在线解码。
包内所有参数与叙事都是示意值，不代表任何真实摄影师风格，也不含任何原图或单图分析原文。

用法：
    python examples/make_sample_package.py [--force]

重新生成会改变 package_hash，若 golden 基线依赖旧包，需同步重录：
    python -m online_decode.tools.record_golden
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from offline_profile.fingerprint_package import export_fingerprint_package  # noqa: E402

BUILD = REPO / ".sample_build"
OUT = REPO / "examples" / "sample_package"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_hash(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def param(value, unit, span, consistency, n, status="published", **extra):
    p25 = round(value - span / 2, 4)
    p75 = round(value + span / 2, 4)
    payload = {
        "value": value,
        "kind": "numeric",
        "unit": unit,
        "p25": p25,
        "p75": p75,
        "span": span,
        "n": n,
        "support": 1.0,
        "consistency": consistency,
        "status": status,
        "apply_policy": "anchor" if status == "published" else "prior_only",
    }
    payload.update(extra)
    return payload


NARRATIVE = {
    "natural_landscape": {
        "summary": "示例桶：自然风景。整体走低对比、收敛高光、局部提饱和的路子，只作演示用。",
        "mood": "平静、开阔。",
        "subject_context": "山脊、水面、草木等常见自然主体。",
        "lighting_context": "以柔和侧光与阴天散射光为主，高光不过曝。",
        "color_language": "中性偏暖的土色与克制的蓝绿并置。",
        "composition_tendencies": "水平线稳定，前景留出引导线。",
        "mood_keywords": ["平静", "开阔", "克制"],
        "recurring_motifs": ["压住的高光", "留细节的暗部"],
        "exceptions": ["强逆光剪影不适用本条叙事"],
        "cautions": ["不要对整桶统一加暖", "示例数据，不可当真实风格依据"],
    },
    "urban_architecture": {
        "summary": "示例桶：城市建筑。强调线条清晰、结构对比与冷灰基调，只作演示用。",
        "mood": "利落、冷静。",
        "subject_context": "楼体、街面、玻璃与金属材质。",
        "lighting_context": "硬光与大面积阴影并存，允许高反差。",
        "color_language": "冷灰为主，少量暖色招牌作点缀。",
        "composition_tendencies": "垂直线校正优先，强调几何秩序。",
        "mood_keywords": ["利落", "冷静", "秩序"],
        "recurring_motifs": ["清晰的边缘", "冷灰中间调"],
        "exceptions": ["夜景霓虹另走一套暖色策略"],
        "cautions": ["不要给整桶加胶片颗粒", "示例数据，不可当真实风格依据"],
    },
}

BUCKET_PARAMS = {
    "natural_landscape": {
        "exposure_ev": param(0.0, "EV", 0.4, 0.72, 12),
        "contrast": param(6.0, "index", 10.0, 0.64, 12),
        "highlights": param(-8.0, "index", 8.0, 0.81, 12),
        "shadows": param(8.0, "index", 10.0, 0.66, 12),
        "vibrance": param(14.0, "index", 20.0, 0.58, 12),
        "saturation": param(4.0, "index", 6.0, 0.62, 12),
        "clarity": param(9.0, "index", 15.0, 0.55, 12),
        "temperature_shift": param(2.0, "index", 5.0, 0.61, 12),
    },
    "urban_architecture": {
        "exposure_ev": param(0.0, "EV", 0.3, 0.75, 10),
        "contrast": param(12.0, "index", 12.0, 0.7, 10),
        "highlights": param(-12.0, "index", 10.0, 0.78, 10),
        "shadows": param(-4.0, "index", 8.0, 0.6, 10),
        "vibrance": param(9.0, "index", 12.0, 0.57, 10),
        "saturation": param(-2.0, "index", 4.0, 0.63, 10),
        "clarity": param(15.0, "index", 14.0, 0.69, 10),
        "dehaze": param(4.0, "index", 8.0, 0.52, 10),
    },
}

WIDE_PARAMS = {
    "natural_landscape": {
        "sharpening_amount": param(32.0, "index", 26.0, 0.31, 12, status="wide_dispersion"),
    },
    "urban_architecture": {
        "sharpening_amount": param(38.0, "index", 30.0, 0.29, 10, status="wide_dispersion"),
    },
}


def build_profile() -> dict:
    buckets = {}
    bucket_parameters = {}
    for bucket_id, params in BUCKET_PARAMS.items():
        buckets[bucket_id] = {
            "bucket_status": "full",
            "n_core": 12 if bucket_id == "natural_landscape" else 10,
            "n_train": 12 if bucket_id == "natural_landscape" else 10,
            "n_holdout": 0,
            "deltas": {
                "highlights": {"delta": -2.0, "unit": "index"},
                "vibrance": {"delta": 3.0, "unit": "index"},
            },
            "unresolved": [],
        }
        merged = dict(params)
        merged.update(WIDE_PARAMS[bucket_id])
        bucket_parameters[bucket_id] = merged
    return {
        "profile_schema_version": "1.0.0",
        "profile_shape": "bucket_primary",
        "profile_id": "profile_sample_demo",
        "build_id": "sample_demo",
        "status": "draft",
        "style_name": "示例风格（合成数据）",
        "validation_scope": "analysis_consistency",
        "application_ready": False,
        "input_fingerprint": "0" * 64,
        "config_hash": "sample_config_hash",
        "response_set_hash": "1" * 64,
        "aggregate_hash": "2" * 64,
        "buckets": buckets,
        "bucket_parameters": bucket_parameters,
    }


def build_signature() -> dict:
    return {
        "signature_schema_version": "1.0.0",
        "taxonomy_version": "scene_v1",
        "registry_version": "1.0.0",
        "evidence_scope": "cross_bucket",
        "parameters": {
            "highlights": {
                "central": -9.0, "unit": "index", "kind": "numeric",
                "cross_scene_invariant": True,
                "evidence_buckets": ["natural_landscape", "urban_architecture"],
                "sign_agreement": 1.0, "weighted_consistency": 0.79,
            },
            "vibrance": {
                "central": 12.0, "unit": "index", "kind": "numeric",
                "cross_scene_invariant": True,
                "evidence_buckets": ["natural_landscape", "urban_architecture"],
                "sign_agreement": 1.0, "weighted_consistency": 0.57,
            },
            "clarity": {
                "central": 10.0, "unit": "index", "kind": "numeric",
                "cross_scene_invariant": True,
                "evidence_buckets": ["natural_landscape", "urban_architecture"],
                "sign_agreement": 1.0, "weighted_consistency": 0.56,
            },
            "sharpening_amount": {
                "central": 35.0, "unit": "index", "kind": "numeric",
                "cross_scene_invariant": True,
                "evidence_buckets": ["natural_landscape", "urban_architecture"],
                "sign_agreement": 1.0, "weighted_consistency": 0.3,
            },
        },
        "invariant_parameters": ["highlights", "vibrance", "clarity", "sharpening_amount"],
        "scene_dependent_parameters": [],
        "insufficient_evidence": [
            {"key": "exposure_ev", "reason": "sample_data"},
            {"key": "temperature_shift", "reason": "sample_data"},
            {"key": "vignette_amount", "reason": "sample_data"},
        ],
        "constraints": [],
        "local_adjustment_policy": "no local adjustments",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成示例指纹包")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的输出")
    args = parser.parse_args()

    if OUT.exists():
        if not args.force:
            print(f"[skip] {OUT} 已存在；如需重建请加 --force")
            return 0
        shutil.rmtree(OUT)
    if BUILD.exists():
        shutil.rmtree(BUILD)

    write_json(BUILD / "profile.json", build_profile())
    write_json(BUILD / "global_signature.json", build_signature())

    index = {"buckets": [], "note": "合成示例包；参数与叙事均为示意值。"}
    for bucket_id, narrative in NARRATIVE.items():
        bucket_dir = BUILD / "buckets" / bucket_id
        narrative_payload = {
            "narrative_schema_version": "1.0.0",
            "bucket_id": bucket_id,
            "status": "draft",
            "language": "zh",
            **narrative,
            "evidence_refs": [],
        }
        write_json(bucket_dir / "narrative.json", narrative_payload)
        narrative_hash = canonical_hash(narrative_payload)
        profile_hash = canonical_hash({"bucket_id": bucket_id, "sample": True})
        write_json(
            bucket_dir / "profile.json",
            {
                "profile_schema_version": "1.0.0",
                "profile_type": "bucket_style",
                "profile_id": f"bucket_{bucket_id}",
                "bucket_id": bucket_id,
                "taxonomy_version": "scene_v1",
                "status": "draft",
                "bucket_status": "full",
                "n_core": 12 if bucket_id == "natural_landscape" else 10,
                "n_train": 12 if bucket_id == "natural_landscape" else 10,
                "n_holdout": 0,
                "profile_hash": profile_hash,
                "parameters": BUCKET_PARAMS[bucket_id],
                "narrative_ref": {
                    "file": f"buckets/{bucket_id}/narrative.json",
                    "narrative_hash": narrative_hash,
                },
            },
        )
        write_json(
            bucket_dir / "delta.json",
            {"deltas": {"highlights": -2.0, "vibrance": 3.0}, "note": "合成示例增量"},
        )
        write_json(
            bucket_dir / "stats.json",
            {
                "params": {
                    key: {"n": spec["n"], "p25": spec["p25"], "p75": spec["p75"], "span": spec["span"]}
                    for key, spec in BUCKET_PARAMS[bucket_id].items()
                },
                "note": "合成示例统计",
            },
        )
        index["buckets"].append(
            {
                "bucket_id": bucket_id,
                "profile_file": f"buckets/{bucket_id}/profile.json",
                "profile_hash": profile_hash,
                "narrative_file": f"buckets/{bucket_id}/narrative.json",
                "narrative_hash": narrative_hash,
            }
        )
    write_json(BUILD / "buckets" / "index.json", index)

    manifest = export_fingerprint_package(str(BUILD), str(OUT), overwrite=True)
    shutil.rmtree(BUILD, ignore_errors=True)
    print(f"[ok] {OUT}")
    print(f"     package_id={manifest['package_id']}  fingerprint_id={manifest['fingerprint_id']}")
    print(f"     files={manifest['counts']['files']}  buckets={manifest['counts']['buckets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

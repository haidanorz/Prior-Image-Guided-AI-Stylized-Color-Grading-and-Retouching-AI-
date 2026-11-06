"""decode_report.json 组装。

报告不是日志的搬运工，它要回答四个问题：
  1. 这次用了哪份指纹、哪个桶、凭什么这么判（可回放）；
  2. 每轮到底改了什么、看到了什么、为什么停；
  3. 最佳轮是哪一轮、凭什么选它；
  4. 成片是哪份设置导出的、哈希对不对得上。
"""

from __future__ import annotations

from typing import Any

from .config import Config
from .state import SessionState


def build(state: SessionState, cfg: Config) -> dict[str, Any]:
    rounds = []
    for record in state.rounds:
        evaluation = record.get("evaluation") or {}
        rounds.append({
            "round": record.get("round"),
            "ok": record.get("ok"),
            "job_ids": record.get("jobs") or {},
            "xmp_sha256": (record.get("xmp") or {}).get("sha256"),
            "model_input": record.get("model_input"),
            "settings_hash": (record.get("apply") or {}).get("settings_hash"),
            "snapshot": {
                "snapshot_id": (record.get("snapshot") or {}).get("snapshot_id"),
                "complete": (record.get("snapshot") or {}).get("complete"),
                "settings_hash": (record.get("snapshot") or {}).get("settings_hash"),
            },
            "render": {
                "path": (record.get("render") or {}).get("path"),
                "local_path": (record.get("render") or {}).get("local_path"),
                "settings_hash": (record.get("render") or {}).get("settings_hash"),
                "width": (record.get("render") or {}).get("width"),
                "height": (record.get("render") or {}).get("height"),
            },
            "written": (record.get("ledger") or {}).get("written_values"),
            "anchor_usage": (record.get("ledger") or {}).get("anchor_usage") or {},
            "rejected": (record.get("ledger") or {}).get("rejected") or [],
            "improvement": evaluation.get("improvement"),
            "decision": evaluation.get("decision"),
            "style_alignment": (evaluation.get("style_alignment") or {}).get("verdict"),
            "technical_defects": (evaluation.get("technical_quality") or {}).get("defects") or [],
            "error": record.get("error"),
            "warnings": record.get("warnings") or [],
        })

    brief = {}
    brief_path = state.root / "creative_brief.json"
    if brief_path.is_file():
        import json

        brief = json.loads(brief_path.read_text(encoding="utf-8"))

    scene = (brief.get("scene") or {})
    decode = brief.get("decode") or {}
    anchor_totals = {"bucket": 0, "global_fallback": 0, "unreferenced": 0, "not_evaluable_written": 0}
    for item in rounds:
        for key in anchor_totals:
            anchor_totals[key] += int((item.get("anchor_usage") or {}).get(key) or 0)
    return {
        "report_schema_version": "1.0.0",
        "ok": state.status == "completed",
        "session_id": state.session_id,
        "status": state.status,
        "substep": state.substep,
        "stop_reason": state.stop_reason,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "config": cfg.to_report(),
        "input": {
            "photo": state.photo,
            "source": state.baseline.get("source"),
            "mode": state.mode,
        },
        "fingerprint": state.fingerprint,
        "decode": {
            "mode": decode.get("mode") or (state.fingerprint or {}).get("decode_mode"),
            "bucket_id": decode.get("bucket_id"),
            "fallback": decode.get("fallback"),
            "not_evaluable_count": len(decode.get("not_evaluable") or []),
        },
        "anchor_usage": anchor_totals,
        "scene": {
            "bucket_id": (scene.get("decision") or {}).get("bucket_id"),
            "level": (scene.get("decision") or {}).get("level"),
            "top_score": (scene.get("decision") or {}).get("top_score"),
            "margin": (scene.get("decision") or {}).get("margin"),
            "fallback": (scene.get("decision") or {}).get("fallback"),
            "fallback_reason": (scene.get("decision") or {}).get("fallback_reason"),
            "branch_rationale": scene.get("branch_rationale"),
            "narrative_ref": scene.get("narrative_ref"),
            "narrative_hash": scene.get("narrative_hash"),
            "hit_cautions": scene.get("hit_cautions"),
        },
        "rounds_used": len(state.rounds),
        "max_rounds": state.max_rounds,
        "stop_on_single_no_progress": {
            "value": state.stop_on_single_no_progress,
            "source": state.stop_source,
        },
        "rounds": rounds,
        "best": state.best,
        "final": state.final,
        "warnings": state.warnings,
        "errors": state.errors,
        "pilot_only": state.mode == "pilot",
        "application_ready": bool(state.fingerprint.get("application_ready")),
        "files": {
            "session": str(state.root / "session.json"),
            "brief": str(state.root / "creative_brief.json"),
            "scene": str(state.root / "scene.json"),
            "rounds": str(state.root / "rounds"),
        },
    }

"""录制 Golden 轨迹：把一次完整会话的输入与输出摘要固化成回归基线。

用法：
    python -m online_decode.tools.record_golden              # 写入 cases.json
    python -m online_decode.tools.record_golden --check      # 只比对，不写

录的是**输入轨迹 + 输出摘要**，不是整个报告：
报告里有时间戳、随机 session_id、临时路径，逐字节比对没有意义；
真正要守住的是"选了哪一轮、为什么停、写了什么、哈希链是否闭合"。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from online_decode.config import Config
from online_decode.drivers.scripted import ScriptedDriver
from online_decode.feedback.lr_port import FakeLrPort
from online_decode.report import build as build_report
from online_decode.session import SessionController
from online_decode.tests.helpers import BASE_SETTINGS, SCENE
from online_decode.tests.test_session import brief_response, evaluation, plan_response

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "tests" / "golden" / "cases.json"
PACKAGE = Path(os.environ.get("ONLINE_DECODE_PACKAGE") or ROOT / "examples" / "sample_package")


def digest(state, cfg) -> dict[str, Any]:
    report = build_report(state, cfg)
    return {
        "ok": report["ok"],
        "status": report["status"],
        "max_rounds": report["max_rounds"],
        "rounds_used": report["rounds_used"],
        "stop_reason": report["stop_reason"],
        "best_round": (report["best"] or {}).get("round"),
        "scene_bucket": report["scene"]["bucket_id"],
        "scene_level": report["scene"]["level"],
        "pilot_only": report["pilot_only"],
        "rounds": [
            {
                "round": item["round"],
                "ok": item["ok"],
                "improvement": item["improvement"],
                "written": item["written"],
                "settings_hash": item["settings_hash"],
                "render_hash": (item["render"] or {}).get("settings_hash"),
                "model_input_source": (item["model_input"] or {}).get("source"),
                "error_code": (item["error"] or {}).get("code"),
            }
            for item in report["rounds"]
        ],
        "final_hash_equals_best": bool(
            report["final"] and report["best"]
            and report["final"]["settings_hash"] == report["best"]["settings_hash"]
        ),
        "error_codes": [item.get("code") for item in report["errors"]],
    }


def case_success() -> dict[str, Any]:
    return {
        "name": "two_rounds_goal_met",
        "max_rounds": 3,
        "port_flags": {},
        "scene": copy.deepcopy(SCENE),
        "brief": brief_response(),
        "plans": [plan_response(-11)],
        "evaluations": [
            evaluation("better", "continue", style="partially_met"),
            evaluation("better", "stop_and_select", style="met"),
        ],
    }


def case_hash_mismatch() -> dict[str, Any]:
    return {
        "name": "render_hash_drift_voids_round",
        "max_rounds": 2,
        "port_flags": {"render_hash_drift": True},
        "scene": copy.deepcopy(SCENE),
        "brief": brief_response(),
        "plans": [],
        "evaluations": [evaluation("better", "stop_and_select")],
    }


def replay(case: dict[str, Any]):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        photo = root / "IMG_0001.jpg"
        photo.write_bytes(b"\xff\xd8\xff\xd9")
        cfg = Config(max_rounds=case["max_rounds"], workspace=root)
        cfg.validate()
        port = FakeLrPort(BASE_SETTINGS, workspace=root, **case.get("port_flags") or {})
        driver = ScriptedDriver(
            scene=copy.deepcopy(case.get("scene") or SCENE),
            brief=copy.deepcopy(case["brief"]),
            plans=copy.deepcopy(case["plans"]),
            evaluations=copy.deepcopy(case["evaluations"]),
        )
        controller = SessionController(cfg, port, driver)
        controller.run(str(photo), str(PACKAGE))
        if not controller.state:
            raise RuntimeError("会话没有建立")
        return digest(controller.state, cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    cases = []
    for maker in (case_success, case_hash_mismatch):
        case = maker()
        case["expected"] = replay(case)
        cases.append(case)

    payload = {
        "golden_schema_version": "1.0.0",
        "note": "由 record_golden 生成；改动状态机、选轮或守卫后先确认差异再重录",
        "package_relative_path": "examples/sample_package",
        "cases": cases,
    }

    if args.check:
        if not OUT.is_file():
            print(f"[FAIL] 缺少 {OUT}", flush=True)
            return 2
        old = json.loads(OUT.read_text(encoding="utf-8"))
        if old != payload:
            print("[FAIL] Golden 轨迹与当前实现不一致，检查差异后决定是否重录")
            return 3
        print(f"[ok] {len(cases)} 条 Golden 轨迹一致")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] 写入 {OUT}（{len(cases)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

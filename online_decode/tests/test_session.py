"""控制器测试。

这些测试的重点不是"跑通"，而是**复现真机上出现过的坏情况**：
预览哈希漂移、评价无法核验、首轮全被守卫拒绝、单轮无改善。

用 FakeLrPort + ScriptedDriver，任何一条都不需要 Lightroom。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from online_decode.config import Config
from online_decode.feedback.lr_port import FakeLrPort
from online_decode.drivers.scripted import ScriptedDriver
from online_decode.params import xmp_renderer
from online_decode.tests.helpers import BASE_SETTINGS, SCENE
from online_decode.session import SessionController
from online_decode.util import canonical_json, sha256_bytes, sha256_file

PACKAGE = Path(__file__).resolve().parents[2] / "examples" / "sample_package"


def brief_response(**overrides):
    data = {
        "image_id": "target",
        "scene_observation": {
            "primary_candidates": [
                {"label": "natural_landscape", "score": 0.85},
                {"label": "urban_architecture", "score": 0.10},
            ],
            "lighting": "soft",
            "brightness": "normal",
            "backlit": False,
        },
        "branch_rationale": "干旱地貌支系：暖色层理配灰白天光，适用柔和漫射光的影调语言",
        "hit_cautions": ["不能用单一色调概括全桶"],
        "preserve": ["天空灰蓝层次", "远山雾感"],
        "goals": ["保留天空层次", "前景暗部不死黑", "整体暖而不艳"],
        "avoid": ["高光溢出", "整体偏绿"],
        "target_corrections": [],
        "initial_xmp": xmp_renderer.render({"Highlights2012": -10}),
        "param_notes": {
            "Highlights2012": {"evidence_level": "runtime_reference",
                               "ref": "fingerprint.global.highlights", "reason": "全局参考值"},
        },
        "uncertainties": [],
    }
    data.update(overrides)
    return data


def plan_response(value=-12):
    return {
        "xmp": xmp_renderer.render({"Highlights2012": value}),
        "param_notes": {
            "Highlights2012": {"evidence_level": "runtime_reference",
                               "ref": "fingerprint.global.highlights", "reason": "按参考微调"},
        },
        "expected_effects": ["中间调更通透"],
    }


def evaluation(improvement="better", decision="continue", defects=None, style="met"):
    return {
        "style_alignment": {"verdict": style, "notes": "影调方向符合 brief"},
        "branch_alignment": {"verdict": "met", "notes": "仍属柔和漫射光语言"},
        "technical_quality": {
            "verdict": "not_met" if defects else "met",
            "defects": defects or [],
        },
        "content_preservation": {"verdict": "met", "notes": "天空与远山保留"},
        "improvement": improvement,
        "decision": decision,
        "is_best_candidate": improvement == "better",
        "evidence": [{"anchor": "whole", "observation": "天空层次保留，暗部没有糊成一片"}],
        "next_adjustment": {},
        "preserve_violations": [],
        "caution_violations": [],
    }


class SessionTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.photo = self.root / "work" / "IMG_0001.jpg"
        self.photo.parent.mkdir(parents=True, exist_ok=True)
        self.photo.write_bytes(b"\xff\xd8\xff\xd9")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_session(self, *, driver, port=None, max_rounds=3, stop_on_single=None,
                    package=None, decode_mode="auto"):
        cfg = Config(
            max_rounds=max_rounds,
            stop_on_single_no_progress=stop_on_single,
            workspace=self.root,
            decode_mode=decode_mode,
        )
        cfg.validate()
        port = port or FakeLrPort(BASE_SETTINGS, workspace=self.root)
        controller = SessionController(cfg, port, driver)
        return controller.run(str(self.photo), str(package or PACKAGE))

    def make_bucket_primary_package(self) -> Path:
        root = self.root / "pkg_bucket_primary"
        root.mkdir(parents=True, exist_ok=True)
        fingerprint = {
            "fingerprint_schema_version": "1.1.0",
            "status": "draft",
            "application_ready": False,
            "validation_scope": "analysis_consistency",
            "shape": "bucket_primary",
            "global_role": "fallback",
            "runtime_rule": "scene -> bucket parameters; missing -> global; both missing -> not_evaluable",
            "global": {"highlights": {"central": -11.0}},
            "buckets": {
                "natural_landscape": {
                    "parameters": {
                        "highlights": {"value": -10.0, "status": "published",
                                       "apply_policy": "anchor", "span": 5.0},
                        "clarity": {"value": 10.0, "status": "published",
                                    "apply_policy": "anchor", "span": 15.0},
                    },
                    "priors": {"shadows": {"value": 5.0, "span": 20.0}},
                    "not_evaluable": ["contrast"],
                }
            },
            "insufficient_evidence": [{"key": "contrast", "reason": "single_bucket_evidence"}],
        }
        fingerprint["fingerprint_hash"] = sha256_bytes(
            canonical_json(fingerprint).encode("utf-8")
        )
        fingerprint["fingerprint_id"] = "fp_test_bucket_primary"
        fp_path = root / "fingerprint.json"
        fp_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
        files = [{"path": "fingerprint.json", "sha256": sha256_file(fp_path)}]
        manifest = {
            "package_schema_version": "1.1.0",
            "fingerprint_hash": fingerprint["fingerprint_hash"],
            "files": files,
        }
        manifest["package_hash"] = sha256_bytes(
            canonical_json({
                "package_schema_version": manifest["package_schema_version"],
                "fingerprint_hash": manifest["fingerprint_hash"],
                "files": files,
            }).encode("utf-8")
        )
        (root / "package_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return root


class HappyPath(SessionTestBase):
    def test_two_rounds_then_goal_met(self):
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(),
            plans=[plan_response(-11)],
            evaluations=[
                evaluation("better", "continue", style="partially_met"),
                evaluation("better", "stop_and_select", style="met"),
            ],
        )
        report = self.run_session(driver=driver)

        self.assertTrue(report["ok"], msg=json.dumps(report.get("errors"), ensure_ascii=False))
        self.assertEqual(report["rounds_used"], 2)
        self.assertEqual(report["best"]["round"], 2)
        self.assertEqual(report["stop_reason"], "goal_met")
        self.assertEqual(report["final"]["best_round"], 2)
        self.assertTrue(Path(report["final"]["path"]).is_file())

        session = json.loads((self.root / "sessions" / report["session_id"] / "session.json").read_text("utf-8"))
        self.assertEqual(session["status"], "completed")
        self.assertEqual(session["max_rounds"], 3)
        self.assertTrue(session["stop_on_single_no_progress"])

    def test_single_round(self):
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(),
            plans=[],
            evaluations=[evaluation("better", "stop_and_select")],
        )
        report = self.run_session(driver=driver, max_rounds=1)
        self.assertTrue(report["ok"])
        self.assertEqual(report["rounds_used"], 1)
        self.assertEqual(report["stop_reason"], "goal_met")

    def test_budget_exhausted_without_stop_signal(self):
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(),
            plans=[plan_response(-11), plan_response(-12)],
            evaluations=[
                evaluation("better", "continue"),
                evaluation("better", "continue"),
                evaluation("better", "continue"),
            ],
        )
        report = self.run_session(driver=driver, max_rounds=3)
        self.assertTrue(report["ok"])
        self.assertEqual(report["rounds_used"], 3)
        self.assertEqual(report["stop_reason"], "budget")

    def test_first_round_cannot_stop(self):
        """强制两轮（2026-11-06 决策）：首轮 stop_and_select 不生效，第 2 轮才允许收尾。"""
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(),
            plans=[plan_response(-11)],
            evaluations=[
                evaluation("better", "stop_and_select"),
                evaluation("better", "stop_and_select"),
            ],
        )
        report = self.run_session(driver=driver, max_rounds=3)

        self.assertTrue(report["ok"], msg=json.dumps(report.get("errors"), ensure_ascii=False))
        self.assertEqual(report["rounds_used"], 2)
        self.assertEqual(report["stop_reason"], "goal_met")
        self.assertEqual(report["config"]["min_rounds"], 2)
        self.assertEqual(len(driver.requests["plan"]), 1)
        # 首轮决策仍如实留档，只是不生效
        self.assertEqual(report["rounds"][0]["decision"], "stop_and_select")

    def test_plan_stop_before_min_rounds_holds_previous_values(self):
        """强制轮次内 plan 请求停止 → 按保留上轮参数继续验证，而不是跳过该轮。"""
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(),
            plans=[{"stop": True}],
            evaluations=[
                evaluation("better", "continue"),
                evaluation("same", "continue"),
            ],
        )
        report = self.run_session(driver=driver, max_rounds=3)

        self.assertTrue(report["ok"], msg=json.dumps(report.get("errors"), ensure_ascii=False))
        self.assertEqual(report["rounds_used"], 2)
        self.assertEqual(report["stop_reason"], "no_progress")
        self.assertEqual(
            report["rounds"][1]["written"], report["rounds"][0]["written"]
        )
        self.assertEqual(report["best"]["round"], 1)


class IncompleteApplyHash(SessionTestBase):
    """真机（Sony 重 RAW 刚导入、LR 尚未物化完 develop 数据）复现：

    apply 时刻只读到不完整子集 → settings_hash_complete=false，而紧随其后的
    get_settings 已读全；旧逻辑把 apply 的不完整哈希当渲染期望值，闸门误杀正确写入。
    修复口径：apply 哈希不完整时，期望值与闸门改用本轮完整快照哈希。
    """

    def test_round_uses_snapshot_hash_when_apply_read_is_incomplete(self):
        port = FakeLrPort(BASE_SETTINGS, workspace=self.root, apply_hash_incomplete=True)
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(),
            plans=[],
            evaluations=[evaluation("better", "stop_and_select")],
        )
        report = self.run_session(driver=driver, port=port, max_rounds=1)

        self.assertTrue(report["ok"], msg=json.dumps(report.get("errors"), ensure_ascii=False))
        self.assertEqual(report["final"]["best_round"], 1)
        self.assertTrue(Path(report["final"]["path"]).is_file())

        session_root = self.root / "sessions" / report["session_id"]
        session = json.loads((session_root / "session.json").read_text("utf-8"))
        record = session["rounds"][0]
        self.assertFalse(record["apply"]["complete"])
        self.assertTrue(record.get("state_hash"))
        # state_hash 必须等于完整快照哈希，而不是 apply 的不完整子集哈希
        self.assertEqual(record["state_hash"], record["snapshot"]["settings_hash"])
        self.assertNotEqual(record["state_hash"], record["apply"]["settings_hash"])
        self.assertEqual(report["best"]["settings_hash"], record["state_hash"])
        self.assertTrue(any("不完整" in w for w in report["rounds"][0].get("warnings") or []))

    def test_unstable_reads_degrade_to_ungated_render(self):
        """读数持续物化（重 RAW 刚导入）：快照与 apply 哈希都不可信时按 pilot 降级。

        基线与轮次读数始终不完整 → 渲染不带期望哈希、显式留痕，会话仍能出片。
        """
        from online_decode.feedback import round as round_mod

        round_mod.STABLE_READ_DELAY_SEC = 0.01
        try:
            port = FakeLrPort(
                BASE_SETTINGS, workspace=self.root,
                apply_hash_incomplete=True, unstable_reads=8,
            )
            driver = ScriptedDriver(scene=SCENE, 
                brief=brief_response(),
                plans=[],
                evaluations=[evaluation("better", "stop_and_select")],
            )
            report = self.run_session(driver=driver, port=port, max_rounds=1)
        finally:
            round_mod.STABLE_READ_DELAY_SEC = 2.5

        self.assertTrue(report["ok"], msg=json.dumps(report.get("errors"), ensure_ascii=False))
        self.assertEqual(report["final"]["best_round"], 1)
        self.assertTrue(Path(report["final"]["path"]).is_file())
        self.assertFalse(report["rounds"][0]["snapshot"]["complete"])
        self.assertTrue(any(
            "pilot 降级" in w for w in report["rounds"][0].get("warnings") or []
        ))
        self.assertTrue(any(
            "pilot 降级" in w for w in report.get("warnings") or []
        ))


class BucketPrimaryFlow(SessionTestBase):
    def scene_response(self):
        return {
            "image_id": "target",
            "scene_observation": {
                "primary_candidates": [
                    {"label": "natural_landscape", "score": 0.85},
                    {"label": "urban_architecture", "score": 0.10},
                ],
                "lighting": "soft",
                "brightness": "normal",
                "backlit": False,
            },
        }

    def test_scene_runs_before_brief_and_carries_anchors(self):
        package = self.make_bucket_primary_package()
        driver = ScriptedDriver(
            scene=self.scene_response(),
            brief=brief_response(),
            plans=[plan_response(-11)],
            evaluations=[
                evaluation("better", "continue"),
                evaluation("better", "stop_and_select"),
            ],
        )
        report = self.run_session(driver=driver, package=package)

        self.assertTrue(report["ok"], msg=json.dumps(report.get("errors"), ensure_ascii=False))
        self.assertEqual(len(driver.requests["scene"]), 1)
        self.assertEqual(driver.requests["scene"][0]["task"], "scene")
        self.assertNotIn("scene", driver.requests["scene"][0])

        brief_request = driver.requests["brief"][0]
        self.assertIn("scene", brief_request)
        self.assertIn("Highlights2012", brief_request["scene"]["anchors"])
        self.assertEqual(
            brief_request["scene"]["anchors"]["Highlights2012"]["role"], "bucket"
        )
        self.assertEqual(report["decode"]["mode"], "bucket_primary")
        self.assertEqual(report["decode"]["bucket_id"], "natural_landscape")
        self.assertEqual(report["decode"]["not_evaluable_count"], 1)
        session_root = self.root / "sessions" / report["session_id"]
        self.assertTrue((session_root / "scene.json").is_file())
        self.assertGreaterEqual(report["anchor_usage"]["bucket"], 1)

    def test_not_evaluable_param_is_writable(self):
        """2026-11-06 用户决策：指纹未覆盖的字段不再禁写，可按本图自主调整。

        旧行为（已废弃）：not_evaluable 字段非 target_adaptive 一律 NOT_EVALUABLE 拒绝。
        新行为：照常写入，证据等级留在台账 entry 上，不进 rejected。
        """
        package = self.make_bucket_primary_package()
        driver = ScriptedDriver(
            scene=self.scene_response(),
            brief=brief_response(
                initial_xmp=xmp_renderer.render({"Highlights2012": -11, "Contrast2012": 20}),
            ),
            plans=[plan_response(-11)],
            evaluations=[
                evaluation("better", "stop_and_select"),
                evaluation("same", "continue"),
            ],
        )
        report = self.run_session(driver=driver, package=package)
        self.assertTrue(report["ok"], msg=json.dumps(report.get("errors"), ensure_ascii=False))
        round_record = report["rounds"][0]
        codes = {item["code"] for item in round_record["rejected"]}
        self.assertNotIn("NOT_EVALUABLE", codes)
        self.assertIn("Highlights2012", round_record["written"])
        self.assertIn("Contrast2012", round_record["written"])
        self.assertEqual(round_record["written"]["Contrast2012"], 20)
        # 自主写入项仍计入 not_evaluable_written 统计（报告自带字段，非新增审计）
        self.assertGreaterEqual(round_record["anchor_usage"]["not_evaluable_written"], 1)


class FailureModes(SessionTestBase):
    def test_render_hash_drift_voids_round(self):
        port = FakeLrPort(BASE_SETTINGS, workspace=self.root, render_hash_drift=True)
        driver = ScriptedDriver(scene=SCENE, brief=brief_response(), plans=[], evaluations=[evaluation()])
        report = self.run_session(driver=driver, port=port, max_rounds=2)

        self.assertFalse(report["ok"])
        self.assertEqual(report["rounds"][0]["error"]["code"], "RENDER_HASH_MISMATCH")
        self.assertIsNone(report["best"])
        self.assertEqual(report["status"], "failed")

    def test_unverifiable_evaluation_blocks_session(self):
        bad = evaluation()
        bad["evidence"] = [{"anchor": "whole", "metric": "clip_high_ratio", "value": 0.9,
                            "observation": "高光大面积溢出"}]
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(), plans=[],
            evaluations=[bad, dict(bad)],  # 允许一次重做，第二次仍不合法
        )
        report = self.run_session(driver=driver, max_rounds=2)
        self.assertFalse(report["ok"])
        self.assertIn("EVIDENCE_UNVERIFIED", json.dumps(report["errors"], ensure_ascii=False))

    def test_all_plan_rejected(self):
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(initial_xmp=xmp_renderer.render({"GrainAmount": 20})),
            plans=[],
            evaluations=[evaluation()],
        )
        report = self.run_session(driver=driver, max_rounds=2)
        self.assertFalse(report["ok"])
        self.assertIn("PLAN_REJECTED", json.dumps(report["errors"], ensure_ascii=False))

    def test_no_progress_stops_at_second_round(self):
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(),
            plans=[plan_response(-9)],
            evaluations=[evaluation("better", "continue"), evaluation("same", "continue")],
        )
        report = self.run_session(driver=driver, max_rounds=3)
        self.assertEqual(report["stop_reason"], "no_progress")
        self.assertEqual(report["rounds_used"], 2)
        self.assertEqual(report["best"]["round"], 1)

    def test_export_hash_skipped_is_filled_by_controller(self):
        """真机成片 25MB 时插件会写 hash_skipped；报告里的哈希必须仍可复算。"""
        import hashlib
        from pathlib import Path

        port = FakeLrPort(BASE_SETTINGS, workspace=self.root, export_hash_skipped=True)
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(),
            plans=[],
            evaluations=[evaluation("better", "stop_and_select")],
        )
        report = self.run_session(driver=driver, port=port, max_rounds=1)
        self.assertTrue(report["ok"])
        final = report["final"]
        self.assertEqual(final["sha256_source"], "controller_fallback")
        self.assertTrue(final["sha256"].startswith("sha256:"))
        digest = hashlib.sha256(Path(final["path"]).read_bytes()).hexdigest()
        self.assertEqual(final["sha256"], "sha256:" + digest)
        self.assertTrue(any("补算" in item for item in report["warnings"]))

    def test_five_rounds_keeps_single_progress_rule_off(self):
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(),
            plans=[plan_response(-9), plan_response(-11), plan_response(-12), plan_response(-13)],
            evaluations=[
                evaluation("better", "continue"),
                evaluation("same", "continue"),
                evaluation("better", "continue"),
                evaluation("same", "continue"),
                evaluation("better", "continue"),
            ],
        )
        report = self.run_session(driver=driver, max_rounds=5)
        self.assertEqual(report["rounds_used"], 5)
        self.assertEqual(report["stop_reason"], "budget")
        self.assertFalse(report["stop_on_single_no_progress"]["value"])
        self.assertEqual(report["stop_on_single_no_progress"]["source"], "derived")


if __name__ == "__main__":
    unittest.main()

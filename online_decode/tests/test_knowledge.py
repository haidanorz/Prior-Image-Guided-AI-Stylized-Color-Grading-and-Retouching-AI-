"""参数知识层测试。

重点不是"能加载"，而是钉住三件容易悄悄跑偏的事：
  1. 知识库里的字段名必须都真实存在于 fields_manifest，否则模型会照着写一个
     不存在的字段，被守卫拒绝后白烧一轮；
  2. `writable` 标注必须与接口真实可写性一致（曲线是显式例外，见 README §8）；
  3. 注入不得改变任何数值结果——这是本层「只提示、不碰数值」的可执行证明，
     而不是一句文档承诺。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from online_decode.config import Config
from online_decode.drivers.scripted import ScriptedDriver
from online_decode.feedback.lr_port import FakeLrPort
from online_decode.params import knowledge as knowledge_mod
from online_decode.session import SessionController
from online_decode.tests.helpers import BASE_SETTINGS
from online_decode.tests.test_session import brief_response, evaluation, plan_response

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "fields_manifest.json"
PACKAGE = Path(__file__).resolve().parents[2] / "examples" / "sample_package"

SCENE = {
    "scene_observation": {
        "primary_candidates": [
            {"label": "natural_landscape", "score": 0.85},
            {"label": "urban_architecture", "score": 0.10},
        ],
        "lighting": "soft",
        "brightness": "normal",
        "backlit": False,
    }
}


def manifest() -> dict:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data.get("fields") or data


class ManifestConsistency(unittest.TestCase):
    def setUp(self) -> None:
        self.knowledge = knowledge_mod.load()
        self.manifest = manifest()
        self.entries = {e["field"]: e for e in self.knowledge["entries"]}

    def test_every_field_exists_in_manifest(self):
        unknown = sorted(set(self.entries) - set(self.manifest))
        self.assertEqual(unknown, [], msg=f"知识库出现清单外的字段名：{unknown}")

    def test_every_v1_field_is_covered(self):
        v1 = {f for f, v in self.manifest.items() if v.get("tier") == "v1"}
        missing = sorted(v1 - set(self.entries))
        self.assertEqual(missing, [], msg=f"这些可写字段没有知识条目：{missing}")

    def test_writable_matches_tier_except_declared_curve_exception(self):
        wrong = []
        for field, entry in self.entries.items():
            expect = self.manifest[field].get("tier") == "v1"
            if entry["writable"] == expect:
                continue
            # 唯一允许的偏差：曲线是 rdf:Seq，tier 是 v1 但接口写不进。
            # 必须在条目上显式声明，不允许静默不一致。
            if entry.get("writable_mismatch_with_manifest") and field.startswith("ToneCurve"):
                continue
            wrong.append((field, entry["writable"], self.manifest[field].get("tier")))
        self.assertEqual(wrong, [], msg=f"writable 与 tier 冲突且未声明例外：{wrong}")

    def test_curve_exception_is_declared_not_implicit(self):
        curves = [f for f in self.entries if f.startswith("ToneCurve")]
        self.assertEqual(len(curves), 5)
        for field in curves:
            self.assertFalse(self.entries[field]["writable"])
            self.assertTrue(self.entries[field].get("writable_mismatch_with_manifest"))
            self.assertTrue(self.entries[field].get("blocked_reason"))

    def test_hash_matches_content(self):
        self.assertTrue(
            knowledge_mod.hash_ok(self.knowledge),
            msg="knowledge_hash 与文件内容不符，改完条目要重算哈希",
        )

    def test_layer_is_declared_l3_and_not_style_evidence(self):
        self.assertEqual(self.knowledge["layer"], "L3_cross_project_reference")
        self.assertEqual(self.knowledge["scope"], "reference_only")
        bounds = self.knowledge["bounds"]
        self.assertTrue(bounds["does_not_enter_fingerprint"])
        self.assertTrue(bounds["does_not_affect_numeric_synthesis"])
        self.assertTrue(bounds["not_style_evidence"])
        self.assertIn("conflict_rule", bounds)

    def test_sources_are_cited(self):
        sources = {s["id"]: s for s in self.knowledge["sources"]}
        self.assertTrue(sources)
        for source in sources.values():
            self.assertTrue(source["url"].startswith("https://"))
        # 每条带引文的条目必须指向一个真实声明的来源
        for field, entry in self.entries.items():
            ref = entry.get("source_ref")
            if entry.get("quote"):
                self.assertIn(ref, sources, msg=f"{field} 引了原文却没指向已声明来源")


class CompactView(unittest.TestCase):
    def setUp(self) -> None:
        self.view = knowledge_mod.build()

    def test_has_audit_metadata(self):
        for key in ("knowledge_id", "layer", "status", "scope", "knowledge_hash",
                    "hash_verified", "full_ref", "bounds", "conflict_rule"):
            self.assertIn(key, self.view)
        self.assertTrue(self.view["hash_verified"])

    def test_detail_groups_expanded_others_collapsed(self):
        self.assertIn("Highlights2012", self.view["params"])
        self.assertIn("Dehaze", self.view["params"])
        # HSL 只给字段名，不逐条展开
        self.assertNotIn("SaturationAdjustmentRed", self.view["params"])
        self.assertIn("SaturationAdjustmentRed", self.view["other_groups"]["hsl"]["fields"])
        self.assertIn("note", self.view["other_groups"]["hsl"])

    def test_tone_zones_present(self):
        zones = self.view["tone_zones"]
        self.assertEqual(zones["Highlights2012"], [70, 90])
        self.assertEqual(zones["Whites2012"], [90, 100])

    def test_blocked_covers_ext_and_curves(self):
        blocked = set(self.view["blocked_fields"])
        self.assertIn("ToneCurvePV2012", blocked)
        self.assertIn("GrainAmount", blocked)
        self.assertIn("Sharpness", blocked)
        # 可写字段绝不能出现在 blocked 里
        self.assertNotIn("Highlights2012", blocked)

    def test_view_stays_small(self):
        size = len(json.dumps(self.view, ensure_ascii=False))
        # 每轮都要塞进请求包，失控会挤占模型的判断空间
        self.assertLess(size, 6000, msg=f"注入视图 {size} 字符，超过预算")


class Injection(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def run_session(self, *, enable: bool) -> tuple[dict, ScriptedDriver]:
        photo = self.root / f"t_{int(enable)}.jpg"
        photo.write_bytes(b"\xff\xd8\xff\xd9")
        cfg = Config(
            max_rounds=3, min_rounds=3, workspace=self.root,
            driver="scripted", parameter_knowledge=enable,
        )
        driver = ScriptedDriver(
            scene=SCENE,
            brief=brief_response(),
            plans=[plan_response(-11), plan_response(-12)],
            evaluations=[
                evaluation("better", "continue"),
                evaluation("better", "continue"),
                evaluation("better", "stop_and_select"),
            ],
        )
        report = SessionController(
            cfg, FakeLrPort(BASE_SETTINGS, workspace=self.root), driver
        ).run(str(photo), str(PACKAGE))
        return report, driver

    def test_brief_request_carries_knowledge(self):
        _, driver = self.run_session(enable=True)
        request = driver.requests["brief"][0]
        self.assertIn("parameter_knowledge", request)
        self.assertEqual(
            request["parameter_knowledge"]["knowledge_id"],
            knowledge_mod.load()["knowledge_id"],
        )

    def test_plan_request_carries_knowledge(self):
        """plan 分支的 decode_next 不返回 brief 全文，所以必须单独挂一份。"""
        _, driver = self.run_session(enable=True)
        self.assertTrue(driver.requests["plan"])
        for request in driver.requests["plan"]:
            self.assertIn("parameter_knowledge", request)

    def test_disabled_injects_nothing(self):
        _, driver = self.run_session(enable=False)
        self.assertNotIn("parameter_knowledge", driver.requests["brief"][0])
        for request in driver.requests["plan"]:
            self.assertNotIn("parameter_knowledge", request)

    def test_brief_records_reference_not_full_copy(self):
        """落盘的 brief 只存 id/hash/路径，不复制全文——否则报告里会出现两份会各自漂移的副本。"""
        _, driver = self.run_session(enable=True)
        session_id = driver.requests["brief"][0]["session_id"]
        brief_path = self.root / "sessions" / session_id / "creative_brief.json"
        self.assertTrue(brief_path.exists(), msg=f"没找到落盘 brief：{brief_path}")
        record = json.loads(brief_path.read_text(encoding="utf-8"))["parameter_knowledge"]
        self.assertEqual(
            sorted(record),
            ["full_ref", "hash_verified", "knowledge_hash", "knowledge_id", "layer", "scope"],
        )
        self.assertNotIn("params", record)

    def test_injection_does_not_change_numbers(self):
        """本层只提示不碰数值：开关前后整条哈希链必须逐项相同。"""

        def collect_hashes(obj) -> list:
            found = []
            if isinstance(obj, dict):
                for key, value in sorted(obj.items()):
                    if key.endswith("settings_hash") or key == "hash":
                        found.append((key, str(value)))
                    else:
                        found.extend(collect_hashes(value))
            elif isinstance(obj, list):
                for item in obj:
                    found.extend(collect_hashes(item))
            return found

        off, _ = self.run_session(enable=False)
        on, _ = self.run_session(enable=True)

        self.assertEqual(off.get("ok"), True)
        self.assertEqual(on.get("ok"), True)
        self.assertEqual(off.get("rounds_used"), on.get("rounds_used"))
        self.assertEqual(off.get("best_round"), on.get("best_round"))
        self.assertEqual(collect_hashes(off), collect_hashes(on))


if __name__ == "__main__":
    unittest.main()

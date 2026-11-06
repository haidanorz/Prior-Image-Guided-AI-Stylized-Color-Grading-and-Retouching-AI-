"""Golden 回放：守住"状态机改完，选轮和哈希链还一样"。

与 test_session 的区别：那些测试断言的是"我认为应该发生什么"，
这里断言的是"上一次真实跑出来是什么样"。以后改守卫、选轮、早停、
报告字段，只要输出变了就必须先看 diff 再决定重录，不能悄悄把基线改掉。

重录：python -m online_decode.tools.record_golden
比对：python -m online_decode.tools.record_golden --check
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from online_decode.config import Config
from online_decode.drivers.scripted import ScriptedDriver
from online_decode.feedback.lr_port import FakeLrPort
from online_decode.session import SessionController
from online_decode.tests.helpers import BASE_SETTINGS, SCENE
from online_decode.tools.record_golden import OUT, PACKAGE, digest

CASES = Path(__file__).resolve().parent / "golden" / "cases.json"


def run_case(case):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        photo = root / "IMG_0001.jpg"
        photo.write_bytes(b"\xff\xd8\xff\xd9")
        cfg = Config(max_rounds=case["max_rounds"], workspace=root)
        cfg.validate()
        port = FakeLrPort(BASE_SETTINGS, workspace=root, **case.get("port_flags") or {})
        driver = ScriptedDriver(scene=SCENE, 
            brief=case["brief"],
            plans=list(case["plans"]),
            evaluations=list(case["evaluations"]),
        )
        controller = SessionController(cfg, port, driver)
        controller.run(str(photo), str(PACKAGE))
        if controller.state is None:
            raise AssertionError("会话没有建立")
        return digest(controller.state, cfg)


class GoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(CASES.read_text(encoding="utf-8"))

    def test_golden_file_is_versioned(self):
        self.assertEqual(self.data["golden_schema_version"], "1.0.0")
        self.assertGreaterEqual(len(self.data["cases"]), 2)
        self.assertTrue(OUT.is_file())

    def test_every_case_replays_to_same_digest(self):
        for case in self.data["cases"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(run_case(case), case["expected"])

    def test_success_case_closes_hash_chain(self):
        case = next(item for item in self.data["cases"] if item["name"] == "two_rounds_goal_met")
        expected = case["expected"]
        self.assertTrue(expected["ok"])
        self.assertTrue(expected["final_hash_equals_best"])
        self.assertEqual(expected["best_round"], 2)
        self.assertEqual(expected["stop_reason"], "goal_met")
        for item in expected["rounds"]:
            self.assertEqual(item["settings_hash"], item["render_hash"])
            self.assertEqual(item["model_input_source"], "xmp")
            self.assertIsNone(item["error_code"])

    def test_failure_case_voids_round(self):
        case = next(item for item in self.data["cases"] if item["name"] == "render_hash_drift_voids_round")
        expected = case["expected"]
        self.assertFalse(expected["ok"])
        self.assertEqual(expected["rounds"][0]["error_code"], "RENDER_HASH_MISMATCH")
        self.assertFalse(expected["final_hash_equals_best"])


if __name__ == "__main__":
    unittest.main()

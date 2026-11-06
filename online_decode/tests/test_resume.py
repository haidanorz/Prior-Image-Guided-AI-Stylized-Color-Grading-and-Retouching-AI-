"""中断续跑测试。

最关键的一条：等模型超时**不能**被记成 failed，否则会话就永远不可恢复了。
其次：续跑只认"已渲染 + 已评价"的轮次，末尾半个轮次不能混进预算。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from online_decode.config import Config
from online_decode.drivers.scripted import ScriptedDriver
from online_decode.errors import Codes
from online_decode.feedback.lr_port import FakeLrPort
from online_decode.session import SessionController
from online_decode.state import SessionState
from online_decode.tests.helpers import BASE_SETTINGS, SCENE
from online_decode.tests.test_session import brief_response, evaluation, plan_response

PACKAGE = Path(__file__).resolve().parents[2] / "examples" / "sample_package"


class ResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.photo = self.root / "work" / "IMG_0001.jpg"
        self.photo.parent.mkdir(parents=True, exist_ok=True)
        self.photo.write_bytes(b"\xff\xd8\xff\xd9")
        self.port = FakeLrPort(BASE_SETTINGS, workspace=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def controller(self, driver, **cfg_kwargs):
        cfg = Config(workspace=self.root, **cfg_kwargs)
        cfg.validate()
        return SessionController(cfg, self.port, driver)

    def interrupted_session(self):
        """第 1 轮完整跑完，第 2 轮等模型时超时。"""
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(),
            plans=[],
            evaluations=[evaluation("better", "continue", style="partially_met")],
        )
        report = self.controller(driver).run(str(self.photo), str(PACKAGE))
        self.assertFalse(report["ok"])
        return report["session_id"]

    def test_timeout_keeps_session_resumable(self):
        session_id = self.interrupted_session()
        state = SessionState.load(self.root / "sessions" / session_id)
        self.assertNotIn(state.status, ("failed", "completed", "aborted"))
        self.assertEqual(state.substep, "awaiting_model")
        self.assertEqual(state.render_count, 1)
        self.assertTrue(any(err.get("resumable") for err in state.errors))

    def test_resume_continues_from_second_round(self):
        session_id = self.interrupted_session()
        root = self.root / "sessions" / session_id

        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(),  # 不会用到，brief 从文件读
            plans=[plan_response(-11)],
            evaluations=[evaluation("better", "stop_and_select", style="met")],
        )
        report = self.controller(driver).resume(root)

        self.assertTrue(report["ok"], msg=json.dumps(report.get("errors"), ensure_ascii=False))
        self.assertEqual(report["rounds_used"], 2)
        self.assertEqual(report["best"]["round"], 2)
        self.assertEqual(report["stop_reason"], "goal_met")
        self.assertTrue(Path(report["final"]["path"]).is_file())
        self.assertFalse(any(err.get("resumable") for err in report["errors"]))
        self.assertTrue(any("恢复" in item for item in report["warnings"]))

    def test_resume_does_not_reuse_previous_render(self):
        session_id = self.interrupted_session()
        root = self.root / "sessions" / session_id
        before = [call for call in self.port.calls]
        first_render_count = before.count("render_preview")

        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(), plans=[plan_response(-11)],
            evaluations=[evaluation("better", "stop_and_select", style="met")],
        )
        self.controller(driver).resume(root)
        added = self.port.calls[len(before):]
        self.assertIn("render_preview", added)
        self.assertGreaterEqual(first_render_count, 2)  # 基线 1 次 + 第 1 轮 1 次

    def test_resume_drops_incomplete_trailing_round(self):
        session_id = self.interrupted_session()
        root = self.root / "sessions" / session_id
        state = SessionState.load(root)
        state.rounds.append({"round": 2, "ok": False, "apply": {"settings_hash": "sha256:x"}})
        state.set_status("applying", "round2")
        state.save()

        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(), plans=[plan_response(-11)],
            evaluations=[evaluation("better", "stop_and_select", style="met")],
        )
        report = self.controller(driver).resume(root)
        self.assertTrue(report["ok"])
        self.assertEqual(report["rounds_used"], 2)
        self.assertTrue(any("丢弃 1 个未完成轮次" in item for item in report["warnings"]))

    def test_resume_rejects_completed_session(self):
        driver = ScriptedDriver(scene=SCENE, 
            brief=brief_response(), plans=[],
            evaluations=[evaluation("better", "stop_and_select")],
        )
        first = self.controller(driver, max_rounds=1).run(str(self.photo), str(PACKAGE))
        self.assertTrue(first["ok"])
        root = self.root / "sessions" / first["session_id"]
        second = self.controller(ScriptedDriver(scene=SCENE, brief=brief_response())).resume(root)
        self.assertIn(Codes.SESSION_RESUME_INVALID, json.dumps(second.get("error") or {}, ensure_ascii=False))

    def test_resume_missing_session(self):
        driver = ScriptedDriver(scene=SCENE, brief=brief_response())
        result = self.controller(driver).resume(self.root / "sessions" / "nope")
        self.assertIn(Codes.SESSION_RESUME_INVALID, json.dumps(result.get("error") or {}, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()

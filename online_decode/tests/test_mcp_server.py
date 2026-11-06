"""MCP 工具层端到端测试：不依赖 Lightroom，用 FakeLrPort 注入。

验证的是"完整算法能否通过工具链跑通"：start → next → submit 循环 → report，
以及 key 校验、abort、契约查询。控制器一行没改，只是换了 Driver。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from online_decode import mcp_server as srv
from online_decode.config import Config
from online_decode.feedback.lr_port import FakeLrPort
from online_decode.params import xmp_renderer
from online_decode.tests.helpers import BASE_SETTINGS, SCENE
from online_decode.tests.test_session import brief_response, evaluation, plan_response

PACKAGE = str(Path(__file__).resolve().parents[2] / "examples" / "sample_package")


def _parse(text: str) -> dict:
    return json.loads(text)


def _answer(task: str, round_no: int, packet: dict) -> dict:
    if task == "scene":
        return dict(SCENE)
    if task == "brief":
        return brief_response()
    if task == "plan":
        return plan_response(-11)
    table = {
        1: evaluation("better", "continue", style="partially_met"),
        2: evaluation("better", "continue", style="met"),
        3: evaluation("same", "restore_previous_best", style="met"),
    }
    return table[round_no]


class McpFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.photo = self.root / "IMG_0001.jpg"
        self.photo.write_bytes(b"\xff\xd8\xff\xd9")
        srv.PORT_FACTORY = lambda cfg: FakeLrPort(BASE_SETTINGS, workspace=cfg.workspace)
        srv.SESSIONS.clear()

    def tearDown(self) -> None:
        # 后台线程可能还在写会话目录；先中止并等它退出，再删临时目录
        for handle in list(srv.SESSIONS.values()):
            try:
                handle.driver.fail_all("测试结束")
                if handle.thread and handle.thread.is_alive():
                    handle.thread.join(timeout=3)
            except Exception:
                pass
        srv.PORT_FACTORY = None
        srv.SESSIONS.clear()
        self.tmp.cleanup()

    def _drive(self, session_id: str, max_steps: int = 40) -> dict:
        report = None
        for _ in range(max_steps):
            # 短超时 + 重试：最后一轮提交后没有新任务，不能白等 20 秒
            payload = _parse(srv.decode_next(session_id, timeout=1.0))
            if not payload.get("pending"):
                status = _parse(srv.decode_status(session_id))
                if status.get("report_ready"):
                    report = _parse(srv.decode_report(session_id))
                    break
                if status.get("driver_done"):
                    self.fail(f"线程结束但没有报告：{status}")
                continue
            response = _answer(payload["task"], payload["round"], payload)
            submitted = _parse(srv.decode_submit(session_id, payload["key"], response))
            self.assertTrue(submitted["ok"], msg=submitted)
        self.assertIsNotNone(report, "工具链没有在限定步数内产出报告")
        return report

    def test_full_algorithm_runs_through_tools(self):
        started = _parse(srv.decode_start(str(self.photo), PACKAGE,
                                          workspace=str(self.root), max_rounds=3))
        self.assertTrue(started["ok"], msg=started)
        session_id = started["session_id"]

        report = self._drive(session_id)
        self.assertTrue(report["ok"], msg=json.dumps(report.get("errors"), ensure_ascii=False))
        self.assertEqual(report["rounds_used"], 3)
        self.assertEqual(report["best"]["round"], 2)
        self.assertEqual(report["stop_reason"], "model_restore")
        self.assertTrue(Path(report["final"]["path"]).is_file())
        for item in report["rounds"]:
            self.assertEqual(item["settings_hash"], item["render"]["settings_hash"])
            self.assertEqual(item["model_input"]["source"], "xmp")

    def test_single_round_session(self):
        started = _parse(srv.decode_start(str(self.photo), PACKAGE,
                                          workspace=str(self.root), max_rounds=1))
        session_id = started["session_id"]
        # 覆盖单轮：第 1 轮直接停止
        original = _answer
        def one_round(task, round_no, packet):
            if task == "evaluate":
                return evaluation("better", "stop_and_select")
            return original(task, round_no, packet)
        import online_decode.tests.test_mcp_server as self_module
        self_module._answer = one_round
        try:
            report = self._drive(session_id)
        finally:
            self_module._answer = original
        self.assertTrue(report["ok"])
        self.assertEqual(report["rounds_used"], 1)

    def test_submit_with_wrong_key_is_rejected(self):
        started = _parse(srv.decode_start(str(self.photo), PACKAGE,
                                          workspace=str(self.root), max_rounds=1))
        session_id = started["session_id"]
        packet = _parse(srv.decode_next(session_id, timeout=20.0))
        self.assertTrue(packet["pending"])
        bad = _parse(srv.decode_submit(session_id, "bogus_key", {"x": 1}))
        self.assertFalse(bad["ok"])
        self.assertIn(packet["key"], bad["pending_keys"])
        srv.decode_abort(session_id)

    def test_default_rounds_is_five(self):
        started = _parse(srv.decode_start(str(self.photo), PACKAGE, workspace=str(self.root)))
        self.assertTrue(started["ok"], msg=started)
        self.assertEqual(started["max_rounds"], 5)
        session_id = started["session_id"]
        import time
        deadline = time.monotonic() + 3
        status = {}
        while time.monotonic() < deadline:
            status = _parse(srv.decode_status(session_id))
            if status.get("max_rounds"):
                break
            time.sleep(0.05)
        self.assertEqual(status["max_rounds"], 5)
        srv.decode_abort(session_id)

    def test_contract_and_fields_tools(self):
        for task in ("brief", "plan", "evaluate"):
            payload = _parse(srv.decode_contract(task))
            self.assertTrue(payload["ok"])
            self.assertIn("$schema", payload["schema"])
        fields = _parse(srv.decode_fields())
        self.assertTrue(fields["ok"])
        self.assertEqual(fields["count"], 65)
        self.assertIn("Highlights2012", fields["fields"])

    def test_verify_package_tool(self):
        payload = _parse(srv.decode_verify_package(PACKAGE))
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["fingerprint_id"].startswith("fp_"))

    def test_sessions_and_status_tools(self):
        started = _parse(srv.decode_start(str(self.photo), PACKAGE,
                                          workspace=str(self.root), max_rounds=1))
        session_id = started["session_id"]
        listing = _parse(srv.decode_sessions())
        self.assertEqual(listing["count"], 1)
        import time
        deadline = time.monotonic() + 3
        status = {}
        while time.monotonic() < deadline:
            status = _parse(srv.decode_status(session_id))
            if "status" in status:
                break
            time.sleep(0.05)
        self.assertIn("status", status)
        self.assertEqual(status["max_rounds"], 1)
        srv.decode_abort(session_id)
        self.assertTrue(_parse(srv.decode_status(session_id))["aborted"])

    def test_unknown_session_is_reported(self):
        payload = _parse(srv.decode_status("s_missing"))
        self.assertFalse(payload["ok"])


if __name__ == "__main__":
    unittest.main()

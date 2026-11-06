from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from online_decode.drivers.agent_driver import AgentDriver
from online_decode.errors import Codes
from online_decode.ports import DriverError


class AgentDriverTest(unittest.TestCase):
    def test_exchange_with_host_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            driver = AgentDriver(tmp, timeout_sec=5, poll_sec=0.02)

            def fake_agent():
                io = Path(tmp) / "sessions" / "s_1" / "agent_io"
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    req = io / "req_brief_00.json"
                    if req.is_file():
                        payload = json.loads(req.read_text(encoding="utf-8"))
                        self.assertEqual(payload["task"], "brief")
                        (io / "resp_brief_00.json").write_text(
                            json.dumps({"branch_rationale": "ok"}), encoding="utf-8"
                        )
                        return
                    time.sleep(0.01)

            thread = threading.Thread(target=fake_agent, daemon=True)
            thread.start()
            response = driver.brief({"session_id": "s_1", "round": 0, "task": "brief"})
            self.assertEqual(response["branch_rationale"], "ok")

    def test_timeout_is_reported_as_retryable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            driver = AgentDriver(tmp, timeout_sec=0.2, poll_sec=0.02)
            with self.assertRaises(DriverError) as ctx:
                driver.plan({"session_id": "s_2", "round": 1, "task": "plan"})
            self.assertEqual(ctx.exception.code, Codes.MODEL_RESPONSE_TIMEOUT)
            self.assertIn("req_plan_01.json", ctx.exception.message)

    def test_stale_response_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            driver = AgentDriver(tmp, timeout_sec=0.3, poll_sec=0.02)
            io = Path(tmp) / "sessions" / "s_3" / "agent_io"
            io.mkdir(parents=True)
            (io / "resp_plan_01.json").write_text('{"adjustments": []}', encoding="utf-8")
            with self.assertRaises(DriverError):
                driver.plan({"session_id": "s_3", "round": 1, "task": "plan"})


if __name__ == "__main__":
    unittest.main()

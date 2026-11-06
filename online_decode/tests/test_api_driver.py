"""ApiDriver 与一键工具的测试（本地假 HTTP 服务，不联网、不花钱）。"""

from __future__ import annotations

import json
import queue
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from online_decode import mcp_server as srv
from online_decode.drivers.api_driver import ApiDriver, _parse_json_object, is_configured
from online_decode.feedback.lr_port import FakeLrPort
from online_decode.ports import DriverError
from online_decode.tests.helpers import BASE_SETTINGS, SCENE
from online_decode.tests.test_session import brief_response, evaluation

PACKAGE = str(Path(__file__).resolve().parents[2] / "examples" / "sample_package")


class SystemPromptTest(unittest.TestCase):
    def test_prompt_covers_scene_and_prior_rules(self):
        from online_decode.drivers import api_driver

        prompt = api_driver.SYSTEM_PROMPT
        self.assertIn("scene / brief / plan / evaluate", prompt)
        self.assertIn("不要输出 XMP", prompt)
        self.assertIn("priors 不是锚点", prompt)
        self.assertIn("not_evaluable", prompt)


class ParseTest(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(_parse_json_object('{"a":1}'), {"a": 1})

    def test_code_fence(self):
        self.assertEqual(_parse_json_object('```json\n{"a": 2}\n```'), {"a": 2})

    def test_non_object_rejected(self):
        self.assertIsNone(_parse_json_object("[1,2]"))
        self.assertIsNone(_parse_json_object("not json"))


class ApiDriverTest(unittest.TestCase):
    def test_missing_key_fails_loudly(self):
        import os

        old = os.environ.pop("ONLINE_DECODE_VISION_KEY", None)
        try:
            with self.assertRaises(DriverError) as ctx:
                ApiDriver(key="")
            self.assertIn("未配置", ctx.exception.message)
        finally:
            if old is not None:
                os.environ["ONLINE_DECODE_VISION_KEY"] = old

    def test_roundtrip_with_mock_http_server(self):
        answers: queue.Queue = queue.Queue()
        answers.put(brief_response())
        answers.put(evaluation("better", "stop_and_select"))

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = body["messages"]
                assert messages[0]["role"] == "system"
                assert messages[1]["content"][0]["type"] == "text"
                payload = json.dumps({"choices": [{"message": {"content": json.dumps(answers.get())}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            driver = ApiDriver(url=f"http://127.0.0.1:{server.server_port}/v1/chat/completions", key="test-key")
            brief = driver.brief({"task": "brief", "images": {"original": None}, "round": 0})
            self.assertIn("initial_xmp", brief)
            ev = driver.evaluate({"task": "evaluate", "images": {"current": None}, "round": 1})
            self.assertEqual(ev["improvement"], "better")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class DecodeRunToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.photo = self.root / "IMG_0001.jpg"
        self.photo.write_bytes(b"\xff\xd8\xff\xd9")
        srv.PORT_FACTORY = lambda cfg: FakeLrPort(BASE_SETTINGS, workspace=cfg.workspace)
        srv.API_DRIVER_FACTORY = None

    def tearDown(self) -> None:
        srv.PORT_FACTORY = None
        srv.API_DRIVER_FACTORY = None
        self.tmp.cleanup()

    def test_without_key_explains_sampling_limit(self):
        import os

        old = os.environ.pop("ONLINE_DECODE_VISION_KEY", None)
        srv.API_DRIVER_FACTORY = None
        try:
            payload = json.loads(srv.decode_run(str(self.photo), PACKAGE, workspace=str(self.root)))
            self.assertFalse(payload["ok"])
            self.assertIn("sampling", payload["reason"])
            self.assertIn("decode_start", payload["alternative"])
        finally:
            if old is not None:
                os.environ["ONLINE_DECODE_VISION_KEY"] = old

    def test_one_click_run_with_injected_driver(self):
        from online_decode.drivers.scripted import ScriptedDriver

        srv.API_DRIVER_FACTORY = lambda: ScriptedDriver(scene=SCENE, 
            brief=brief_response(),
            plans=[],
            evaluations=[evaluation("better", "stop_and_select")],
        )
        payload = json.loads(srv.decode_run(str(self.photo), PACKAGE,
                                            workspace=str(self.root), max_rounds=1))
        self.assertTrue(payload["ok"], msg=json.dumps(payload.get("errors"), ensure_ascii=False))
        self.assertEqual(payload["rounds_used"], 1)
        self.assertEqual(payload["config"]["driver"], "api")
        self.assertTrue(Path(payload["final"]["path"]).is_file())


if __name__ == "__main__":
    unittest.main()

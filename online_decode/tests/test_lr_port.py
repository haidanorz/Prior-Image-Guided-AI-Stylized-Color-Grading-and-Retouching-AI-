"""端口层字段兼容测试。

控制器直连队列读的是**插件原始结果**，不是 MCP summarize 后的形态。
这里固定住两种键名，防止以后只改了一侧导致基线哈希读不到。
"""

from __future__ import annotations

import unittest

from online_decode.feedback.lr_port import Outcome, settings_hash

HASH = "sha256:" + "a" * 64


class OutcomeTest(unittest.TestCase):
    def test_plugin_raw_get_settings_uses_snapshot(self):
        out = Outcome(ok=True, action="get_settings", entry={
            "ok": True,
            "snapshot": {"snapshot_id": "snap-1", "settings_hash": HASH, "complete": True},
            "settings": {"Exposure2012": 0.0},
        })
        self.assertEqual(out.settings_hash, HASH)
        self.assertEqual(out.snapshot["snapshot_id"], "snap-1")
        self.assertTrue(out.snapshot["complete"])

    def test_mcp_summarized_get_settings_uses_settings_snapshot(self):
        out = Outcome(ok=True, action="get_settings", entry={
            "ok": True,
            "settings_snapshot": {"snapshot_id": "snap-2", "settings_hash": HASH},
        })
        self.assertEqual(out.settings_hash, HASH)

    def test_render_hash_is_found(self):
        out = Outcome(ok=True, action="render_preview", entry={"ok": True, "render": {"settings_hash": HASH}})
        self.assertEqual(out.settings_hash, HASH)

    def test_apply_top_level_hash_wins(self):
        out = Outcome(ok=True, action="apply", entry={"ok": True, "settings_hash": HASH})
        self.assertEqual(out.settings_hash, HASH)

    def test_missing_hash_is_none(self):
        self.assertIsNone(Outcome(ok=True, action="get_settings", entry={"ok": True}).settings_hash)


class HashCompatTest(unittest.TestCase):
    def test_quantization_ignores_readback_noise(self):
        base = {"Exposure2012": 0.35, "Clarity2012": 9, "WhiteBalance": "Custom"}
        noisy = {"WhiteBalance": "Custom", "Clarity2012": 9.00001, "Exposure2012": 0.350001}
        self.assertEqual(settings_hash(base), settings_hash(noisy))

    def test_real_change_changes_hash(self):
        self.assertNotEqual(
            settings_hash({"Clarity2012": 9}),
            settings_hash({"Clarity2012": 10}),
        )


if __name__ == "__main__":
    unittest.main()

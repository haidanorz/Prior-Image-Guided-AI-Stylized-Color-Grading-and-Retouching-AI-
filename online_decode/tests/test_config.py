from __future__ import annotations

import unittest

from online_decode.config import Config, load
from online_decode.errors import Codes, DecodeError


class ConfigTest(unittest.TestCase):
    def test_default_five_rounds(self):
        cfg = Config()
        cfg.validate()
        self.assertEqual(cfg.max_rounds, 5)
        # n>=4 自动关闭单轮早停，否则 5 轮会被压成 2 轮
        self.assertFalse(cfg.effective_stop_on_single_no_progress)
        self.assertEqual(cfg.stop_source, "derived")
        self.assertEqual(cfg.to_report()["max_rounds"], 5)

    def test_short_budget_still_uses_single_no_progress(self):
        self.assertTrue(Config(max_rounds=3).effective_stop_on_single_no_progress)

    def test_stop_rule_derived_from_n(self):
        small = Config(max_rounds=2)
        large = Config(max_rounds=5)
        self.assertTrue(small.effective_stop_on_single_no_progress)
        self.assertFalse(large.effective_stop_on_single_no_progress)
        explicit = Config(max_rounds=5, stop_on_single_no_progress=True)
        self.assertTrue(explicit.effective_stop_on_single_no_progress)
        self.assertEqual(explicit.stop_source, "explicit")

    def test_out_of_range_fails_loudly(self):
        for value in (0, 6, -1):
            with self.assertRaises(DecodeError) as ctx:
                load(None, {"max_rounds": value})
            self.assertEqual(ctx.exception.code, Codes.CONFIG_INVALID)

    def test_unknown_key_rejected(self):
        with self.assertRaises(DecodeError):
            load(None, {"max_round": 3})  # 少个 s，不能静默忽略

    def test_report_carries_effective_values(self):
        payload = Config(max_rounds=4).to_report()
        self.assertEqual(payload["max_rounds"], 4)
        self.assertFalse(payload["stop_on_single_no_progress"])
        self.assertEqual(payload["stop_on_single_no_progress_source"], "derived")


if __name__ == "__main__":
    unittest.main()

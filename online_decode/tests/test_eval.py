from __future__ import annotations

import unittest

from online_decode.eval import selector, verify_evidence


def record(round_no, improvement="better", style="met", defects=None, ok=True):
    return {
        "round": round_no,
        "ok": ok,
        "evaluation": {
            "improvement": improvement,
            "decision": "continue",
            "style_alignment": {"verdict": style},
            "technical_quality": {"defects": defects or []},
            "preserve_violations": [],
            "caution_violations": [],
        },
    }


class SelectorTest(unittest.TestCase):
    def test_prefers_later_round_only_when_better(self):
        rounds = [record(1, style="partially_met"), record(2, style="met")]
        self.assertEqual(selector.pick_best(rounds)["round"], 2)

    def test_tie_goes_to_earlier_round(self):
        # 无递进（两轮都判 same）时，仍按"更早更保守"选早轮
        rounds = [record(1, improvement="same"), record(2, improvement="same")]
        self.assertEqual(selector.pick_best(rounds)["round"], 1)

    def test_chain_better_prefers_later_round(self):
        """improvement=better 是"比上一轮更好"的链式语义：两轮都 better 时选更晚的。

        真机复现（DSC07562）：第 2 轮修正了第 1 轮的天空过深，两轮同为 met+better，
        旧排序因"更早优先"把第 2 轮的真实改善吞掉。
        """
        rounds = [record(1, improvement="better"), record(2, improvement="better")]
        self.assertEqual(selector.pick_best(rounds)["round"], 2)

    def test_hard_failure_never_wins(self):
        rounds = [record(1, ok=False), record(2)]
        self.assertEqual(selector.pick_best(rounds)["round"], 2)

    def test_none_when_every_round_failed(self):
        self.assertIsNone(selector.pick_best([record(1, ok=False)]))

    def test_stop_on_budget(self):
        stop, reason = selector.should_stop([record(1)], max_rounds=1,
                                            stop_on_single_no_progress=True, has_best=True)
        self.assertTrue(stop)
        self.assertEqual(reason, "budget")

    def test_single_round_no_progress_needs_second_round(self):
        stop, _ = selector.should_stop([record(1, improvement="same")], max_rounds=3,
                                       stop_on_single_no_progress=True, has_best=True)
        self.assertFalse(stop, "首轮判无改善就停会把会话压成一轮")
        stop2, reason = selector.should_stop(
            [record(1), record(2, improvement="same")],
            max_rounds=5, stop_on_single_no_progress=True, has_best=True,
        )
        self.assertTrue(stop2)
        self.assertEqual(reason, "no_progress")

    def test_regression_stops_immediately(self):
        stop, reason = selector.should_stop([record(1), record(2, improvement="worse")],
                                            max_rounds=3, stop_on_single_no_progress=False,
                                            has_best=True)
        self.assertTrue(stop)
        self.assertEqual(reason, "regression")

    def test_five_round_budget_does_not_early_stop(self):
        stop, _ = selector.should_stop([record(1), record(2, improvement="same")],
                                       max_rounds=5, stop_on_single_no_progress=False,
                                       has_best=True)
        self.assertFalse(stop)

    def test_two_consecutive_no_progress_stops_long_budget(self):
        """默认 n=5 档：连续两轮无改善必须停，不能一路跑到预算耗尽。"""
        stop, reason = selector.should_stop(
            [record(1), record(2, improvement="same"), record(3, improvement="same")],
            max_rounds=5, stop_on_single_no_progress=False, has_best=True,
        )
        self.assertTrue(stop)
        self.assertEqual(reason, "no_progress")

    def test_alternating_progress_keeps_long_budget(self):
        stop, _ = selector.should_stop(
            [record(1), record(2, improvement="same"), record(3, improvement="better")],
            max_rounds=5, stop_on_single_no_progress=False, has_best=True,
        )
        self.assertFalse(stop)


class EvidenceTest(unittest.TestCase):
    metrics = {
        "clip_high_ratio": 0.004,
        "luma": {"p50": 0.42, "p95": 0.88},
    }

    def base(self):
        return {
            "style_alignment": {"verdict": "met"},
            "branch_alignment": {"verdict": "met"},
            "technical_quality": {"verdict": "met", "defects": []},
            "content_preservation": {"verdict": "met"},
            "improvement": "better",
            "decision": "continue",
            "evidence": [{"anchor": "upper_right", "metric": "luma.p50", "value": 0.42,
                          "observation": "天空亮度保持在中间调"}],
        }

    def test_accepts_verified_evidence(self):
        ok, problems = verify_evidence.verify(self.base(), self.metrics)
        self.assertTrue(ok, msg=problems)

    def test_rejects_metric_that_does_not_exist(self):
        payload = self.base()
        payload["evidence"][0]["metric"] = "does.not.exist"
        ok, problems = verify_evidence.verify(payload, self.metrics)
        self.assertFalse(ok)
        self.assertTrue(any("不存在" in item for item in problems))

    def test_rejects_value_mismatch(self):
        payload = self.base()
        payload["evidence"][0]["value"] = 0.9
        ok, problems = verify_evidence.verify(payload, self.metrics)
        self.assertFalse(ok)
        self.assertTrue(any("不符" in item for item in problems))

    def test_rejects_bad_anchor(self):
        payload = self.base()
        payload["evidence"][0]["anchor"] = "左上角"
        ok, problems = verify_evidence.verify(payload, self.metrics)
        self.assertFalse(ok)

    def test_rejects_empty_evidence(self):
        payload = self.base()
        payload["evidence"] = []
        ok, problems = verify_evidence.verify(payload, self.metrics)
        self.assertFalse(ok)

    def test_rejects_illegal_improvement(self):
        payload = self.base()
        payload["improvement"] = "much better"
        ok, problems = verify_evidence.verify(payload, self.metrics)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from online_decode.params import xmp_renderer
from online_decode.params.fields import Fields
from online_decode.params.guard import Guard

BASELINE = {"settings": {"Temperature": 5500, "Tint": 8}}


class GuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fields = Fields.load()
        cls.guard = Guard(cls.fields)

    def build(self, adjustments, *, round_no=1, max_rounds=3, references=None,
              reference_spans=None, prev_values=None, prev_step=None, strength=1.0):
        return self.guard.build_ledger(
            adjustments,
            round_no=round_no,
            max_rounds=max_rounds,
            baseline=BASELINE,
            references={"Clarity2012": 9} if references is None else references,
            reference_spans=reference_spans,
            prev_values=prev_values,
            prev_step=prev_step,
            strength=strength,
        )

    def test_accepts_reference_backed_value(self):
        ledger = self.build([
            {"key": "clarity", "value": 12, "evidence_level": "runtime_reference",
             "ref": "fingerprint.global.clarity", "reason": "参考"},
        ])
        self.assertEqual(ledger.values, {"Clarity2012": 12})
        self.assertFalse(ledger.rejected)

    def test_anchors_temperature_shift_on_baseline(self):
        ledger = self.build([
            {"key": "temperature_shift", "value": 350, "evidence_level": "experimental_prior",
             "ref": "insufficient_evidence.temperature_shift", "reason": "偏暖"},
        ])
        self.assertEqual(ledger.values["Temperature"], 5850)
        self.assertEqual(ledger.values["WhiteBalance"], "Custom")
        self.assertIn("WhiteBalance=Custom", ledger.auto_added)

    def test_shift_without_baseline_is_rejected(self):
        guard = Guard(self.fields)
        ledger = guard.build_ledger(
            [{"key": "temperature_shift", "value": 350, "evidence_level": "experimental_prior",
              "ref": "x", "reason": "偏暖"}],
            round_no=1, max_rounds=3, baseline={"settings": {}},
        )
        self.assertTrue(ledger.is_empty)
        self.assertEqual(ledger.rejected[0].code, "BASELINE_MISSING")

    def test_ext_field_rejected(self):
        ledger = self.build([
            {"key": "grain_amount", "value": 20, "evidence_level": "planner_judgement",
             "ref": "narrative", "reason": "胶片感"},
        ])
        self.assertTrue(ledger.is_empty)
        self.assertEqual(ledger.rejected[0].code, "FIELD_UNSUPPORTED")

    def test_unknown_field_rejected(self):
        ledger = self.build([
            {"key": "hue_adjust_red", "value": 5, "evidence_level": "planner_judgement",
             "ref": "narrative", "reason": "错名"},
        ])
        self.assertEqual(ledger.rejected[0].code, "FIELD_UNKNOWN")

    def test_alias_resolves_to_canonical(self):
        ledger = self.build([
            {"key": "ColorGradeShadowHue", "value": 210, "evidence_level": "planner_judgement",
             "ref": "narrative", "reason": "冷阴影"},
        ])
        self.assertIn("SplitToningShadowHue", ledger.values)

    def test_out_of_range_is_clamped_with_note(self):
        ledger = self.build([
            {"key": "clarity", "value": 400, "evidence_level": "planner_judgement",
             "ref": "narrative", "reason": "夸张"},
        ], references={})
        self.assertEqual(ledger.values["Clarity2012"], 100)
        self.assertTrue(any("钳制" in note for note in ledger.entries[0].notes))

    def test_band_check_rejects_far_from_reference(self):
        ledger = self.build([
            {"key": "clarity", "value": 60, "evidence_level": "planner_judgement",
             "ref": "narrative", "reason": "远超参考"},
        ])
        self.assertTrue(ledger.is_empty)
        self.assertEqual(ledger.rejected[0].code, "OUT_OF_BAND")

    def test_span_widens_reference_band(self):
        entry = {
            "key": "clarity", "value": 35, "evidence_level": "planner_judgement",
            "ref": "narrative", "reason": "按桶内分散度放宽",
        }
        strict = self.build([dict(entry)])
        self.assertTrue(strict.is_empty)
        self.assertEqual(strict.rejected[0].code, "OUT_OF_BAND")

        widened = self.build([dict(entry)], reference_spans={"Clarity2012": 20})
        self.assertEqual(widened.values["Clarity2012"], 35)
        self.assertFalse(widened.rejected)

    def test_band_only_applies_to_first_two_rounds(self):
        entry = {
            "key": "clarity", "value": 60, "evidence_level": "planner_judgement",
            "ref": "narrative", "reason": "远超参考",
        }
        for round_no in (1, 2):
            ledger = self.build([dict(entry)], round_no=round_no, max_rounds=5)
            self.assertTrue(ledger.is_empty, msg=f"第 {round_no} 轮应被带宽拒绝")
            self.assertEqual(ledger.rejected[0].code, "OUT_OF_BAND")

        ledger = self.build([dict(entry)], round_no=3, max_rounds=5)
        self.assertEqual(ledger.values["Clarity2012"], 60)
        self.assertTrue(any("带宽已" in note for note in ledger.entries[0].notes))

    def test_span_term_is_capped(self):
        entry = {
            "key": "clarity", "value": 60, "evidence_level": "planner_judgement",
            "ref": "narrative", "reason": "极端 span 不应无限放宽",
        }
        ledger = self.build([dict(entry)], reference_spans={"Clarity2012": 100})
        self.assertTrue(ledger.is_empty)
        self.assertEqual(ledger.rejected[0].code, "OUT_OF_BAND")

    def test_planner_judgement_needs_ref(self):
        ledger = self.build([
            {"key": "clarity", "value": 10, "evidence_level": "planner_judgement", "reason": "无出处"},
        ])
        self.assertEqual(ledger.rejected[0].code, "REF_MISSING")

    def test_narrative_only_relabelled(self):
        ledger = self.build([
            {"key": "clarity", "value": 10, "evidence_level": "narrative_only",
             "ref": "narrative.summary", "reason": "叙事推断"},
        ])
        self.assertEqual(ledger.entries[0].evidence_level, "planner_judgement")
        self.assertTrue(ledger.warnings)

    def test_last_round_damping(self):
        ledger = self.build([
            {"key": "clarity", "value": 40, "evidence_level": "planner_judgement",
             "ref": "narrative", "reason": "末轮想大改"},
        ], round_no=3, max_rounds=3, references={},
            prev_values={"Clarity2012": 10}, prev_step={"Clarity2012": 4})
        self.assertLessEqual(float(ledger.values["Clarity2012"]), 20)
        self.assertTrue(any("末轮阻尼" in note for note in ledger.entries[0].notes))

    def test_grayscale_kills_color_channels(self):
        ledger = self.build([
            {"key": "ConvertToGrayscale", "value": True, "evidence_level": "planner_judgement",
             "ref": "narrative", "reason": "黑白"},
            {"key": "Vibrance", "value": 20, "evidence_level": "planner_judgement",
             "ref": "narrative", "reason": "加饱和"},
        ], references={})
        self.assertIn("ConvertToGrayscale", ledger.values)
        self.assertNotIn("Vibrance", ledger.values)
        self.assertEqual(ledger.rejected[0].code, "CONFLICT_GRAYSCALE")


class XmpTest(unittest.TestCase):
    def test_roundtrip_escapes_and_numbers(self):
        text = xmp_renderer.render({"Clarity2012": 9, "Temperature": 5850.0,
                                    "WhiteBalance": "Custom", "ConvertToGrayscale": False})
        self.assertIn('crs:Clarity2012="9"', text)
        self.assertIn('crs:Temperature="5850"', text)
        self.assertIn('crs:WhiteBalance="Custom"', text)
        self.assertIn('crs:ConvertToGrayscale="False"', text)
        self.assertTrue(text.startswith("<?xpacket"))
        self.assertTrue(text.rstrip().endswith("<?xpacket end=\"w\"?>"))

    def test_rejects_composite_values(self):
        with self.assertRaises(ValueError):
            xmp_renderer.render({"ToneCurvePV2012": [0, 0, 255, 255]})


if __name__ == "__main__":
    unittest.main()

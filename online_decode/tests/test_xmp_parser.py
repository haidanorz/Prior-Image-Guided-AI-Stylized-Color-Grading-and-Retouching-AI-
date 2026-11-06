from __future__ import annotations

import unittest

from online_decode.params import xmp_parser, xmp_renderer
from online_decode.params.fields import Fields
from online_decode.params.guard import Guard


def raw(body: str) -> str:
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"\n'
        f"{body}"
        "  />\n </rdf:RDF>\n</x:xmpmeta>\n"
    )


class ParserTest(unittest.TestCase):
    def test_reads_scalar_attributes(self):
        parsed = xmp_parser.parse(xmp_renderer.render({"Exposure2012": 0.35, "Contrast2012": 12}))
        self.assertTrue(parsed.ok, msg=parsed.issues)
        self.assertAlmostEqual(parsed.values["Exposure2012"], 0.35, places=4)
        self.assertEqual(parsed.values["Contrast2012"], 12)

    def test_coerces_signs_and_bools(self):
        parsed = xmp_parser.parse(raw(
            '   crs:Exposure2012="+0.10"\n'
            '   crs:Highlights2012="-45"\n'
            '   crs:ConvertToGrayscale="True"\n'
            '   crs:WhiteBalance="Custom"\n'
        ))
        self.assertEqual(parsed.values["Exposure2012"], 0.1)
        self.assertEqual(parsed.values["Highlights2012"], -45)
        self.assertIs(parsed.values["ConvertToGrayscale"], True)
        self.assertEqual(parsed.values["WhiteBalance"], "Custom")

    def test_metadata_is_separated(self):
        parsed = xmp_parser.parse(raw(
            '   crs:PresetType="Normal"\n'
            '   crs:Version="15.0"\n'
            '   crs:Clarity2012="9"\n'
        ))
        self.assertNotIn("PresetType", parsed.values)
        self.assertIn("PresetType", parsed.metadata)
        self.assertEqual(parsed.values, {"Clarity2012": 9})

    def test_duplicates_are_reported(self):
        parsed = xmp_parser.parse(raw(
            '   crs:Clarity2012="9"\n'
            '   crs:Clarity2012="12"\n'
        ))
        self.assertEqual(parsed.duplicates, ["Clarity2012"])
        self.assertEqual(parsed.values["Clarity2012"], 12)

    def test_sequence_is_rejected(self):
        text = raw('   crs:Clarity2012="9"\n') + "<rdf:Seq><rdf:li>0</rdf:li></rdf:Seq>"
        parsed = xmp_parser.parse(text)
        self.assertFalse(parsed.ok)
        self.assertTrue(any("rdf:Seq" in item for item in parsed.issues))

    def test_non_xmp_is_rejected(self):
        parsed = xmp_parser.parse('{"Clarity2012": 9}')
        self.assertFalse(parsed.ok)
        self.assertTrue(parsed.issues)


class GuardFromXmpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fields = Fields.load()
        cls.guard = Guard(cls.fields)

    def build(self, body, notes=None):
        parsed = xmp_parser.parse(raw(body))
        return self.guard.build_ledger_from_xmp(
            parsed.values,
            notes=notes,
            round_no=1,
            max_rounds=3,
            baseline={"settings": {"Temperature": 5500, "Tint": 8}},
        )

    def test_unknown_key_rejected_but_valid_key_applies(self):
        ledger = self.build(
            '   crs:Clarity2012="9"\n'
            '   crs:MysterySlider="42"\n'
        )
        self.assertEqual(ledger.values["Clarity2012"], 9)
        self.assertEqual(ledger.rejected[0].given_key, "MysterySlider")
        self.assertEqual(ledger.rejected[0].code, "FIELD_UNKNOWN")

    def test_ext_field_rejected(self):
        ledger = self.build('   crs:GrainAmount="20"\n')
        self.assertTrue(ledger.is_empty)
        self.assertEqual(ledger.rejected[0].code, "FIELD_UNSUPPORTED")

    def test_out_of_range_clamped(self):
        ledger = self.build('   crs:Clarity2012="400"\n')
        self.assertEqual(ledger.values["Clarity2012"], 100)

    def test_missing_notes_only_warns(self):
        ledger = self.build('   crs:Clarity2012="9"\n')
        self.assertFalse(ledger.rejected)
        self.assertTrue(ledger.warnings)

    def test_temperature_from_xmp_is_absolute(self):
        ledger = self.build('   crs:Temperature="6200"\n')
        self.assertEqual(ledger.values["Temperature"], 6200)
        self.assertEqual(ledger.values["WhiteBalance"], "Custom")


if __name__ == "__main__":
    unittest.main()

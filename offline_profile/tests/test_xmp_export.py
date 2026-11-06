"""Pilot XMP 导出测试。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from offline_profile.xmp_export import export_pilot_xmp


class TestXmpExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="xmp_export_")
        self.build = os.path.join(self.tmp, "build")
        os.makedirs(os.path.join(self.build, "buckets", "natural_landscape"))
        os.makedirs(os.path.join(self.build, "buckets", "urban_architecture"))
        signature = {
            "parameters": {
                "highlights": {"central": -19.0},
                "shadows": {"central": -2.5},
                "blacks": {"central": -1.0},
                "vibrance": {"central": 12.75},
                "saturation": {"central": 10.0},
                "sharpening_amount": {"central": 32.5},
            }
        }
        profile = {
            "build_id": "build_test",
            "status": "draft",
            "validation_scope": "analysis_consistency",
            "buckets": {
                "natural_landscape": {
                    "status": "draft",
                    "deltas": {"shadows": {"delta": 12.5}, "blacks": {"delta": 9.0}},
                },
                "urban_architecture": {
                    "status": "draft",
                    "deltas": {"shadows": {"delta": -12.5}, "blacks": {"delta": -9.0}},
                },
            },
        }
        for bucket, contrast in (("natural_landscape", -8.0), ("urban_architecture", 15.0)):
            with open(os.path.join(self.build, "buckets", bucket, "stats.json"), "w", encoding="utf-8") as handle:
                json.dump({"params": {"contrast": {"central": contrast}, "clarity": {"central": -8.0}}}, handle)
        with open(os.path.join(self.build, "global_signature.json"), "w", encoding="utf-8") as handle:
            json.dump(signature, handle)
        with open(os.path.join(self.build, "profile.json"), "w", encoding="utf-8") as handle:
            json.dump(profile, handle)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _attrs(self, path):
        root = ET.parse(path).getroot()
        description = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
        return description.attrib

    def test_exports_global_and_bucket_presets(self):
        out = os.path.join(self.tmp, "out")
        manifest = export_pilot_xmp(self.build, out)
        self.assertEqual(manifest["status"], "pilot_only")
        self.assertFalse(manifest["application_ready"])
        self.assertEqual(len(manifest["presets"]), 3)

        natural = self._attrs(os.path.join(out, "pilot_natural_landscape.xmp"))
        self.assertEqual(natural["{http://ns.adobe.com/camera-raw-settings/1.0/}Shadows2012"], "10.00")
        self.assertEqual(natural["{http://ns.adobe.com/camera-raw-settings/1.0/}Contrast2012"], "-8.00")
        urban = self._attrs(os.path.join(out, "pilot_urban_architecture.xmp"))
        self.assertEqual(urban["{http://ns.adobe.com/camera-raw-settings/1.0/}Shadows2012"], "-15.00")
        self.assertEqual(urban["{http://ns.adobe.com/camera-raw-settings/1.0/}Contrast2012"], "15.00")
        self.assertEqual(natural["{http://ns.adobe.com/camera-raw-settings/1.0/}WhiteBalance"], "As Shot")
        self.assertIn("temperature_shift", [item["key"] for item in manifest["skipped"]])

    def test_bucket_primary_uses_absolute_parameters(self):
        profile_path = os.path.join(self.build, "profile.json")
        with open(profile_path, encoding="utf-8") as handle:
            profile = json.load(handle)
        profile["profile_shape"] = "bucket_primary"
        profile["bucket_parameters"] = {
            "urban_architecture": {
                "contrast": {"value": 18.0, "status": "published"},
                "shadows": {"value": -20.0, "status": "published"},
                "clarity": {"value": 15.0, "status": "published"},
                "blacks": {"value": -15.0, "status": "wide_dispersion"},
            },
            "natural_landscape": {
                "shadows": {"value": 5.0, "status": "wide_dispersion"},
            },
        }
        with open(profile_path, "w", encoding="utf-8") as handle:
            json.dump(profile, handle)

        out = os.path.join(self.tmp, "out_bucket_primary")
        manifest = export_pilot_xmp(self.build, out)
        urban = self._attrs(os.path.join(out, "pilot_urban_architecture.xmp"))
        ns = "{http://ns.adobe.com/camera-settings/1.0/}"
        self.assertEqual(urban["{http://ns.adobe.com/camera-raw-settings/1.0/}Contrast2012"], "18.00")
        self.assertEqual(urban["{http://ns.adobe.com/camera-raw-settings/1.0/}Shadows2012"], "-20.00")
        preset = next(item for item in manifest["presets"] if item["scope"] == "urban_architecture")
        self.assertIn("contrast", preset["candidate_params"])
        self.assertIn("blacks", preset["pilot_only_params"])

    def test_strict_excludes_unstable_bucket_params(self):
        out = os.path.join(self.tmp, "out_strict")
        manifest = export_pilot_xmp(self.build, out, include_unstable=False)
        self.assertTrue(manifest["application_ready"] is False)
        urban = self._attrs(os.path.join(out, "pilot_urban_architecture.xmp"))
        self.assertEqual(urban["{http://ns.adobe.com/camera-raw-settings/1.0/}Contrast2012"], "0.00")


if __name__ == "__main__":
    unittest.main()

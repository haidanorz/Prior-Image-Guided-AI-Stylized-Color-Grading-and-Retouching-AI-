from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from online_decode.errors import Codes, DecodeError
from online_decode.fingerprint.resolver import (
    decide_bucket,
    load_narrative,
    references,
    verify_package,
)

PACKAGE = Path(__file__).resolve().parents[2] / "examples" / "sample_package"


def obs(label="natural_landscape", score=0.85, second=("urban_architecture", 0.10)):
    candidates = [{"label": label, "score": score}]
    if second:
        candidates.append({"label": second[0], "score": second[1]})
    return {"primary_candidates": candidates, "lighting": "soft", "brightness": "normal", "backlit": False}


class BucketPrimaryTest(unittest.TestCase):
    def _pkg(self, fingerprint):
        from online_decode.fingerprint.resolver import PackageInfo

        return PackageInfo(
            path=Path("."), manifest={}, fingerprint=fingerprint,
            fingerprint_id="fp_test", fingerprint_hash="h", status="draft",
            application_ready=False, schema_version="1.1.0",
        )

    def test_bucket_parameters_override_global(self):
        pkg = self._pkg(
            {
                "shape": "bucket_primary",
                "global": {"highlights": {"central": -11.0}, "vibrance": {"central": 15.0}},
                "buckets": {
                    "natural_landscape": {
                        "parameters": {"highlights": {"value": -10.0, "consistency": 0.91}},
                        "priors": {"shadows": {"value": 5.0}},
                        "not_evaluable": ["contrast"],
                    }
                },
                "insufficient_evidence": [{"key": "contrast", "reason": "single_bucket_evidence"}],
            }
        )
        refs = references(pkg, "natural_landscape")
        self.assertEqual(refs["Highlights2012"]["value"], -10.0)
        self.assertEqual(refs["Highlights2012"]["role"], "bucket")
        self.assertEqual(refs["Vibrance"]["value"], 15.0)
        self.assertEqual(refs["Vibrance"]["role"], "global_fallback")
        self.assertTrue(refs["Contrast2012"]["not_evaluable"])
        self.assertNotIn("Shadows2012", refs)

    def test_legacy_shape_still_uses_delta(self):
        pkg = self._pkg(
            {
                "global": {"highlights": {"central": -11.0}},
                "buckets": {"natural_landscape": {"delta": {"highlights": -2.0}}},
                "insufficient_evidence": [],
            }
        )
        refs = references(pkg, "natural_landscape")
        self.assertEqual(refs["Highlights2012"]["value"], -2.0)
        self.assertEqual(refs["Highlights2012"]["role"], "bucket_delta")


class ResolverTest(unittest.TestCase):
    def test_real_package_verifies(self):
        pkg = verify_package(PACKAGE)
        manifest = json.loads((PACKAGE / "package_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(pkg.fingerprint_id.startswith("fp_"))
        self.assertEqual(len(pkg.file_hashes), len(manifest["files"]))
        self.assertIn("natural_landscape", pkg.bucket_ids)

    def test_tampered_file_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pkg"
            shutil.copytree(PACKAGE, copy)
            readme = copy / "README.md"
            readme.write_text(readme.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(DecodeError) as ctx:
                verify_package(copy)
            self.assertEqual(ctx.exception.code, Codes.PACKAGE_HASH_MISMATCH)

    def test_missing_file_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pkg"
            shutil.copytree(PACKAGE, copy)
            (copy / "global_signature.json").unlink()
            with self.assertRaises(DecodeError) as ctx:
                verify_package(copy)
            self.assertEqual(ctx.exception.code, Codes.PACKAGE_HASH_MISMATCH)

    def test_core_bucket_hit(self):
        pkg = verify_package(PACKAGE)
        decision = decide_bucket(pkg, obs(score=0.85))
        self.assertEqual(decision["level"], "core")
        self.assertEqual(decision["bucket_id"], "natural_landscape")
        self.assertIsNone(decision["fallback"])

    def test_low_confidence_falls_back_to_global(self):
        pkg = verify_package(PACKAGE)
        decision = decide_bucket(pkg, obs(score=0.50))
        self.assertEqual(decision["level"], "associate")
        self.assertIsNone(decision["bucket_id"])
        self.assertEqual(decision["fallback"], "global")
        self.assertEqual(decision["fallback_reason"], "core_score_not_met")

    def test_narrow_margin_falls_back(self):
        pkg = verify_package(PACKAGE)
        decision = decide_bucket(pkg, obs(score=0.70, second=("urban_architecture", 0.68)))
        self.assertEqual(decision["level"], "associate")
        self.assertEqual(decision["fallback_reason"], "core_margin_not_met")

    def test_other_label_is_unassigned(self):
        pkg = verify_package(PACKAGE)
        decision = decide_bucket(pkg, obs(label="other", score=0.9, second=None))
        self.assertEqual(decision["level"], "unassigned")
        self.assertIsNone(decision["bucket_id"])

    def test_narrative_hash_verified(self):
        pkg = verify_package(PACKAGE)
        narrative = load_narrative(pkg, "natural_landscape")
        self.assertIsNotNone(narrative)
        self.assertTrue(narrative["cautions"])

    def test_references_carry_evidence_levels(self):
        pkg = verify_package(PACKAGE)
        refs = references(pkg, "natural_landscape")
        runtime = [k for k, v in refs.items() if v["evidence_level"] == "runtime_reference"]
        weak = [k for k, v in refs.items() if v["evidence_level"] == "experimental_prior"]
        self.assertTrue(runtime, msg="指纹应至少有全局 runtime 参考")
        self.assertTrue(weak, msg="insufficient_evidence 应被标成 experimental_prior")
        for key in weak:
            self.assertIsNone(refs[key]["value"])


if __name__ == "__main__":
    unittest.main()

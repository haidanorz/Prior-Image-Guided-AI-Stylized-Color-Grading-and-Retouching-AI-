"""特征指纹包导出测试。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from offline_profile.fingerprint_package import export_fingerprint_package


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _canonical(payload):
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class TestFingerprintPackage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fpkg_")
        self.build = os.path.join(self.tmp, "build")
        _write(
            os.path.join(self.build, "global_signature.json"),
            {
                "taxonomy_version": "scene_v1",
                "registry_version": "1.0.0",
                "parameters": {
                    "highlights": {"central": -19.0, "unit": "index", "kind": "numeric",
                                   "cross_scene_invariant": True, "evidence_buckets": ["natural_landscape", "urban_architecture"],
                                   "sign_agreement": 1.0, "weighted_consistency": 1.0},
                },
                "insufficient_evidence": [{"key": "contrast", "reason": "single_bucket_evidence"}],
                "scene_dependent_parameters": ["shadows"],
                "invariant_parameters": ["highlights"],
                "constraints": [],
                "local_adjustment_policy": "no local adjustments",
            },
        )
        _write(
            os.path.join(self.build, "profile.json"),
            {
                "build_id": "build_test",
                "profile_id": "profile_test",
                "validation_scope": "analysis_consistency",
                "input_fingerprint": "a" * 64,
                "config_hash": "b" * 16,
                "response_set_hash": "c" * 64,
                "aggregate_hash": "d" * 64,
                "buckets": {
                    "natural_landscape": {
                        "bucket_status": "full",
                        "n_core": 34,
                        "n_train": 27,
                        "n_holdout": 7,
                        "deltas": {"shadows": {"delta": 12.5, "unit": "index"}, "blacks": {"delta": 9.0, "unit": "index"}},
                        "unresolved": [],
                    }
                },
            },
        )
        _write(os.path.join(self.build, "buckets", "index.json"), {"buckets": []})
        _write(os.path.join(self.build, "buckets", "natural_landscape", "profile.json"), {"profile_hash": "e" * 64})
        _write(os.path.join(self.build, "buckets", "natural_landscape", "delta.json"), {"deltas": {}})
        _write(os.path.join(self.build, "buckets", "natural_landscape", "stats.json"), {"params": {}})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_package_hashes_and_contents(self):
        out = os.path.join(self.tmp, "pkg")
        manifest = export_fingerprint_package(self.build, out)
        self.assertEqual(manifest["status"], "draft")
        self.assertFalse(manifest["application_ready"])
        self.assertTrue(manifest["package_id"].startswith("fpkg_"))
        self.assertTrue(os.path.isfile(os.path.join(out, "package_manifest.json")))
        self.assertFalse(os.path.exists(out + ".zip"))

        with open(os.path.join(out, "fingerprint.json"), encoding="utf-8") as handle:
            fingerprint = json.load(handle)
        self.assertEqual(fingerprint["buckets"]["natural_landscape"]["delta"]["shadows"], 12.5)
        self.assertEqual(fingerprint["buckets"]["natural_landscape"]["profile_hash"], "e" * 64)
        recompute = {k: v for k, v in fingerprint.items() if k != "fingerprint_hash"}
        recompute.pop("fingerprint_id")
        self.assertEqual(_canonical(recompute), fingerprint["fingerprint_hash"])

        for item in manifest["files"]:
            path = os.path.join(out, item["path"])
            self.assertTrue(os.path.isfile(path), item["path"])
            with open(path, "rb") as handle:
                self.assertEqual(hashlib.sha256(handle.read()).hexdigest(), item["sha256"])
        expected = _canonical(
            {
                "package_schema_version": manifest["package_schema_version"],
                "fingerprint_hash": manifest["fingerprint_hash"],
                "files": manifest["files"],
            }
        )
        self.assertEqual(expected, manifest["package_hash"])


if __name__ == "__main__":
    unittest.main()

"""ProfileComposer 的数值口径测试。"""

from __future__ import annotations

import unittest

from offline_profile.contract import load_parameters
from offline_profile.profile_compose import (
    _cross_validate,
    build_bucket_parameters,
    _param_index,
    analysis_coverage,
    bucket_param_stats,
    build_bucket_profile,
    circular_delta,
    circular_dispersion,
    circular_median,
    weighted_median,
)


class TestAnalysisCoverage(unittest.TestCase):
    def _analysis(self, count):
        keys = ["contrast", "highlights", "shadows", "blacks", "clarity", "saturation", "vibrance"]
        return {"edit_hypotheses": {key: {} for key in keys[:count]}}

    def test_sparse_analysis_is_blocked(self):
        cfg = {"analysis_min_mean_hypotheses": 6.0, "analysis_min_key_coverage": 0.65}
        result = analysis_coverage({"a": self._analysis(3), "b": self._analysis(3)}, cfg)
        self.assertFalse(result["passed"])
        self.assertIn("mean_hypotheses_below_threshold", result["reasons"])

    def test_dense_analysis_passes(self):
        cfg = {"analysis_min_mean_hypotheses": 6.0, "analysis_min_key_coverage": 0.65}
        result = analysis_coverage({"a": self._analysis(7), "b": self._analysis(7)}, cfg)
        self.assertTrue(result["passed"])
        self.assertEqual(result["min_key_coverage"], 1.0)


class TestKfoldValidation(unittest.TestCase):
    def test_out_of_fold_compares_global_and_bucket(self):
        analyses = {}
        members = {}
        for bucket, value in (("a", -5.0), ("b", -15.0)):
            ids = [f"img_{bucket}_{index:03d}" for index in range(10)]
            members[bucket] = {"train": ids, "holdout": []}
            for image_id in ids:
                analyses[image_id] = {
                    "image_id": image_id,
                    "edit_hypotheses": {
                        "highlights": {
                            "estimate": value,
                            "range": [value - 5, value + 5],
                            "confidence": 0.8,
                            "evidence": "bright sky retains texture",
                        }
                    },
                }
        stats = {
            bucket: {
                "params": {
                    "highlights": {"central": value, "n_valid": 10, "stable": True}
                },
                "reliability": 1.0,
            }
            for bucket, value in (("a", -5.0), ("b", -15.0))
        }
        split = {
            "validation_mode": "kfold",
            "buckets": {
                "a": {"folds": [members["a"]["train"][:5], members["a"]["train"][5:]]},
                "b": {"folds": [members["b"]["train"][:5], members["b"]["train"][5:]]},
            },
        }
        result = _cross_validate(
            analyses=analyses,
            stats=stats,
            bucket_members=members,
            global_params={"highlights": {"central": -10.0}},
            bucket_deltas={"a": {"deltas": {}}, "b": {"deltas": {}}},
            spec_by_key=_param_index(load_parameters()),
            conf_min=0.30,
            iterations=50,
            seed=0,
            split=split,
        )
        for bucket in ("a", "b"):
            entry = result[bucket]["params"]["highlights"]
            self.assertEqual(entry["n"], 10)
            self.assertEqual(entry["mae_global_only"], 5.0)
            self.assertEqual(entry["mae_bucket_only"], 0.0)
            self.assertEqual(entry["relative_improvement"], 1.0)
            self.assertEqual(entry["ci90"], [1.0, 1.0])
            self.assertTrue(entry["ci_excludes_zero"])
            self.assertFalse(entry["published"])
            self.assertIsNone(entry["published_delta"])
        self.assertEqual(result["a"]["n_folds"], 2)


class TestBucketParameters(unittest.TestCase):
    def _stat(self, value, p25, p75, n, consistency, stable):
        return {
            "central": value, "p25": p25, "p75": p75, "n_valid": n,
            "support": n / 20, "consistency": consistency, "stable": stable,
        }

    def test_status_matrix(self):
        spec = _param_index(load_parameters())
        stats = {
            "natural_landscape": {
                "params": {
                    "highlights": self._stat(-10.0, -15.0, -10.0, 18, 0.9, True),
                    "contrast": self._stat(10.0, -8.0, 20.0, 18, 0.75, True),
                    "shadows": self._stat(5.0, -5.0, 15.0, 18, 0.4, False),
                    "exposure_ev": self._stat(0.0, -0.1, 0.0, 18, 0.7, True),
                }
            }
        }
        cv = {
            "natural_landscape": {
                "params": {
                    "highlights": {"relative_improvement": 0.06, "ci90": [0.0, 0.1]},
                    "contrast": {"relative_improvement": -0.20, "ci90": [-0.3, -0.1]},
                    "shadows": {"relative_improvement": 0.22, "ci90": [0.05, 0.4]},
                    "exposure_ev": {"relative_improvement": 0.0, "ci90": [0.0, 0.0]},
                }
            }
        }
        params = build_bucket_parameters(stats, cv, spec)["natural_landscape"]
        self.assertEqual(params["highlights"]["status"], "published")
        self.assertEqual(params["contrast"]["status"], "withheld")
        self.assertEqual(params["contrast"]["reason"], "cv_vs_pooled_negative")
        self.assertEqual(params["shadows"]["status"], "wide_dispersion")
        self.assertEqual(params["shadows"]["apply_policy"], "prior_only")
        self.assertEqual(params["exposure_ev"]["status"], "neutral")
        self.assertEqual(params["exposure_ev"]["apply_policy"], "off")


class TestWeightedMedian(unittest.TestCase):
    def test_exact_half_takes_midpoint(self):
        self.assertEqual(weighted_median([(10.0, 1.0), (20.0, 1.0)]), 15.0)

    def test_unequal_weight_does_not_split(self):
        self.assertEqual(weighted_median([(10.0, 2.0), (20.0, 1.0)]), 10.0)


class TestBucketProfile(unittest.TestCase):
    def test_profile_hash_is_stable_and_contains_refs(self):
        stats = {
            "bucket_id": "portrait",
            "n_core": 12,
            "n_train": 10,
            "n_holdout": 2,
            "params": {"contrast": {"outliers": ["img_a"]}},
            "representatives": ["img_a"],
        }
        delta = {
            "bucket_id": "portrait",
            "status": "draft",
            "bucket_status": "full",
            "deltas": {"contrast": {"delta": 12.0, "status": "candidate"}},
            "unresolved": [],
            "dropped": [],
            "conditional_deltas": [],
        }
        signature = {"taxonomy_version": "scene_v1"}
        profile = {
            "profile_id": "profile_x",
            "aggregate_hash": "agg",
            "global_signature": {"file_hash": "glob"},
            "holdout_eval": {"portrait": {"n_holdout": 2}},
            "merge_check": {"portrait": {"within_tolerance": True}},
        }
        first = build_bucket_profile("portrait", stats, delta, signature, profile)
        second = build_bucket_profile("portrait", stats, delta, signature, profile)
        self.assertEqual(first["profile_hash"], second["profile_hash"])
        self.assertEqual(first["base_profile_ref"]["aggregate_hash"], "agg")
        self.assertEqual(first["evidence"]["stats_hash"], second["evidence"]["stats_hash"])
        self.assertEqual(first["outliers"], ["img_a"])


class TestCircular(unittest.TestCase):
    def test_delta_uses_short_arc(self):
        self.assertEqual(circular_delta(10.0, 350.0), 20.0)
        self.assertEqual(circular_delta(350.0, 10.0), -20.0)

    def test_median_around_zero_not_180(self):
        center = circular_median([(350.0, 1.0), (10.0, 1.0)])
        self.assertLessEqual(min(abs(center - 0.0), abs(center - 360.0)), 10.0)

    def test_dispersion_is_concentration(self):
        self.assertAlmostEqual(circular_dispersion([(350.0, 1.0), (10.0, 1.0)]), 0.9848, places=3)
        self.assertAlmostEqual(circular_dispersion([(0.0, 1.0), (180.0, 1.0)]), 0.0, places=3)

    def test_circular_consistency_is_not_sign_based(self):
        spec = {"kind": "circular", "unit": "degree", "practical_zero_band": 15.0,
                "significance_threshold": 12.0, "direction": "circular"}
        stats = bucket_param_stats(
            [("a", 350.0, 1.0), ("b", 10.0, 1.0)], spec, n_train=2
        )
        self.assertEqual(stats["dominant_class"], "circular")
        self.assertAlmostEqual(stats["consistency"], 0.9848, places=3)
        self.assertFalse(stats["stable"])


if __name__ == "__main__":
    unittest.main()

"""契约层与单图/分桶/划分的自动化验收。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from offline_profile.agent_flow import build_status, finalize_build, list_tasks, preview_build, start_build, submit_batch
from offline_profile.analysis.adapter import AnalysisRequest, ModelInfo, build_adapter
from offline_profile.analysis.normalize import normalize_analysis
from offline_profile.bucketing import assign_scene
from offline_profile.config import config_hash, resolve_config
from offline_profile.errors import AdapterError, FatalError, InvalidAnalysis
from offline_profile.intake import load_input
from offline_profile.pipeline import run
from offline_profile.splitting import build_evaluation_split
from offline_profile.util import read_json, read_jsonl
from offline_profile.verify import verify_build
from offline_profile.version import ANALYSIS_SCHEMA_VERSION, PROMPT_VERSION

from .fixtures import MODEL_INFO, base_analysis, build_analyses, build_input, write_mock


def model_info() -> ModelInfo:
    return ModelInfo(
        provider=MODEL_INFO["provider"],
        model=MODEL_INFO["model"],
        version=MODEL_INFO["version"],
        is_mock=True,
    )


class TestConfig(unittest.TestCase):
    def test_unknown_key_rejected(self):
        with self.assertRaises(FatalError):
            resolve_config({"core_top_scores": 0.7})

    def test_hash_changes_with_threshold(self):
        base = resolve_config()
        other = resolve_config({"core_margin": 0.2})
        self.assertNotEqual(config_hash(base), config_hash(other))

    def test_overwrite_not_hashed(self):
        self.assertEqual(
            config_hash(resolve_config({"overwrite": True})),
            config_hash(resolve_config()),
        )

    def test_invalid_adapter_rejected(self):
        with self.assertRaises(FatalError):
            resolve_config({"adapter": "openai"})

    def test_threshold_order(self):
        with self.assertRaises(FatalError):
            resolve_config({"associate_top_score": 0.7, "core_top_score": 0.6})


class TestNormalize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="op_norm_")
        cls.cfg = resolve_config()
        cls.input_info = build_input(cls.tmp, count=1)
        cls.analyses = build_analyses(cls.input_info["input_dir"], cls.cfg)
        cls.intake = load_input(cls.input_info["input_dir"])
        cls.image = cls.intake.images[0]
        cls.payload = cls.analyses[cls.image.image_id]
        cls.fingerprint = cls.payload["input_fingerprint"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _normalize(self, payload, raw_text=None):
        text = raw_text if raw_text is not None else json.dumps(payload, ensure_ascii=False)
        return normalize_analysis(
            text,
            image=self.image,
            expected_fingerprint=self.fingerprint,
            config=self.cfg,
            model_info=model_info(),
        )

    def test_valid(self):
        result = self._normalize(self.payload)
        self.assertEqual(result.payload["image_id"], self.image.image_id)
        self.assertAlmostEqual(sum(c["score"] for c in result.payload["scene_observation"]["primary_candidates"]), 1.0, places=5)

    def test_scores_renormalized(self):
        payload = json.loads(json.dumps(self.payload))
        payload["scene_observation"]["primary_candidates"] = [
            {"label": "portrait", "score": 0.8},
            {"label": "other", "score": 0.8},
        ]
        result = self._normalize(payload)
        scores = {item["label"]: item["score"] for item in result.payload["scene_observation"]["primary_candidates"]}
        self.assertAlmostEqual(scores["portrait"], 0.5, places=5)
        self.assertIn("candidate_scores_renormalized", result.warnings)

    def test_duplicate_labels_merged(self):
        payload = json.loads(json.dumps(self.payload))
        payload["scene_observation"]["primary_candidates"] = [
            {"label": "portrait", "score": 0.5},
            {"label": "portrait", "score": 0.3},
            {"label": "other", "score": 0.2},
        ]
        result = self._normalize(payload)
        labels = [item["label"] for item in result.payload["scene_observation"]["primary_candidates"]]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertIn("duplicate_candidate_labels", result.warnings)

    def test_brightness_overridden(self):
        payload = json.loads(json.dumps(self.payload))
        payload["scene_observation"]["brightness"] = "bright"
        result = self._normalize(payload)
        self.assertEqual(result.payload["scene_observation"]["brightness"], "mid")
        self.assertIn("brightness_overridden_by_objective", result.warnings)

    def test_unknown_parameter_rejected(self):
        payload = json.loads(json.dumps(self.payload))
        payload["edit_hypotheses"]["fake_slider"] = {
            "estimate": 1,
            "range": [0, 2],
            "confidence": 0.5,
            "evidence": "invented",
        }
        with self.assertRaises(InvalidAnalysis) as ctx:
            self._normalize(payload)
        self.assertTrue(any("fake_slider" in detail for detail in ctx.exception.details))

    def test_out_of_range_rejected(self):
        payload = json.loads(json.dumps(self.payload))
        payload["edit_hypotheses"]["exposure_ev"] = {
            "estimate": 5.0,
            "range": [4.0, 6.0],
            "confidence": 0.5,
            "evidence": "too strong",
        }
        with self.assertRaises(InvalidAnalysis):
            self._normalize(payload)

    def test_estimate_outside_range_rejected(self):
        payload = json.loads(json.dumps(self.payload))
        payload["edit_hypotheses"]["exposure_ev"] = {
            "estimate": 0.5,
            "range": [0.0, 0.2],
            "confidence": 0.5,
            "evidence": "inconsistent",
        }
        with self.assertRaises(InvalidAnalysis):
            self._normalize(payload)

    def test_missing_narrative_rejected(self):
        payload = json.loads(json.dumps(self.payload))
        payload.pop("narrative_observation")
        with self.assertRaises(InvalidAnalysis):
            self._normalize(payload)

    def test_narrative_with_numeric_parameter_rejected(self):
        payload = json.loads(json.dumps(self.payload))
        payload["narrative_observation"]["atmosphere"] = "阴影被抬起约 -15 EV，画面更平。"
        with self.assertRaises(InvalidAnalysis):
            self._normalize(payload)

    def test_circular_range_wraps(self):
        payload = json.loads(json.dumps(self.payload))
        payload["edit_hypotheses"]["shadow_tint_hue"] = {
            "estimate": 355.0,
            "range": [350.0, 10.0],
            "confidence": 0.5,
            "evidence": "cool shadows with a slight blue cast",
        }
        result = self._normalize(payload)
        self.assertEqual(result.payload["edit_hypotheses"]["shadow_tint_hue"]["estimate"], 355.0)
        payload["edit_hypotheses"]["shadow_tint_hue"]["estimate"] = 180.0
        with self.assertRaises(InvalidAnalysis):
            self._normalize(payload)

    def test_id_mismatch_rejected(self):
        payload = json.loads(json.dumps(self.payload))
        payload["image_id"] = "img_ffffffffffffffff"
        with self.assertRaises(InvalidAnalysis):
            self._normalize(payload)

    def test_markdown_fence_stripped(self):
        text = "```json\n" + json.dumps(self.payload, ensure_ascii=False) + "\n```"
        result = self._normalize(self.payload, raw_text=text)
        self.assertIn("markdown_fence_stripped", result.warnings)

    def test_nan_rejected(self):
        text = json.dumps(self.payload, ensure_ascii=False).replace('"estimate": 0.2', '"estimate": NaN')
        with self.assertRaises(InvalidAnalysis):
            self._normalize(self.payload, raw_text=text)


class TestParameterRegistry(unittest.TestCase):
    def test_confirmed_normalization_scales(self):
        from offline_profile.contract import edit_parameter_index

        index = edit_parameter_index()
        self.assertEqual(index["exposure_ev"]["scale"], 1.5)
        self.assertEqual(index["exposure_ev"]["unit"], "EV")
        for key in ("shadow_tint_hue", "highlight_tint_hue"):
            self.assertEqual(index[key]["scale"], 180.0)
            self.assertEqual(index[key]["kind"], "circular")

    def test_registry_keys_match_schema_usage(self):
        from offline_profile.contract import edit_parameter_index

        index = edit_parameter_index()
        self.assertEqual(len(index), 17)
        self.assertIn("temperature_shift", index)
        self.assertNotIn("temperature_kelvin", index)


class TestReplayAdapter(unittest.TestCase):
    def test_reads_frozen_responses_with_model_info(self):
        tmp = tempfile.mkdtemp(prefix="op_replay_")
        try:
            info = build_input(tmp, count=1)
            analyses = build_analyses(info["input_dir"])
            responses = os.path.join(tmp, "responses")
            write_mock(responses, analyses)
            with open(os.path.join(responses, "_model_info.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "provider": "agent",
                        "model": "agent-vision",
                        "version": "2026-11-04",
                        "is_mock": False,
                    },
                    handle,
                )
            adapter = build_adapter("replay", mock_dir=responses)
            self.assertEqual(adapter.model_info.provider, "agent")
            self.assertFalse(adapter.model_info.is_mock)
            image = load_input(info["input_dir"]).images[0]
            response = adapter.analyze(
                AnalysisRequest(image=image, prompt=None, input_fingerprint="0" * 64)
            )
            self.assertIn(image.image_id, response.text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_replay_requires_model_info(self):
        tmp = tempfile.mkdtemp(prefix="op_replay_missing_")
        try:
            with self.assertRaises(AdapterError):
                build_adapter("replay", mock_dir=tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestBucketing(unittest.TestCase):
    def setUp(self):
        self.cfg = resolve_config()

    def _assign(self, candidates):
        analysis = {
            "image_id": "img_0000000000000001",
            "scene_observation": {
                "primary_candidates": candidates,
                "lighting": "daylight",
                "brightness": "mid",
                "backlit": False,
            },
        }
        return assign_scene(analysis, self.cfg)

    def test_core(self):
        result = self._assign([{"label": "portrait", "score": 0.8}, {"label": "other", "score": 0.15}])
        self.assertEqual(result["level"], "core")
        self.assertEqual(result["bucket_id"], "portrait")

    def test_associate_by_score(self):
        result = self._assign([{"label": "portrait", "score": 0.5}, {"label": "other", "score": 0.3}])
        self.assertEqual(result["level"], "associate")
        self.assertIsNone(result["bucket_id"])

    def test_associate_by_margin(self):
        result = self._assign([{"label": "portrait", "score": 0.7}, {"label": "other", "score": 0.6}])
        self.assertEqual(result["level"], "associate")
        self.assertEqual(result["reason"], "core_margin_not_met")

    def test_unassigned_low_score(self):
        result = self._assign([{"label": "portrait", "score": 0.3}, {"label": "other", "score": 0.2}])
        self.assertEqual(result["level"], "unassigned")

    def test_other_never_core(self):
        result = self._assign([{"label": "other", "score": 0.95}, {"label": "portrait", "score": 0.05}])
        self.assertEqual(result["level"], "unassigned")
        self.assertEqual(result["reason"], "other_label")


def _assignments(bucket_counts: dict[str, int], start: int = 1) -> list[dict]:
    rows = []
    index = start
    for bucket, count in bucket_counts.items():
        for _ in range(count):
            rows.append(
                {
                    "image_id": f"img_{index:016x}",
                    "level": "core",
                    "bucket_id": bucket,
                }
            )
            index += 1
    return rows


class TestSplitting(unittest.TestCase):
    def setUp(self):
        self.cfg = resolve_config()
        self.holdout_cfg = resolve_config({"validation_mode": "holdout"})

    def test_kfold_partitions_core_deterministically(self):
        assignments = _assignments({"portrait": 12})
        first = build_evaluation_split(assignments, [], self.cfg)
        second = build_evaluation_split(list(reversed(assignments)), [], self.cfg)
        bucket = first["buckets"]["portrait"]
        self.assertEqual(bucket["split_status"], "kfold")
        self.assertEqual(bucket["n_folds"], 10)
        self.assertEqual(bucket["n_holdout"], 0)
        self.assertEqual(bucket["n_train"], 12)
        flat = [image_id for fold in bucket["folds"] for image_id in fold]
        self.assertEqual(sorted(flat), sorted(item["image_id"] for item in assignments))
        self.assertTrue(all(fold for fold in bucket["folds"]))
        self.assertEqual(first["split_hash"], second["split_hash"])
        self.assertEqual(bucket["folds"], second["buckets"]["portrait"]["folds"])

    def test_holdout_mode_still_available(self):
        assignments = _assignments({"portrait": 12})
        split = build_evaluation_split(assignments, [], self.holdout_cfg)
        bucket = split["buckets"]["portrait"]
        self.assertEqual(bucket["split_status"], "split")
        self.assertEqual(bucket["n_holdout"], 2)
        self.assertEqual(bucket["n_train"], 10)

    def test_near_duplicate_same_fold(self):
        assignments = _assignments({"portrait": 12})
        first_two = [assignments[0]["image_id"], assignments[1]["image_id"]]
        split = build_evaluation_split(assignments, [first_two], self.cfg)
        folds = split["buckets"]["portrait"]["folds"]
        self.assertTrue(any(set(first_two) <= set(fold) for fold in folds))

    def test_near_duplicate_same_side_holdout(self):
        assignments = _assignments({"portrait": 12})
        first_two = [assignments[0]["image_id"], assignments[1]["image_id"]]
        split = build_evaluation_split(assignments, [first_two], self.holdout_cfg)
        bucket = split["buckets"]["portrait"]
        self.assertTrue(
            set(first_two) <= set(bucket["train"]) or set(first_two) <= set(bucket["holdout"])
        )

    def test_cross_bucket_group_blocks_split(self):
        assignments = _assignments({"portrait": 8, "natural_landscape": 8})
        portrait_id = assignments[0]["image_id"]
        landscape_id = next(item["image_id"] for item in assignments if item["bucket_id"] == "natural_landscape")
        split = build_evaluation_split(assignments, [[portrait_id, landscape_id]], self.cfg)
        self.assertEqual(split["buckets"]["portrait"]["split_status"], "blocked")
        self.assertEqual(split["buckets"]["portrait"]["reason"], "cross_bucket_near_duplicate")

    def test_nine_core_still_gets_folds(self):
        split = build_evaluation_split(_assignments({"portrait": 9}), [], self.cfg)
        bucket = split["buckets"]["portrait"]
        self.assertEqual(bucket["status"], "full")
        self.assertEqual(bucket["split_status"], "kfold")
        self.assertEqual(bucket["n_folds"], 9)
        self.assertEqual(bucket["n_holdout"], 0)

    def test_provisional_and_exception(self):
        split = build_evaluation_split(_assignments({"portrait": 6, "indoor": 3}), [], self.cfg)
        self.assertEqual(split["buckets"]["portrait"]["status"], "provisional")
        self.assertEqual(split["buckets"]["indoor"]["status"], "exception_group")

    def test_provisional_gets_diagnostic_folds_only(self):
        split = build_evaluation_split(_assignments({"portrait": 6}), [], self.cfg)
        bucket = split["buckets"]["portrait"]
        self.assertEqual(bucket["split_status"], "provisional")
        self.assertEqual(bucket["train"], [])
        self.assertEqual(bucket["n_train"], 0)
        self.assertEqual(bucket["diagnostic_n_folds"], 6)
        self.assertEqual(sorted(i for fold in bucket["diagnostic_folds"] for i in fold),
                         sorted(item["image_id"] for item in _assignments({"portrait": 6})))


class TestIntake(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="op_intake_")
        self.info = build_input(self.tmp, count=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_only_ready_consumed(self):
        manifest_path = os.path.join(self.info["input_dir"], "manifest.jsonl")
        rows = read_jsonl(manifest_path)
        skipped = dict(rows[0])
        skipped["status"] = "skipped"
        skipped["skip_reason"] = "too_small"
        skipped["image_id"] = "img_fffffffffffffffe"
        with open(manifest_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(skipped, ensure_ascii=False) + "\n")
        intake = load_input(self.info["input_dir"])
        self.assertEqual(intake.ready_count, 3)
        self.assertEqual(intake.non_ready, 1)

    def test_preview_hash_mismatch_fatal(self):
        record = self.info["records"][0]
        with open(os.path.join(self.info["input_dir"], record["preview_path"]), "ab") as handle:
            handle.write(b"tamper")
        with self.assertRaises(FatalError):
            load_input(self.info["input_dir"])

    def test_mixed_versions_fatal(self):
        manifest_path = os.path.join(self.info["input_dir"], "manifest.jsonl")
        rows = read_jsonl(manifest_path)
        rows[0]["feature_schema_version"] = "0.9.0"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with self.assertRaises(FatalError):
            load_input(self.info["input_dir"])


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="op_pipe_")
        cls.cfg = resolve_config()
        cls.info = build_input(cls.tmp, count=13, near_groups=[[1, 2]])
        cls.analyses = build_analyses(cls.info["input_dir"], cls.cfg)
        cls.bad_id = cls.info["ids"][-1]
        cls.analyses.pop(cls.bad_id)
        cls.mock_dir = os.path.join(cls.tmp, "mock")
        write_mock(cls.mock_dir, cls.analyses)
        with open(os.path.join(cls.mock_dir, f"{cls.bad_id}.json"), "w", encoding="utf-8") as handle:
            json.dump({"raw_text": "this is not json"}, handle)
        cls.build_root = os.path.join(cls.tmp, "build")
        cls.result = run(
            cls.info["input_dir"], cls.build_root, mock_dir=cls.mock_dir, progress=lambda _m: None
        )
        cls.rows = read_jsonl(os.path.join(cls.build_root, "manifest.jsonl"))
        cls.episodes_first = read_jsonl(os.path.join(cls.build_root, "episodes.jsonl"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_counts_and_coverage(self):
        analysis = self.result["report"]["analysis"]
        self.assertEqual(analysis["counts"], {"normalized": 12, "invalid": 1, "error": 0})
        self.assertAlmostEqual(analysis["coverage"], 12 / 13, places=5)
        self.assertTrue(analysis["coverage_ok"])
        self.assertEqual(len(self.rows), 13)

    def test_assignments_closed(self):
        assignments = read_jsonl(os.path.join(self.build_root, "assignments.jsonl"))
        self.assertEqual(len(assignments), 12)
        self.assertTrue(all(item["level"] == "core" for item in assignments))
        self.assertTrue(all(item["bucket_id"] == "portrait" for item in assignments))

    def test_split(self):
        split = read_json(os.path.join(self.build_root, "evaluation_split.json"))
        bucket = split["buckets"]["portrait"]
        self.assertEqual(bucket["status"], "full")
        self.assertEqual(bucket["split_status"], "kfold")
        self.assertEqual(bucket["n_folds"], 10)
        self.assertEqual(bucket["n_holdout"], 0)
        self.assertEqual(bucket["n_train"], 12)
        pair = set(self.info["ids"][:2])
        self.assertTrue(any(pair <= set(fold) for fold in bucket["folds"]))

    def test_episodes_record_invalid(self):
        episodes = self.episodes_first
        invalid = [row for row in episodes if row["event_type"] == "analysis_invalid" and row["image_id"] == self.bad_id]
        self.assertEqual(len(invalid), 3)
        retries = [row for row in episodes if row["event_type"] == "analysis_retry" and row["image_id"] == self.bad_id]
        self.assertEqual(len(retries), 2)

    def test_verify(self):
        result = verify_build(self.build_root)
        self.assertTrue(result["ok"], result["problems"][:5])
        self.assertEqual(result["normalized"], 12)

    def test_cache_reuse_and_rerun(self):
        second = run(
            self.info["input_dir"], self.build_root, mock_dir=self.mock_dir, progress=lambda _m: None
        )
        self.assertEqual(second["report"]["analysis"]["counts"]["normalized"], 12)
        rows = read_jsonl(os.path.join(self.build_root, "manifest.jsonl"))
        cached = [row for row in rows if row["analysis_status"] == "normalized" and row["cached"]]
        self.assertEqual(len(cached), 12)
        self.assertEqual(verify_build(self.build_root)["ok"], True)

    def test_build_state_guard(self):
        with self.assertRaises(FatalError):
            run(
                self.info["input_dir"],
                self.build_root,
                mock_dir=self.mock_dir,
                overrides={"core_margin": 0.2},
                progress=lambda _m: None,
            )


class TestScopeIsolation(unittest.TestCase):
    def test_no_network_or_lightroom_imports(self):
        import ast

        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        forbidden = ("requests", "urllib", "socket", "http", "subprocess", "openai", "anthropic")
        for name in os.listdir(package_dir):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(package_dir, name), encoding="utf-8") as handle:
                source = handle.read()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    self.assertFalse(
                        any(module == item or module.startswith(item + ".") for item in forbidden),
                        f"{name} 引入了不允许的模块：{module}",
                    )
            if name not in ("xmp_export.py", "fingerprint_package.py", "cli.py"):
                self.assertNotIn("xmp", source.lower(), f"{name} 出现了 XMP 相关代码")


class TestAgentFlow(unittest.TestCase):
    @staticmethod
    def _partial(analysis: dict) -> dict:
        keys = (
            "scene_observation",
            "visual_observation",
            "narrative_observation",
            "edit_hypotheses",
            "quality",
            "local_adjustments_suspected",
            "overall_impression",
            "warnings",
        )
        return {key: analysis[key] for key in keys}

    def test_start_submit_finalize(self):
        tmp = tempfile.mkdtemp(prefix="op_agent_")
        try:
            info = build_input(tmp, count=3)
            build_root = os.path.join(tmp, "build")
            started = start_build(info["input_dir"], build_root)
            self.assertEqual(started["task_count"], 3)
            tasks = list_tasks(build_root)
            self.assertEqual(tasks["remaining"], 3)
            self.assertEqual(len(tasks["tasks"]), 3)

            analyses = build_analyses(info["input_dir"])
            submissions = [
                {"image_id": image_id, "analysis": self._partial(payload)}
                for image_id, payload in analyses.items()
            ]
            result = submit_batch(build_root, submissions)
            self.assertTrue(all(item["accepted"] for item in result["results"]))
            preview = preview_build(build_root)
            self.assertEqual(preview["assignments"]["levels"]["core"], 3)
            self.assertEqual(preview["status"], "open")
            final = finalize_build(build_root)
            self.assertEqual(final["counts"], {"normalized": 3, "invalid": 0, "error": 0})
            self.assertEqual(verify_build(build_root)["ok"], True)
            self.assertEqual(build_status(build_root)["status"], "finalized")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_invalid_then_corrected(self):
        tmp = tempfile.mkdtemp(prefix="op_agent_invalid_")
        try:
            info = build_input(tmp, count=1)
            build_root = os.path.join(tmp, "build")
            start_build(info["input_dir"], build_root)
            image_id = info["ids"][0]
            analysis = self._partial(build_analyses(info["input_dir"])[image_id])
            analysis["edit_hypotheses"]["fake_slider"] = {
                "estimate": 1,
                "range": [0, 2],
                "confidence": 0.5,
                "evidence": "invented",
            }
            rejected = submit_batch(build_root, [{"image_id": image_id, "analysis": analysis}])
            self.assertFalse(rejected["results"][0]["accepted"])
            self.assertEqual(rejected["counts"]["invalid"], 1)

            analysis["edit_hypotheses"].pop("fake_slider")
            accepted = submit_batch(
                build_root, [{"image_id": image_id, "analysis": analysis}], replace=True
            )
            self.assertTrue(accepted["results"][0]["accepted"])
            self.assertEqual(accepted["counts"]["submitted"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_limit(self):
        tmp = tempfile.mkdtemp(prefix="op_agent_limit_")
        try:
            info = build_input(tmp, count=5)
            started = start_build(info["input_dir"], os.path.join(tmp, "build"), limit=2)
            self.assertEqual(started["task_count"], 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestCLI(unittest.TestCase):
    def test_cli_run_and_verify(self):
        from contextlib import redirect_stdout
        import io

        from offline_profile.cli import main

        tmp = tempfile.mkdtemp(prefix="op_cli_")
        try:
            info = build_input(tmp, count=3)
            analyses = build_analyses(info["input_dir"])
            mock_dir = os.path.join(tmp, "mock")
            write_mock(mock_dir, analyses)
            build_root = os.path.join(tmp, "build")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(
                    [
                        "--input",
                        info["input_dir"],
                        "--build-root",
                        build_root,
                        "--mock-dir",
                        mock_dir,
                        "--quiet",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)
            summary = json.loads(buffer.getvalue())
            self.assertEqual(summary["analysis_counts"]["normalized"], 3)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(["verify", "--build-root", build_root, "--json"])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(buffer.getvalue())["ok"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

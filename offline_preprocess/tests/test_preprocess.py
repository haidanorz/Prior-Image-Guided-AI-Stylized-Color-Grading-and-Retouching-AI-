"""自动化验收测试（对应需求文档 AC-01 ~ AC-10）。

运行：
    python -m unittest discover -s offline_preprocess/tests -t . -v
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
from PIL import Image

from offline_preprocess import cli
from offline_preprocess.config import DEFAULTS, config_hash, resolve_config
from offline_preprocess.errors import FatalError
from offline_preprocess.features import compute_features
from offline_preprocess.geometry import compute_target_size
from offline_preprocess.run import run
from offline_preprocess.schema_utils import has_jsonschema, load_feature_schema, validate_feature_payload
from offline_preprocess.similarity import hamming_distance, phash64
from offline_preprocess.util import read_json, read_jsonl, sha256_file
from offline_preprocess.verify import verify_output

from .fixtures import HAS_LCMS, build_full_corpus, build_small_corpus

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def snapshot(root: str) -> dict[str, str]:
    result = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            result[os.path.normcase(full)] = sha256_file(full)
    return result


def count_supported(root: str) -> int:
    total = 0
    for _dirpath, _dirnames, filenames in os.walk(root):
        total += sum(1 for name in filenames if os.path.splitext(name)[1].lower() in SUPPORTED)
    return total


class TestGeometrySpec(unittest.TestCase):
    """PP-006 固定验收示例。"""

    def test_acceptance_table(self):
        cases = [
            ((6000, 4000), 1600, (1600, 1067), True),
            ((4000, 6000), 1600, (1067, 1600), True),
            ((8000, 1000), 1600, (1600, 200), True),
            ((1600, 1200), 1600, (1600, 1200), False),
            ((1200, 900), 1600, (1200, 900), False),
        ]
        for (w, h), long_edge, expected, resampled in cases:
            with self.subTest(size=(w, h)):
                target = compute_target_size(w, h, long_edge)
                self.assertEqual((target.width, target.height), expected)
                self.assertEqual(target.resampled, resampled)
                self.assertIsNone(target.warning)

    def test_extreme_aspect_clamped(self):
        target = compute_target_size(5000, 1, 1600)
        self.assertEqual((target.width, target.height), (1600, 1))
        self.assertEqual(target.warning, "extreme_resize_rounding")

    def test_no_upscale(self):
        target = compute_target_size(800, 600, 1600)
        self.assertFalse(target.resampled)
        self.assertEqual(target.scale, 1.0)


class TestColorMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = resolve_config()

    def _features(self, color, size=(400, 300)):
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[:] = color
        return compute_features(arr, self.cfg)[0]

    def test_primaries_lab(self):
        red = self._features((255, 0, 0))
        self.assertAlmostEqual(red["lab"]["L"]["mean"], 53.2408, delta=0.2)
        self.assertAlmostEqual(red["lab"]["a"]["mean"], 80.0925, delta=0.2)
        self.assertAlmostEqual(red["lab"]["b"]["mean"], 67.2032, delta=0.2)

    def test_white_black_luminance(self):
        white = self._features((255, 255, 255))
        black = self._features((0, 0, 0))
        self.assertAlmostEqual(white["luminance"]["mean"], 1.0, places=6)
        self.assertAlmostEqual(black["luminance"]["mean"], 0.0, places=6)
        self.assertAlmostEqual(white["luminance"]["near_white_ratio"], 1.0, places=6)
        self.assertAlmostEqual(black["luminance"]["near_black_ratio"], 1.0, places=6)
        self.assertEqual(black["sharpness"]["laplacian_variance"], 0.0)

    def test_solid_palette_and_saturation(self):
        red = self._features((255, 0, 0))
        self.assertEqual(len(red["palette"]), 1)
        self.assertEqual(red["palette"][0]["hex"], "#ff0000")
        self.assertAlmostEqual(red["palette"][0]["ratio"], 1.0, places=6)
        self.assertAlmostEqual(red["saturation"]["mean"], 1.0, places=6)
        self.assertAlmostEqual(red["hue"]["histogram"][3], 1.0, places=5)

    def test_histogram_sums(self):
        rng = np.random.RandomState(3)
        arr = rng.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        features = compute_features(arr, self.cfg)[0]
        self.assertAlmostEqual(sum(features["luminance"]["histogram"]["counts"]), 1.0, places=4)
        self.assertAlmostEqual(sum(features["hue"]["histogram"]), 1.0, places=4)
        self.assertAlmostEqual(sum(p["ratio"] for p in features["palette"]), 1.0, places=6)

    def test_no_chromatic_pixels_warning(self):
        gray = self._features((128, 128, 128))
        self.assertEqual(gray["hue"]["chromatic_pixel_ratio"], 0.0)
        self.assertTrue(all(v == 0.0 for v in gray["hue"]["histogram"]))

    def test_palette_deterministic(self):
        rng = np.random.RandomState(9)
        arr = rng.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        first = compute_features(arr, self.cfg)[0]["palette"]
        second = compute_features(arr, self.cfg)[0]["palette"]
        self.assertEqual(first, second)


class TestConfig(unittest.TestCase):
    def test_unknown_key_rejected(self):
        with self.assertRaises(FatalError):
            resolve_config({"long_egde": 1200})

    def test_threshold_fraction(self):
        cfg = resolve_config({"near_black_threshold": "4/255", "near_white_threshold": 250})
        self.assertEqual(cfg["near_black_threshold"], 4)
        self.assertEqual(cfg["near_white_threshold"], 250)

    def test_hash_changes_with_policy(self):
        a = resolve_config()
        b = resolve_config({"jpeg_quality": 80})
        self.assertNotEqual(config_hash(a), config_hash(b))
        self.assertEqual(config_hash(a), config_hash(resolve_config()))

    def test_invalid_policy_rejected(self):
        with self.assertRaises(FatalError):
            resolve_config({"alpha_policy": "composite_white"})
        with self.assertRaises(FatalError):
            resolve_config({"missing_icc_policy": "assume_adobe_rgb"})

    def test_defaults_match_spec(self):
        for key in DEFAULTS:
            self.assertIn(key, resolve_config())


class TestSimilarity(unittest.TestCase):
    def test_phash_stable_and_distance(self):
        rng = np.random.RandomState(11)
        a = rng.randint(0, 256, (200, 300), dtype=np.uint8)
        self.assertEqual(phash64(a), phash64(a.copy()))
        b = 255 - a
        self.assertGreaterEqual(hamming_distance(phash64(a), phash64(b)), 0)
        self.assertEqual(hamming_distance("0000000000000000", "0000000000000001"), 1)
        self.assertEqual(hamming_distance("ffffffffffffffff", "0000000000000000"), 64)


class TestFullRun(unittest.TestCase):
    """完整语料上的一次运行，覆盖 AC-01 ~ AC-07、AC-09、AC-10。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="op_full_")
        cls.info = build_full_corpus(cls.tmp)
        cls.inbox = cls.info["inbox"]
        cls.out = os.path.join(cls.tmp, "输出 目录")
        cls.before = snapshot(cls.inbox)
        cls.result = run(cls.inbox, cls.out, progress=lambda _m: None)
        cls.rows = read_jsonl(os.path.join(cls.out, "manifest.jsonl"))
        cls.by_key = {key: cls._find(cls.rows, path) for key, path in cls.info["paths"].items()}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def _find(rows, path):
        target = os.path.normcase(os.path.normpath(path))
        for row in rows:
            if os.path.normcase(os.path.normpath(row["source_path"])) == target:
                return row
        raise AssertionError(f"manifest 缺少记录：{path}")

    def _feature(self, key):
        row = self.by_key[key]
        return read_json(os.path.join(self.out, row["feature_path"]))

    def test_ac01_sources_unchanged(self):
        after = snapshot(self.inbox)
        self.assertEqual(self.before, after)
        self.assertEqual(set(self.before), set(after))

    def test_ac02_counts_closed(self):
        expected = count_supported(self.inbox)
        self.assertEqual(len(self.rows), expected)
        counts = self.result["report"]["status_counts"]
        self.assertEqual(sum(counts.values()), expected)
        self.assertTrue(self.result["report"]["counts_consistent"])
        seen = [row["source_path"] for row in self.rows]
        self.assertEqual(len(seen), len(set(os.path.normcase(os.path.normpath(p)) for p in seen)))

    def test_expected_statuses(self):
        ready = [
            "landscape",
            "portrait",
            "square",
            "orient6",
            "gray",
            "deep16",
            "disguised",
            "dup_a",
            "near_a",
            "near_b",
            "black",
            "white",
            "red",
            "gradient_gray",
            "low_sat",
            "high_sat",
            "extreme",
            "cn_subdir",
            "exif_meta",
            "noise",
        ]
        ready.append("icc_srgb")
        for key in ready:
            with self.subTest(key=key):
                self.assertEqual(self.by_key[key]["status"], "ready")
        self.assertEqual(self.by_key["alpha"]["status"], "skipped")
        self.assertEqual(self.by_key["alpha"]["skip_reason"], "unsupported_alpha")
        self.assertEqual(self.by_key["too_small"]["status"], "skipped")
        self.assertEqual(self.by_key["too_small"]["skip_reason"], "too_small")
        self.assertEqual(self.by_key["dup_copy"]["status"], "duplicate_exact")
        self.assertEqual(self.by_key["dup_rename"]["status"], "duplicate_exact")
        self.assertEqual(self.by_key["dup_copy"]["duplicate_of"], self.by_key["dup_a"]["image_id"])
        for key, code in (
            ("corrupt", "DECODE_FAILED"),
            ("truncated", "DECODE_FAILED"),
            ("notimage", "DECODE_FAILED"),
            ("multiframe", "MULTI_FRAME_UNSUPPORTED"),
            ("fake_gif", "UNSUPPORTED_FORMAT"),
        ):
            with self.subTest(key=key):
                self.assertEqual(self.by_key[key]["status"], "error")
                self.assertEqual(self.by_key[key]["error_code"], code)

    def test_ac03_preview_properties(self):
        cfg = resolve_config()
        ready = [row for row in self.rows if row["status"] == "ready"]
        self.assertEqual(len(os.listdir(os.path.join(self.out, "previews"))), len(ready))
        self.assertEqual(len(os.listdir(os.path.join(self.out, "features"))), len(ready))
        for row in ready:
            with self.subTest(image_id=row["image_id"]):
                path = os.path.join(self.out, row["preview_path"])
                self.assertEqual(sha256_file(path), row["preview_sha256"])
                with Image.open(path) as im:
                    im.load()
                    self.assertEqual(im.format, "JPEG")
                    self.assertEqual(im.mode, "RGB")
                    self.assertLessEqual(max(im.size), cfg["long_edge"])
                    self.assertTrue(im.info.get("icc_profile"))
                    self.assertEqual(int(im.getexif().get(274, 1)), 1)
                    feature = read_json(os.path.join(self.out, row["feature_path"]))
                    geometry = feature["geometry"]
                    self.assertEqual(im.size, (geometry["output_width"], geometry["output_height"]))
                    if not geometry["resampled"]:
                        self.assertEqual(
                            im.size, (geometry["oriented_width"], geometry["oriented_height"])
                        )

    def test_geometry_details(self):
        landscape = self._feature("landscape")["geometry"]
        self.assertEqual((landscape["output_width"], landscape["output_height"]), (1600, 1067))
        self.assertTrue(landscape["resampled"])
        self.assertEqual(landscape["rounding_rule"], "floor_x_plus_0.5_v1")

        portrait = self._feature("portrait")["geometry"]
        self.assertEqual((portrait["output_width"], portrait["output_height"]), (1067, 1600))

        square = self._feature("square")["geometry"]
        self.assertFalse(square["resampled"])
        self.assertEqual((square["output_width"], square["output_height"]), (1200, 1200))

        extreme = self._feature("extreme")
        self.assertEqual((extreme["geometry"]["output_width"], extreme["geometry"]["output_height"]), (1600, 1))
        self.assertIn("extreme_resize_rounding", extreme["warnings"])

    def test_ac05_orientation(self):
        feature = self._feature("orient6")
        geometry = feature["geometry"]
        self.assertEqual(geometry["orientation"], 6)
        self.assertTrue(geometry["orientation_applied"])
        self.assertEqual((geometry["decoded_width"], geometry["decoded_height"]), (1200, 800))
        self.assertEqual((geometry["oriented_width"], geometry["oriented_height"]), (800, 1200))
        self.assertEqual((geometry["output_width"], geometry["output_height"]), (800, 1200))

    def test_ac05_icc_and_missing_icc(self):
        gray = self._feature("gray")
        self.assertTrue(gray["source"]["was_grayscale"])
        self.assertTrue(gray["color_management"]["assumed_srgb"])
        self.assertIn("missing_icc_assumed_srgb", gray["warnings"])
        deep = self._feature("deep16")
        self.assertTrue(deep["source"]["was_grayscale"])
        icc = self._feature("icc_srgb")
        self.assertTrue(icc["color_management"]["icc_present"])
        self.assertFalse(icc["color_management"]["assumed_srgb"])
        self.assertEqual(icc["color_management"]["conversion_path"], "icc_srgb_identity")
        self.assertNotIn("missing_icc_assumed_srgb", icc["warnings"])

    def test_icc_transform_failure_is_error(self):
        row = self.by_key.get("icc_broken")
        if row is None:
            self.skipTest("当前环境无法生成 LAB 配置样本（需要 littlecms）")
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error_code"], "ICC_TRANSFORM_FAILED")

    def test_ac06_duplicates(self):
        dup = self.by_key["dup_copy"]
        self.assertIsNone(dup["preview_path"])
        self.assertIsNone(dup["feature_path"])
        self.assertEqual(self.by_key["dup_a"]["status"], "ready")
        near = read_json(os.path.join(self.out, "near_duplicates.json"))
        self.assertTrue(near["review_required"])
        group_ids = {member["image_id"] for group in near["groups"] for member in group["members"]}
        self.assertIn(self.by_key["near_a"]["image_id"], group_ids)
        self.assertIn(self.by_key["near_b"]["image_id"], group_ids)
        self.assertEqual(self.by_key["near_a"]["status"], "ready")
        self.assertEqual(self.by_key["near_b"]["status"], "ready")
        for group in near["groups"]:
            for pair in group["pairs"]:
                self.assertLessEqual(pair["hamming"], near["threshold"])

    def test_ac07_paths_and_json(self):
        for row in self.rows:
            if row["preview_path"]:
                self.assertRegex(row["preview_path"], r"^previews/img_[0-9a-f]{16}\.jpg$")
            if row["feature_path"]:
                self.assertRegex(row["feature_path"], r"^features/img_[0-9a-f]{16}\.json$")
        with open(os.path.join(self.out, "manifest.jsonl"), "rb") as handle:
            raw = handle.read()
        self.assertIn("输入 目录".encode("utf-8"), raw)
        self.assertNotIn(b"\\u8f93", raw)
        self.assertIn("子目录 中文", raw.decode("utf-8"))
        self.assertIn(".PNG", raw.decode("utf-8"))

    def test_ac04_feature_schema_and_finite(self):
        if not has_jsonschema():
            self.skipTest("jsonschema 未安装")
        schema = load_feature_schema()
        self.assertIsNotNone(schema)
        for row in self.rows:
            if row["status"] != "ready":
                continue
            with self.subTest(image_id=row["image_id"]):
                payload = read_json(os.path.join(self.out, row["feature_path"]))
                self.assertEqual(validate_feature_payload(payload, schema), [])
                text = json.dumps(payload, allow_nan=False)
                self.assertNotIn("NaN", text)
                self.assertNotIn("Infinity", text)

    def test_ac09_stats_from_memory_not_jpeg(self):
        feature = self._feature("noise")
        source = np.array(Image.open(self.info["paths"]["noise"]).convert("RGB"))
        expected = compute_features(source, resolve_config())[0]
        self.assertEqual(
            round(feature["luminance"]["mean"], 6), round(expected["luminance"]["mean"], 6)
        )
        self.assertEqual(
            feature["lab"]["chroma"]["mean"], round(expected["lab"]["chroma"]["mean"], 6)
        )
        preview = np.array(
            Image.open(os.path.join(self.out, self.by_key["noise"]["preview_path"])).convert("RGB")
        )
        preview_features = compute_features(preview, resolve_config())[0]
        # 若统计来自输出 JPEG，4:2:0 色度子采样 + 量化会让彩度均值明显下降
        self.assertLess(
            preview_features["lab"]["chroma"]["mean"],
            feature["lab"]["chroma"]["mean"] * 0.8,
        )

    def test_ac10_scope_isolation(self):
        forbidden = ("requests", "urllib", "socket", "http", "subprocess", "openai", "anthropic", "torch")
        for name in os.listdir(PACKAGE_DIR):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(PACKAGE_DIR, name), encoding="utf-8") as handle:
                source = handle.read()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    self.assertFalse(
                        any(module == f or module.startswith(f + ".") for f in forbidden),
                        f"{name} 引入了不允许的模块：{module}",
                    )
            for marker in ("AI-XMP", "demo_color_only", "style_profile", "tools_resize"):
                self.assertNotIn(marker, source, f"{name} 引用了旧项目：{marker}")

    def test_exif_summary(self):
        feature = self._feature("exif_meta")
        summary = feature["exif_summary"]
        self.assertEqual(summary["camera_make"], "TestMake")
        self.assertEqual(summary["camera_model"], "TestModel")
        self.assertEqual(summary["iso"], 200)
        self.assertEqual(summary["exposure_time"], "1/250")
        self.assertAlmostEqual(summary["f_number"], 2.8, places=3)
        self.assertNotIn("gps", json.dumps(summary).lower())

    def test_disguised_extension_uses_content(self):
        feature = self._feature("disguised")
        self.assertEqual(feature["source"]["format"], "PNG")

    def test_error_log(self):
        errors = read_jsonl(os.path.join(self.out, "errors.jsonl"))
        error_rows = [row for row in self.rows if row["status"] == "error"]
        self.assertEqual(len(errors), len(error_rows))
        for entry in errors:
            self.assertTrue(entry["stage"])
            self.assertTrue(entry["code"])
            self.assertTrue(entry["message"])
            self.assertTrue(entry["source_path"])

    def test_verify_command(self):
        result = verify_output(self.out)
        self.assertTrue(result["ok"], result["problems"][:5])
        self.assertEqual(result["checked"], len(self.rows))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["verify", "--output-dir", self.out, "--json"])
        self.assertEqual(code, 0)


class TestSmallRunBehaviour(unittest.TestCase):
    """小语料：幂等续跑、截断修复、输出目录嵌套排除。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="op_small_")
        self.info = build_small_corpus(self.tmp)
        self.inbox = self.info["inbox"]
        self.out = os.path.join(self.inbox, "_预处理输出")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self):
        return run(self.inbox, self.out, progress=lambda _m: None)

    def test_ac08_idempotent_resume_and_truncation(self):
        first = self._run()
        rows_one = read_jsonl(os.path.join(self.out, "manifest.jsonl"))
        hashes = {row["image_id"]: row["preview_sha256"] for row in rows_one}

        second = self._run()
        self.assertEqual(second["report"]["resume"]["reused"], 2)
        self.assertEqual(second["report"]["resume"]["reprocessed"], 0)
        rows_two = read_jsonl(os.path.join(self.out, "manifest.jsonl"))
        self.assertEqual({row["image_id"]: row["preview_sha256"] for row in rows_two}, hashes)
        self.assertEqual(second["report"]["input_file_count"], 2)

        # 截断预览 → 必须识别为不完整并重新生成
        row = next(r for r in rows_two if r["status"] == "ready")
        preview_path = os.path.join(self.out, row["preview_path"])
        with open(preview_path, "wb") as f:
            f.write(b"")
        third = self._run()
        self.assertGreaterEqual(third["report"]["resume"]["reprocessed"], 1)
        with Image.open(preview_path) as im:
            im.load()
            self.assertEqual(im.size[0] > 0, True)
        repaired = {r["image_id"]: r for r in read_jsonl(os.path.join(self.out, "manifest.jsonl"))}
        self.assertEqual(repaired[row["image_id"]]["preview_sha256"], sha256_file(preview_path))

        # 截断特征 JSON → 同样重新生成
        feature_path = os.path.join(self.out, repaired[row["image_id"]]["feature_path"])
        with open(feature_path, "w", encoding="utf-8") as f:
            f.write("{")
        fourth = self._run()
        self.assertGreaterEqual(fourth["report"]["resume"]["reprocessed"], 1)
        payload = read_json(feature_path)
        self.assertEqual(payload["image_id"], row["image_id"])

    def test_config_change_requires_new_dir_or_overwrite(self):
        self._run()
        with self.assertRaises(FatalError):
            run(self.inbox, self.out, overrides={"jpeg_quality": 80}, progress=lambda _m: None)
        result = run(
            self.inbox,
            self.out,
            overrides={"jpeg_quality": 80, "overwrite": True},
            progress=lambda _m: None,
        )
        self.assertEqual(result["report"]["status_counts"]["ready"], 2)

    def test_output_dir_equal_input_rejected(self):
        with self.assertRaises(FatalError):
            run(self.inbox, self.inbox, progress=lambda _m: None)

    def test_cli_run(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(
                ["--input-dir", self.inbox, "--output-dir", self.out, "--quiet", "--json"]
            )
        self.assertEqual(code, 0)
        summary = json.loads(buffer.getvalue())
        self.assertEqual(summary["status_counts"]["ready"], 2)


if __name__ == "__main__":
    unittest.main()

"""桶级语言描述测试。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from offline_profile.narrative import collect_bucket_descriptions, submit_bucket_narrative


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _analysis(image_id):
    return {
        "image_id": image_id,
        "scene_observation": {
            "lighting": "daylight",
            "brightness": "mid",
            "backlit": False,
            "subject": "a coastal headland",
            "environment_notes": "open sea and grass",
        },
        "visual_observation": {"contrast": "neutral", "warmth": "high"},
        "edit_hypotheses": {
            "highlights": {"evidence": "bright sky retains texture"},
            "blacks": {"evidence": "shadow floor is lifted"},
        },
        "quality": {"issues": ["soft_focus"]},
        "local_adjustments_suspected": [],
        "overall_impression": "A quiet coastal scene.",
        "warnings": [],
    }


class TestNarrative(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="narrative_")
        self.build = os.path.join(self.tmp, "build")
        ids = [f"img_{index:016x}" for index in range(1, 4)]
        _write(os.path.join(self.build, "profile.json"), {"build_id": "build_x"})
        _write(
            os.path.join(self.build, "buckets", "natural_landscape", "stats.json"),
            {"bucket_id": "natural_landscape", "members_train": ids},
        )
        _write(
            os.path.join(self.build, "assignments.jsonl"),
            [],
        )
        with open(os.path.join(self.build, "assignments.jsonl"), "w", encoding="utf-8") as handle:
            for image_id in ids:
                handle.write(json.dumps({
                    "image_id": image_id, "bucket_id": "natural_landscape", "level": "core"
                }, ensure_ascii=False) + "\n")
        os.makedirs(os.path.join(self.build, "mcp"), exist_ok=True)
        with open(os.path.join(self.build, "mcp", "tasks.jsonl"), "w", encoding="utf-8") as handle:
            for image_id in ids:
                rel = f"analyses/normalized/{image_id}.json"
                _write(os.path.join(self.build, rel), _analysis(image_id))
                handle.write(json.dumps({"image_id": image_id, "normalized_path": rel}, ensure_ascii=False) + "\n")
        self.ids = ids

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _payload(self):
        return {
            "language": "zh",
            "summary": "海岸高地与开阔海面反复出现，画面安静而克制。",
            "mood": "安静、开阔",
            "mood_keywords": ["海岸", "开阔", "克制"],
            "subject_context": "海岸草坡、礁石与海面构成主体，人物很少。",
            "lighting_context": "以漫射日光为主，天空保持浅色而不刺眼。",
            "color_language": "冷色海面与暖色草地形成温和对比。",
            "composition_tendencies": "前景草坡引导视线向海平线延伸。",
            "recurring_motifs": ["海岸草坡", "开阔海面"],
            "exceptions": ["少数逆光场景反差更高"],
            "cautions": ["不要套用到城市或室内场景"],
            "evidence_refs": self.ids,
        }

    def test_collect_and_submit(self):
        collected = collect_bucket_descriptions(self.build, "natural_landscape")
        self.assertEqual(collected["n"], 3)
        self.assertTrue(os.path.isfile(os.path.join(self.build, collected["descriptions_file"])))
        self.assertTrue(os.path.isfile(os.path.join(self.build, collected["prompt_file"])))
        narrative = submit_bucket_narrative(self.build, "natural_landscape", self._payload())
        self.assertEqual(narrative["status"], "draft")
        self.assertEqual(len(narrative["narrative_hash"]), 64)
        self.assertEqual(narrative["source"]["descriptions_hash"], collected["descriptions_hash"])

    def test_rejects_unknown_evidence(self):
        collect_bucket_descriptions(self.build, "natural_landscape")
        payload = self._payload()
        payload["evidence_refs"] = ["img_ffffffffffffffff"]
        with self.assertRaises(ValueError):
            submit_bucket_narrative(self.build, "natural_landscape", payload)


if __name__ == "__main__":
    unittest.main()

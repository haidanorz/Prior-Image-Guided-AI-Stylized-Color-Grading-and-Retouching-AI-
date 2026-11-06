from __future__ import annotations

import unittest

from online_decode import contracts


def brief(**overrides):
    data = {
        "scene_observation": {"primary_candidates": [{"label": "natural_landscape", "score": 0.9}]},
        "branch_rationale": "干旱地貌支系，暖色层理配灰白天光",
        "preserve": ["天空层次"],
        "goals": ["保留天空层次", "暗部不死黑"],
        "avoid": ["高光溢出"],
        "initial_xmp": "<x:xmpmeta/>",
    }
    data.update(overrides)
    return data


def plan(**overrides):
    data = {"xmp": "<x:xmpmeta/>", "expected_effects": ["更通透"]}
    data.update(overrides)
    return data


def evaluation():
    return {
        "style_alignment": {"verdict": "met"},
        "branch_alignment": {"verdict": "met"},
        "technical_quality": {"verdict": "met", "defects": []},
        "content_preservation": {"verdict": "met"},
        "improvement": "better",
        "decision": "continue",
        "evidence": [{"anchor": "upper_right", "observation": "云层层次保留"}],
    }


def request(**overrides):
    data = {
        "request_schema_version": "1.0.0",
        "task": "plan",
        "session_id": "s_1",
        "round": 1,
        "images": {"original": "a.jpg", "current": "b.jpg", "previous": None},
    }
    data.update(overrides)
    return data


class SchemaFileTest(unittest.TestCase):
    def test_all_schemas_are_valid_draft7(self):
        import jsonschema

        for name in contracts.ALL_SCHEMAS:
            with self.subTest(name=name):
                jsonschema.Draft7Validator.check_schema(contracts.load(name))


class ContractTest(unittest.TestCase):
    def test_valid_payloads_pass(self):
        for name, payload in (
            (contracts.REQUEST, request()),
            (contracts.BRIEF, brief()),
            (contracts.PLAN, plan()),
            (contracts.EVALUATION, evaluation()),
        ):
            with self.subTest(name=name):
                self.assertEqual(contracts.validate(name, payload), [])

    def test_brief_needs_initial_output(self):
        payload = brief()
        payload.pop("initial_xmp")
        self.assertTrue(contracts.validate(contracts.BRIEF, payload))

    def test_brief_rejects_too_few_goals(self):
        self.assertTrue(contracts.validate(contracts.BRIEF, brief(goals=["只写一条"])))

    def test_brief_rejects_bad_candidate_score(self):
        payload = brief(scene_observation={"primary_candidates": [{"label": "a", "score": 1.7}]})
        self.assertTrue(contracts.validate(contracts.BRIEF, payload))

    def test_plan_needs_xmp_or_adjustments_or_stop(self):
        self.assertTrue(contracts.validate(contracts.PLAN, {"expected_effects": ["x"]}))

    def test_plan_allows_explicit_stop(self):
        self.assertEqual(contracts.validate(contracts.PLAN, {"stop": True}), [])

    def test_plan_accepts_xmp_path(self):
        self.assertEqual(contracts.validate(contracts.PLAN, plan(xmp=None, xmp_path="r01.xmp")), [])

    def test_evaluation_needs_evidence(self):
        payload = evaluation()
        payload.pop("evidence")
        self.assertTrue(contracts.validate(contracts.EVALUATION, payload))

    def test_evaluation_rejects_bad_anchor(self):
        payload = evaluation()
        payload["evidence"][0]["anchor"] = "左上角"
        self.assertTrue(contracts.validate(contracts.EVALUATION, payload))

    def test_evaluation_rejects_freeform_verdict(self):
        payload = evaluation()
        payload["style_alignment"]["verdict"] = "pretty good"
        self.assertTrue(contracts.validate(contracts.EVALUATION, payload))

    def test_request_rejects_unknown_task(self):
        self.assertTrue(contracts.validate(contracts.REQUEST, request(task="fix_everything")))

    def test_request_round_over_five(self):
        self.assertTrue(contracts.validate(contracts.REQUEST, request(round=6)))

    def test_validate_or_raise_carries_schema_name(self):
        from online_decode.errors import Codes, DecodeError

        with self.assertRaises(DecodeError) as ctx:
            contracts.validate_or_raise(contracts.BRIEF, {}, stage="brief")
        self.assertEqual(ctx.exception.code, Codes.MODEL_RESPONSE_INVALID)
        self.assertEqual(ctx.exception.detail["schema"], contracts.BRIEF)


if __name__ == "__main__":
    unittest.main()

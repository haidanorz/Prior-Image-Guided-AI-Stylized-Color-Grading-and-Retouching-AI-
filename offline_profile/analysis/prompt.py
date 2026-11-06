"""英文分析提示词构造。

提示词只提供单图预览、客观特征摘要和固定契约；不提供其他照片的结论，
也不允许模型创造参数键。所有阈值与枚举来自 Schema/参数登记表。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..contract import edit_parameter_index, load_parameters, load_taxonomy, visual_attribute_levels
from ..intake import IntakeImage
from ..version import ANALYSIS_SCHEMA_VERSION, PROMPT_VERSION
from .objective import feature_digest, objective_brightness


@dataclass(frozen=True)
class PromptBundle:
    text: str
    sha256: str
    version: str


def build_prompt(image: IntakeImage, *, input_fingerprint: str, cfg: dict) -> PromptBundle:
    taxonomy = load_taxonomy()
    parameters = load_parameters()
    brightness = objective_brightness(image.feature, cfg)
    digest = json.dumps(feature_digest(image.feature), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    scene_lines = "\n".join(f"- {item['label']}: {item['description']}" for item in taxonomy["labels"])
    visual_lines = "\n".join(
        f"- {key}: {' | '.join(levels)}" for key, levels in visual_attribute_levels().items()
    )
    param_lines = "\n".join(
        f"- {item['key']} ({item['kind']}, {item['unit']}, {item['min']}..{item['max']}): {item['description']}"
        for item in parameters["edit_parameters"]
    )

    text = f"""You are a disciplined photography analyst. You receive ONE rendered photograph and its objective image statistics. Your job is to describe visible tendencies and plausible editing hypotheses for this single image only.

HARD RULES
1. Do not identify people, places or brands. Describe only visible content.
2. Never claim the photographer's actual settings. edit_hypotheses are hypotheses about what could produce the look, not facts.
3. Use only the registered parameter keys listed below. Never invent keys.
4. Every edit hypothesis must include estimate, range, confidence and visible evidence. Omit a key when there is no visible evidence.
5. Never output precise Kelvin values, curve points, mask geometry or local slider values.
6. Candidate scores express ranking only; they are not calibrated probabilities. Scores must be in [0,1], labels must be unique, and after normalization they must sum to 1.
7. The objective brightness class is fixed by measured luminance. Echo it exactly: {brightness}.
8. Numeric coverage is part of the contract: `edit_hypotheses` must not be shortened to make room for narrative. For every registered parameter with visible evidence, include an entry. A typical image should contain 6-10 edit hypotheses. Do not omit a supported parameter just to keep the response short.
9. Write `narrative_observation` in Simplified Chinese. It is a language layer, not a second set of measurements: describe mood, atmosphere, light character, sense of place, seasonal cues and compositional rhythm that are visible in this image. Keep each narrative field to one or two sentences.
10. Narrative text must be grounded in visible cues. Do not name locations, people or brands, do not infer camera settings, and never write numeric slider values or parameter units.
11. If a seasonal or place cue is not visible, write "unknown" instead of guessing.
12. Output ONE JSON object and nothing else. No markdown fences, no commentary.

IMAGE CONTEXT
image_id: {image.image_id}
source_sha256: {image.source_sha256}
input_fingerprint: {input_fingerprint}
analysis_schema_version: {ANALYSIS_SCHEMA_VERSION}
prompt_version: {PROMPT_VERSION}
objective_brightness: {brightness}

OBJECTIVE FEATURE DIGEST (authoritative measurements)
{digest}

SCENE TAXONOMY (scene_v1)
{scene_lines}

LIGHTING VALUES
daylight | golden_hour | overcast | artificial | night | mixed | unknown

VISUAL OBSERVATION KEYS (ordinal values)
{visual_lines}

EDIT PARAMETER REGISTRY
{param_lines}

REQUIRED OUTPUT SHAPE
{{
  "analysis_schema_version": "{ANALYSIS_SCHEMA_VERSION}",
  "image_id": "{image.image_id}",
  "source_sha256": "{image.source_sha256}",
  "input_fingerprint": "{input_fingerprint}",
  "model_info": {{"provider": "...", "model": "...", "version": "...", "is_mock": false}},
  "prompt_version": "{PROMPT_VERSION}",
  "scene_observation": {{
    "primary_candidates": [{{"label": "<taxonomy label>", "score": 0.0}}],
    "lighting": "<lighting value>",
    "subject": "<short visible subject>",
    "brightness": "{brightness}",
    "backlit": true | false | "unknown",
    "backlit_evidence": "<visible evidence or why unknown>",
    "environment_notes": "<short notes>"
  }},
  "visual_observation": {{ "<every visual key>": "<ordinal value>" }},
  "edit_hypotheses": {{
    "<registered parameter key>": {{"estimate": 0.0, "range": [0.0, 0.0], "confidence": 0.0, "evidence": "<visible evidence>"}}
  }},
  "narrative_observation": {{
    "mood": "<one sentence, Chinese>",
    "atmosphere": "<one or two sentences, Chinese>",
    "light_quality": "<visible light character, Chinese>",
    "sense_of_place": "<visible spatial character without naming a location>",
    "seasonal_cues": "<visible seasonal cues, or unknown>",
    "composition_rhythm": "<how the frame is organized and paced>",
    "keywords": ["<3-6 short Chinese keywords>"],
    "basis": "<the visible cues this narrative is based on>"
  }},
  "quality": {{"overall": "good|acceptable|poor|unknown", "issues": [], "confidence": 0.0}},
  "local_adjustments_suspected": [{{"region": "sky|subject|background|foreground|edges|mixed|unknown", "kind": "<short kind>", "evidence": "<visible evidence>", "confidence": 0.0}}],
  "overall_impression": "<one or two sentences>",
  "warnings": []
}}

Interpretation notes:
- brightness must equal the objective class above.
- For a circular hue parameter, the estimate is an angle in degrees [0,360).
- If the image is monochrome, say so in warnings instead of inventing color evidence.
- Keep the response compact; omit hypotheses you cannot support visually.
- `narrative_observation` is generated in the SAME response as the structured analysis; do not write a separate essay or repeat numeric estimates.
- Write `edit_hypotheses` first, then the narrative. Narrative must never replace a numeric hypothesis that has visible evidence.
- narrative keywords must be short, non-duplicated, and descriptive of the visible image, not of photography technique.
"""
    digest_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PromptBundle(text=text, sha256=digest_hash, version=PROMPT_VERSION)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["PromptBundle", "build_prompt", "prompt_hash", "edit_parameter_index"]

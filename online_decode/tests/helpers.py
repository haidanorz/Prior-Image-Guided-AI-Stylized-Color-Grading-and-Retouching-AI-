"""测试共用夹具。"""

from __future__ import annotations

# 一份"看起来正常"的 Develop 设置：包含白平衡与主要影调字段。
# 不追求 65 个字段全齐——FakeLrPort 的哈希只要求自洽，不要求与插件一致。
BASE_SETTINGS = {
    "Exposure2012": 0.0,
    "Contrast2012": 5,
    "Highlights2012": -10,
    "Shadows2012": 8,
    "Whites2012": 3,
    "Blacks2012": 12,
    "Texture": 0,
    "Clarity2012": 9,
    "Dehaze": 0,
    "Vibrance": 17,
    "Saturation": 2,
    "Temperature": 5500,
    "Tint": 8,
    "WhiteBalance": "Custom",
}


# 场景观测响应：bucket_primary 流程的第一步（只看原图、看不到桶参数）。
SCENE = {
    "scene_observation": {
        "primary_candidates": [
            {"label": "natural_landscape", "score": 0.85},
            {"label": "urban_architecture", "score": 0.10},
        ],
        "lighting": "soft",
        "brightness": "normal",
        "backlit": False,
    }
}

"""离线建档后处理（消费预处理产物，构建可验证的风格档案）。

本轮实现范围：契约层 + 单图分析层 + 场景分桶 + 训练/留出划分。
统计、验证发布与语言技能层在后续迭代加入。
"""

from .version import (
    ANALYSIS_SCHEMA_VERSION,
    PARAMETER_REGISTRY_VERSION,
    PROMPT_VERSION,
    TAXONOMY_VERSION,
)

__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "PROMPT_VERSION",
    "TAXONOMY_VERSION",
    "PARAMETER_REGISTRY_VERSION",
]

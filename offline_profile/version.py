"""版本常量。任何影响契约或统计的变更都必须提升对应版本。"""

ANALYSIS_SCHEMA_VERSION = "1.1.0"
PROMPT_VERSION = "scene-analysis-v2.1"
TAXONOMY_VERSION = "scene_v1"
PARAMETER_REGISTRY_VERSION = "1.0.0"

# 后续统计/发布层使用，本轮先固定，避免下游临时猜版本
PROFILE_SCHEMA_VERSION = "1.0.0"
SKILL_SCHEMA_VERSION = "1.0.0"
AGGREGATION_VERSION = "1.0.0"

VALIDATION_SCOPE = "analysis_consistency"
APPLICATION_READY = False

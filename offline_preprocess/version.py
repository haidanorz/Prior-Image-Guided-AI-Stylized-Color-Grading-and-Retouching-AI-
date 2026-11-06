"""版本常量。

preprocess_version      预处理实现版本，代码语义变化必须提升。
feature_schema_version  特征 JSON 的 Schema 版本，任何统计口径/字段变化必须提升。
"""

PREPROCESS_VERSION = "1.0.0"
FEATURE_SCHEMA_VERSION = "1.0.0"

# 几何取整规则版本（floor(x + 0.5)，不接受银行家舍入）
ROUNDING_RULE_VERSION = "floor_x_plus_0.5_v1"

# 重采样滤波器标识（仅用于记录，实际由 Pillow 实现）
RESAMPLE_FILTER = "lanczos"

# 统计来源固定说明，写入每份特征 JSON
STATISTICS_SOURCE = "normalized_srgb_resized_before_jpeg"

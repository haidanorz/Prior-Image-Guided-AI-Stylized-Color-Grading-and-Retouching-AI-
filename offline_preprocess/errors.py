"""稳定错误码、处理阶段与异常类型。

错误码是下游数据契约的一部分：只能新增，不得改名或改变语义。
"""

from __future__ import annotations


class Stage:
    """处理阶段标识，写入 errors.jsonl 与 run_report。"""

    SCAN = "scan"
    HASH = "hash"
    DECODE = "decode"
    ORIENT = "orient"
    COLOR = "color"
    RESIZE = "resize"
    FEATURES = "features"
    ENCODE = "encode"
    WRITE = "write"
    VERIFY = "verify"
    REPORT = "report"


class ErrorCode:
    """稳定错误码。"""

    READ_FAILED = "READ_FAILED"
    HASH_FAILED = "HASH_FAILED"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    DECODE_FAILED = "DECODE_FAILED"
    MULTI_FRAME_UNSUPPORTED = "MULTI_FRAME_UNSUPPORTED"
    ORIENTATION_FAILED = "ORIENTATION_FAILED"
    ICC_CORRUPT = "ICC_CORRUPT"
    ICC_TRANSFORM_FAILED = "ICC_TRANSFORM_FAILED"
    ICC_ENGINE_UNAVAILABLE = "ICC_ENGINE_UNAVAILABLE"
    ICC_REQUIRED_FOR_MODE = "ICC_REQUIRED_FOR_MODE"
    UNSUPPORTED_COLOR_MODE = "UNSUPPORTED_COLOR_MODE"
    RESIZE_FAILED = "RESIZE_FAILED"
    FEATURE_FAILED = "FEATURE_FAILED"
    ENCODE_FAILED = "ENCODE_FAILED"
    WRITE_FAILED = "WRITE_FAILED"
    VERIFY_FAILED = "VERIFY_FAILED"
    ID_COLLISION = "ID_COLLISION"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# 致命错误（整批中止，不写入任何单图记录）
class FatalCode:
    CONFIG_INVALID = "CONFIG_INVALID"
    INPUT_DIR_INVALID = "INPUT_DIR_INVALID"
    OUTPUT_DIR_INVALID = "OUTPUT_DIR_INVALID"
    OUTPUT_STATE_INCOMPATIBLE = "OUTPUT_STATE_INCOMPATIBLE"


class SkipReason:
    UNSUPPORTED_ALPHA = "unsupported_alpha"
    TOO_SMALL = "too_small"


class Status:
    READY = "ready"
    DUPLICATE_EXACT = "duplicate_exact"
    SKIPPED = "skipped"
    ERROR = "error"

    ALL = (READY, DUPLICATE_EXACT, SKIPPED, ERROR)


class WarningCode:
    MISSING_ICC_ASSUMED_SRGB = "missing_icc_assumed_srgb"
    EXTREME_RESIZE_ROUNDING = "extreme_resize_rounding"
    NO_CHROMATIC_PIXELS = "no_chromatic_pixels"
    INVALID_ORIENTATION = "invalid_orientation_defaulted"
    EXIF_UNREADABLE = "exif_unreadable"
    ICC_DESCRIPTION_UNREADABLE = "icc_description_unreadable"
    HIGH_BIT_DEPTH_SCALED = "high_bit_depth_scaled"


class PreprocessError(Exception):
    """单图处理失败，携带稳定错误码与阶段。"""

    def __init__(self, code: str, stage: str, message: str, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.message = message
        self.detail = detail


class FatalError(Exception):
    """整批任务失败（配置、目录、输出状态不兼容）。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class SkipImage(Exception):
    """按规则跳过，不是错误。"""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message

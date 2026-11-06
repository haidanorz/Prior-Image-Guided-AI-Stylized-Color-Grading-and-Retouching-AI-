"""稳定错误码、阶段与异常类型。

分析失败按单图隔离；只有输入/配置/输出目录层面的问题才整批中止。
"""

from __future__ import annotations


class Stage:
    INTAKE = "intake"
    ANALYZE = "analyze"
    NORMALIZE = "normalize"
    ASSIGN = "assign"
    SPLIT = "split"
    WRITE = "write"
    VERIFY = "verify"


class ErrorCode:
    # 输入
    INPUT_MANIFEST_MISSING = "INPUT_MANIFEST_MISSING"
    INPUT_MANIFEST_EMPTY = "INPUT_MANIFEST_EMPTY"
    INPUT_VERSION_MIXED = "INPUT_VERSION_MIXED"
    INPUT_RECORD_INVALID = "INPUT_RECORD_INVALID"
    INPUT_PREVIEW_MISSING = "INPUT_PREVIEW_MISSING"
    INPUT_PREVIEW_HASH_MISMATCH = "INPUT_PREVIEW_HASH_MISMATCH"
    INPUT_FEATURE_MISSING = "INPUT_FEATURE_MISSING"
    INPUT_FEATURE_INVALID = "INPUT_FEATURE_INVALID"
    INPUT_NO_READY = "INPUT_NO_READY"
    INPUT_BUILD_OVERLAP = "INPUT_BUILD_OVERLAP"

    # 单图分析
    ANALYSIS_TRANSPORT_ERROR = "ANALYSIS_TRANSPORT_ERROR"
    ANALYSIS_FORMAT_ERROR = "ANALYSIS_FORMAT_ERROR"
    ANALYSIS_INVALID = "ANALYSIS_INVALID"
    ANALYSIS_ID_MISMATCH = "ANALYSIS_ID_MISMATCH"
    MODEL_NOT_CONFIGURED = "MODEL_NOT_CONFIGURED"

    # 基础设施
    CONFIG_INVALID = "CONFIG_INVALID"
    WRITE_FAILED = "WRITE_FAILED"
    CACHE_CORRUPT = "CACHE_CORRUPT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class FatalCode:
    CONFIG_INVALID = "CONFIG_INVALID"
    INPUT_INVALID = "INPUT_INVALID"
    BUILD_ROOT_INVALID = "BUILD_ROOT_INVALID"
    BUILD_STATE_INCOMPATIBLE = "BUILD_STATE_INCOMPATIBLE"


class AnalysisStatus:
    NORMALIZED = "normalized"
    INVALID = "invalid"
    ERROR = "error"
    CACHED = "cached"  # 结果来自缓存，语义上仍是 normalized


class EpisodeType:
    ANALYSIS_RETRY = "analysis_retry"
    ANALYSIS_INVALID = "analysis_invalid"
    ANALYSIS_ERROR = "analysis_error"
    ANALYSIS_RECOVERED = "analysis_recovered"


class FatalError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class AnalysisFailure(Exception):
    """单图分析失败（可隔离，不终止批处理）。"""

    def __init__(self, code: str, stage: str, message: str, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.message = message
        self.detail = detail


class AdapterError(Exception):
    """模型适配器层面的传输/格式错误，可重试。"""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class InvalidAnalysis(Exception):
    """规范化阶段判定响应非法；details 为可读问题列表。"""

    def __init__(self, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or []

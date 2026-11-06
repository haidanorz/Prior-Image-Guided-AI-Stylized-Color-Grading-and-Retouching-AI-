"""离线建档预处理（独立模块，不依赖任何旧 Demo / 插件 / 队列）。"""

from .version import FEATURE_SCHEMA_VERSION, PREPROCESS_VERSION

__all__ = ["PREPROCESS_VERSION", "FEATURE_SCHEMA_VERSION"]
__version__ = PREPROCESS_VERSION

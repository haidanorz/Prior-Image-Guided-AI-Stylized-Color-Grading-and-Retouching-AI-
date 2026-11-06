"""通用工具：哈希、原子写入、JSON 规范化、数值清洗。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

_CHUNK = 1024 * 1024


def now_utc_iso() -> str:
    """UTC ISO8601 时间戳（秒精度，Z 结尾）。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(path: str, data: bytes) -> None:
    """同目录临时文件写完 fsync 后原子替换，避免半写文件。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def dump_json_text(obj: Any, *, pretty: bool = True) -> str:
    """输出 UTF-8 友好的 JSON 文本；禁止 NaN/Infinity（会直接抛错）。"""
    payload = to_jsonable(obj)
    if pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def atomic_write_json(path: str, obj: Any, *, pretty: bool = True) -> None:
    atomic_write_bytes(path, dump_json_text(obj, pretty=pretty).encode("utf-8"))


def canonical_json(obj: Any) -> str:
    """键排序、无空白的规范化 JSON，用于计算配置哈希。"""
    return json.dumps(to_jsonable(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def round_float(value: float, digits: int = 6) -> float:
    """保留至多 digits 位小数；拒绝 NaN/Infinity，归一 -0.0。"""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"non-finite float in output: {v!r}")
    r = round(v, digits)
    return 0.0 if r == 0 else r


def to_jsonable(obj: Any) -> Any:
    """递归转成可 JSON 序列化的原生类型，并清洗浮点。"""
    import numpy as np  # 延迟导入，便于工具脚本在无 numpy 时仍可做纯文本操作

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round_float(float(obj))
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return round_float(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    # 兜底：numpy 数组等不可直接序列化的类型应当在上游转好
    raise TypeError(f"unsupported JSON value type: {type(obj)!r}")

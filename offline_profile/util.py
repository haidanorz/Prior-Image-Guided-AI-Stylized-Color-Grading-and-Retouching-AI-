"""通用工具：哈希、原子写入、规范 JSON、JSONL 追加。"""

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
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: str, data: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float in output: {obj!r}")
        rounded = round(obj, 6)
        return 0.0 if rounded == 0 else rounded
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    raise TypeError(f"unsupported JSON value type: {type(obj)!r}")


def dump_json_text(obj: Any, *, pretty: bool = True) -> str:
    payload = to_jsonable(obj)
    if pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def atomic_write_json(path: str, obj: Any, *, pretty: bool = True) -> None:
    atomic_write_bytes(path, dump_json_text(obj, pretty=pretty).encode("utf-8"))


def canonical_json(obj: Any) -> str:
    return json.dumps(
        to_jsonable(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def canonical_hash(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, rows: list[dict]) -> None:
    """追加写入并 fsync；调用方负责保证只追加、不覆盖。"""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = "".join(dump_json_text(row, pretty=False) for row in rows)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_atomic(path: str, rows: list[dict]) -> None:
    text = "".join(dump_json_text(row, pretty=False) for row in rows)
    atomic_write_bytes(path, text.encode("utf-8"))


def relative_posix(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")

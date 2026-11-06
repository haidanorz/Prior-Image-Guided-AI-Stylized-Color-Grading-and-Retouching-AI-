"""小工具：原子写、哈希、时间戳、路径。

工程纪律（项目里踩过）：不要 `open(path,"w")` 就地改写 —— 先写 .new 再 os.replace，
否则中途报错会留下半个 JSON，后续读取方会把"半截"当成"空"。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp = tempfile.mkstemp(prefix=target.name + ".", suffix=".new", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    return atomic_write_bytes(path, dumps(payload).encode("utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def prefixed(hash_hex: str) -> str:
    return hash_hex if hash_hex.startswith("sha256:") else "sha256:" + hash_hex


def is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

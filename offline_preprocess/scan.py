"""输入目录枚举。

只按扩展名挑选候选文件，真正的格式判定留到解码阶段（扩展名不可信）。
不跟随符号链接/目录联接；输出目录若位于输入目录内则整棵子树被排除。
"""

from __future__ import annotations

import os
from typing import Iterable

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def normalize_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def _norm_key(path: str) -> str:
    return os.path.normcase(normalize_path(path))


def is_within(child: str, parent: str) -> bool:
    """child 是否等于 parent 或位于其下（大小写不敏感，适配 Windows）。"""
    child_key = _norm_key(child)
    parent_key = _norm_key(parent)
    if child_key == parent_key:
        return True
    if not parent_key.endswith(os.sep):
        parent_key += os.sep
    return child_key.startswith(parent_key)


def scan_inputs(input_dir: str, recursive: bool, exclude_dirs: Iterable[str] = ()) -> tuple[list[str], list[dict]]:
    """返回 (候选文件绝对路径列表, 目录读取问题列表)，顺序稳定。"""
    excluded = [_norm_key(d) for d in exclude_dirs]
    root = normalize_path(input_dir)
    files: list[str] = []
    issues: list[dict] = []

    if not recursive:
        with os.scandir(root) as it:
            for entry in it:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except OSError as exc:
                    issues.append({"path": entry.path, "message": str(exc)})
                    continue
                if _is_supported(entry.name):
                    files.append(normalize_path(entry.path))
        files.sort(key=_sort_key)
        return files, issues

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False, onerror=None):
        kept: list[str] = []
        for name in sorted(dirnames, key=str.lower):
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                continue
            key = _norm_key(full)
            if any(key == ex or key.startswith(ex + os.sep) for ex in excluded):
                continue
            kept.append(name)
        dirnames[:] = kept

        for name in filenames:
            if not _is_supported(name):
                continue
            full = os.path.join(dirpath, name)
            try:
                if not os.path.isfile(full) or os.path.islink(full):
                    continue
            except OSError as exc:
                issues.append({"path": full, "message": str(exc)})
                continue
            files.append(normalize_path(full))

    files.sort(key=_sort_key)
    return files, issues


def _is_supported(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS


def _sort_key(path: str) -> tuple[str, str]:
    # 大小写不敏感优先，再用原始字符串打破平局，保证跨平台稳定
    return (path.lower(), path)

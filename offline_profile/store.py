"""构建目录与原子写入。"""

from __future__ import annotations

import os

from .util import atomic_write_json


class BuildStore:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.raw_dir = os.path.join(self.root, "analyses", "raw")
        self.normalized_dir = os.path.join(self.root, "analyses", "normalized")
        self.cache_dir = os.path.join(self.root, "cache")
        for path in (self.raw_dir, self.normalized_dir, self.cache_dir):
            os.makedirs(path, exist_ok=True)

    def write_raw(self, image_id: str, payload: dict) -> str:
        path = os.path.join(self.raw_dir, f"{image_id}.json")
        atomic_write_json(path, payload)
        return path

    def write_normalized(self, image_id: str, payload: dict) -> str:
        path = os.path.join(self.normalized_dir, f"{image_id}.json")
        atomic_write_json(path, payload)
        return path

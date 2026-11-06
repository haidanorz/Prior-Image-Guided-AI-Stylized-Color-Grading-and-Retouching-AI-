"""主色板提取：Lab 空间 k-means（固定种子 + k-means++ 初始化）。

随机性完全由配置里的 random_seed 决定；同一输入、同一 numpy 版本下结果稳定。
主色比例为全图像素占比，不是采样占比。
"""

from __future__ import annotations

import numpy as np

from .color import lab_to_srgb8

_ALGORITHM = "lab_kmeans_v1"


def extract_palette(
    lab: np.ndarray, k: int, seed: int, max_sample: int, max_iter: int
) -> list[dict]:
    """lab: (N,3) float64；返回按占比降序的主色列表。"""
    pixels = np.asarray(lab, dtype=np.float64)
    n = pixels.shape[0]
    if n == 0 or k < 1:
        return []

    rng = np.random.RandomState(seed)
    if n > max_sample:
        idx = rng.choice(n, size=max_sample, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    unique_count = _unique_count(sample)
    k_eff = int(min(k, unique_count))
    if k_eff < 1:
        return []

    centers = _kmeans_pp(sample, k_eff, rng)
    centers = _lloyd(sample, centers, max_iter)

    counts = _assign_counts(pixels, centers)
    total = int(counts.sum())
    order = np.argsort(-counts, kind="stable")

    entries: list[dict] = []
    merged: dict[str, dict] = {}
    for cluster in order:
        ratio = float(counts[cluster]) / float(total) if total else 0.0
        rgb = lab_to_srgb8(centers[cluster])
        hex_value = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        if hex_value in merged:
            merged[hex_value]["ratio"] = float(merged[hex_value]["ratio"] + ratio)
            continue
        entry = {
            "hex": hex_value,
            "ratio": ratio,
            "lab": [float(centers[cluster][0]), float(centers[cluster][1]), float(centers[cluster][2])],
        }
        merged[hex_value] = entry
        entries.append(entry)

    entries.sort(key=lambda item: (-item["ratio"], item["hex"]))
    return entries


def palette_algorithm() -> str:
    return _ALGORITHM


def _unique_count(sample: np.ndarray) -> int:
    if sample.size == 0:
        return 0
    return int(np.unique(sample, axis=0).shape[0])


def _sq_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """(N,K) 平方欧氏距离；用矩阵乘代替 N×K×3 广播，降低内存与耗时。"""
    p2 = np.einsum("ij,ij->i", points, points)[:, None]
    c2 = np.einsum("ij,ij->i", centers, centers)[None, :]
    dist = p2 + c2 - 2.0 * (points @ centers.T)
    return np.maximum(dist, 0.0)


def _kmeans_pp(sample: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    centers = [sample[rng.randint(sample.shape[0])]]
    closest = _sq_distances(sample, np.asarray(centers))[:, 0]
    for _ in range(1, k):
        total = float(closest.sum())
        if total <= 0:
            break
        probs = closest / total
        index = int(rng.choice(sample.shape[0], p=probs))
        centers.append(sample[index])
        new_dist = _sq_distances(sample, sample[index : index + 1])[:, 0]
        closest = np.minimum(closest, new_dist)
    return np.asarray(centers, dtype=np.float64)


def _lloyd(sample: np.ndarray, centers: np.ndarray, max_iter: int) -> np.ndarray:
    centers = centers.copy()
    for _ in range(max_iter):
        assign = np.argmin(_sq_distances(sample, centers), axis=1)
        new_centers = centers.copy()
        for cluster in range(centers.shape[0]):
            members = sample[assign == cluster]
            if members.shape[0] == 0:
                continue
            new_centers[cluster] = members.mean(axis=0)
        shift = float(np.max(np.abs(new_centers - centers)))
        centers = new_centers
        if shift < 1e-4:
            break
    return centers


def _assign_counts(pixels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    counts = np.zeros(centers.shape[0], dtype=np.int64)
    step = 200_000
    for start in range(0, pixels.shape[0], step):
        block = pixels[start : start + step]
        assign = np.argmin(_sq_distances(block, centers), axis=1)
        counts += np.bincount(assign, minlength=centers.shape[0]).astype(np.int64)
    return counts

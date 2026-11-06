"""桶内验证划分：K 折交叉验证（默认 10 折）或单次留出。

两种模式都只使用 core 成员、按哈希排序、近重复组整体落在一折（或同侧）。
划分不参考聚合结果；无法形成有效折时标记原因，不降低门槛。
"""

from __future__ import annotations

import hashlib
import math

from .util import canonical_hash
from .version import TAXONOMY_VERSION


def build_evaluation_split(
    assignments: list[dict],
    near_duplicate_groups: list[list[str]],
    cfg: dict,
) -> dict:
    core_by_bucket: dict[str, list[str]] = {}
    for item in assignments:
        if item["level"] != "core":
            continue
        core_by_bucket.setdefault(item["bucket_id"], []).append(item["image_id"])
    for ids in core_by_bucket.values():
        ids.sort()

    components = _near_duplicate_components(near_duplicate_groups, set(sum(core_by_bucket.values(), [])))
    cross_bucket = _cross_bucket_components(components, core_by_bucket)
    mode = str(cfg.get("validation_mode", "holdout"))
    cv_folds = int(cfg.get("cv_folds", 10))

    buckets: dict[str, dict] = {}
    for bucket_id in sorted(core_by_bucket):
        ids = core_by_bucket[bucket_id]
        n_core = len(ids)
        status = _bucket_status(n_core, cfg)
        entry = {
            "bucket_id": bucket_id,
            "status": status,
            "n_core": n_core,
            "n_train": 0,
            "n_holdout": 0,
            "holdout_target": 0,
            "train": [],
            "holdout": [],
            "folds": [],
            "n_folds": 0,
            "fold_sizes": [],
            "near_duplicate_units": [],
            "split_status": status if status != "full" else "no_split",
            "split_hash": None,
            "reason": None,
        }

        if status == "exception_group":
            entry["reason"] = "below_provisional_min_core"
        elif status == "provisional":
            entry["reason"] = "provisional_bucket_no_holdout"
        elif bucket_id in cross_bucket:
            entry["split_status"] = "blocked"
            entry["reason"] = "cross_bucket_near_duplicate"
        elif mode == "kfold":
            units = _bucket_units(ids, components)
            folds = _assign_folds(units, cv_folds, int(cfg["split_seed"]))
            if len(folds) < 2:
                entry["split_status"] = "insufficient_folds"
                entry["reason"] = "too_few_units"
            else:
                entry.update(
                    {
                        "n_train": n_core,
                        "n_holdout": 0,
                        "train": list(ids),
                        "holdout": [],
                        "folds": folds,
                        "n_folds": len(folds),
                        "fold_sizes": [len(fold) for fold in folds],
                        "near_duplicate_units": [sorted(unit) for unit in units],
                        "split_status": "kfold",
                        "reason": None,
                    }
                )
                entry["split_hash"] = _fold_hash(bucket_id, folds, units, cfg)
        else:
            holdout_target = _holdout_target(n_core, float(cfg["holdout_ratio"]))
            entry["holdout_target"] = holdout_target
            if n_core < 10:
                entry["split_status"] = "insufficient_holdout"
                entry["reason"] = "n_core_below_10"
            else:
                units = _bucket_units(ids, components)
                chosen = _choose_units(units, holdout_target, int(cfg["split_seed"]))
                holdout = sorted(member for unit in chosen for member in unit)
                if len(holdout) == 0:
                    entry["split_status"] = "insufficient_holdout"
                    entry["reason"] = "no_unit_selected"
                elif n_core - len(holdout) < 8:
                    entry["split_status"] = "insufficient_holdout"
                    entry["reason"] = "holdout_would_leave_too_few_training"
                else:
                    training = sorted(set(ids) - set(holdout))
                    entry.update(
                        {
                            "n_train": len(training),
                            "n_holdout": len(holdout),
                            "train": training,
                            "holdout": holdout,
                            "near_duplicate_units": [sorted(unit) for unit in chosen],
                            "split_status": "split",
                            "reason": None,
                        }
                    )
                    entry["split_hash"] = _split_hash(
                        bucket_id, training, holdout, [sorted(unit) for unit in units], cfg
                    )

        buckets[bucket_id] = entry

    # 诊断折：仅用于只读评估，不进入训练集/全局基线
    for bucket_id, entry in (buckets.items() if mode == "kfold" else []):
        if entry["split_status"] in ("blocked", "kfold"):
            continue
        if entry["n_core"] < 3:
            continue
        units = _bucket_units(core_by_bucket[bucket_id], components)
        folds = _assign_folds(units, cv_folds, int(cfg["split_seed"]))
        if len(folds) < 2:
            continue
        entry["diagnostic_folds"] = folds
        entry["diagnostic_n_folds"] = len(folds)
        entry["diagnostic_fold_sizes"] = [len(fold) for fold in folds]
        entry["diagnostic_fold_hash"] = _fold_hash(bucket_id, folds, units, cfg)

    result = {
        "taxonomy_version": TAXONOMY_VERSION,
        "validation_mode": mode,
        "cv_folds": cv_folds if mode == "kfold" else None,
        "split_seed": int(cfg["split_seed"]),
        "holdout_ratio": float(cfg["holdout_ratio"]),
        "near_duplicate_components": [sorted(unit) for unit in components],
        "buckets": buckets,
    }
    result["split_hash"] = split_hash_of(result)
    return result


def split_hash_of(split: dict) -> str:
    """重算划分哈希；构建与 verify 共用同一函数，避免两处口径漂移。"""
    mode = split.get("validation_mode", "holdout")
    buckets = split.get("buckets", {})
    if mode == "kfold":
        payload = {
            "validation_mode": "kfold",
            "taxonomy_version": split.get("taxonomy_version"),
            "split_seed": split.get("split_seed"),
            "cv_folds": split.get("cv_folds"),
            "buckets": {
                key: {
                    "split_hash": value.get("split_hash"),
                    "train": value.get("train"),
                    "holdout": value.get("holdout", []),
                    "folds": value.get("folds", []),
                    "diagnostic_folds": value.get("diagnostic_folds", []),
                    "split_status": value.get("split_status"),
                }
                for key, value in buckets.items()
            },
        }
        return canonical_hash(payload)
    return canonical_hash(
        {
            "taxonomy_version": split.get("taxonomy_version"),
            "split_seed": split.get("split_seed"),
            "holdout_ratio": split.get("holdout_ratio"),
            "buckets": {
                key: {
                    "split_hash": value.get("split_hash"),
                    "train": value.get("train"),
                    "holdout": value.get("holdout"),
                    "split_status": value.get("split_status"),
                }
                for key, value in buckets.items()
            },
        }
    )


def _bucket_status(n_core: int, cfg: dict) -> str:
    if n_core >= int(cfg["full_bucket_min_core"]):
        return "full"
    if n_core >= int(cfg["provisional_bucket_min_core"]):
        return "provisional"
    return "exception_group"


def _holdout_target(n_core: int, ratio: float) -> int:
    return max(1, int(math.floor(n_core * ratio + 0.5)))


def _near_duplicate_components(groups: list[list[str]], core_ids: set[str]) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for group in groups:
        members = [member for member in group if member in core_ids]
        for member in members:
            find(member)
        for member in members[1:]:
            union(members[0], member)

    components: dict[str, list[str]] = {}
    for item in parent:
        components.setdefault(find(item), []).append(item)
    return [sorted(members) for members in components.values() if len(members) >= 2]


def _cross_bucket_components(components: list[list[str]], core_by_bucket: dict[str, list[str]]) -> set[str]:
    bucket_of = {image_id: bucket for bucket, ids in core_by_bucket.items() for image_id in ids}
    blocked: set[str] = set()
    for component in components:
        buckets = {bucket_of[image_id] for image_id in component}
        if len(buckets) > 1:
            blocked.update(buckets)
    return blocked


def _bucket_units(bucket_ids: list[str], components: list[list[str]]) -> list[list[str]]:
    """桶内单位 = 与桶相交的近重复组（整组）+ 未覆盖的单例。"""
    bucket_set = set(bucket_ids)
    units = [sorted(set(component) & bucket_set) for component in components if set(component) & bucket_set]
    covered = {image_id for unit in units for image_id in unit}
    units.extend([image_id] for image_id in bucket_ids if image_id not in covered)
    return units


def _choose_units(units: list[list[str]], target: int, seed: int) -> list[list[str]]:
    ordered = sorted(
        units,
        key=lambda unit: hashlib.sha256(f"{seed}|{'|'.join(sorted(unit))}".encode("utf-8")).hexdigest(),
    )
    chosen: list[list[str]] = []
    count = 0
    for unit in ordered:
        if count >= target:
            break
        chosen.append(unit)
        count += len(unit)
    return chosen


def _assign_folds(units: list[list[str]], k: int, seed: int) -> list[list[str]]:
    """按哈希排序后贪心均衡分折；近重复组整组进同一折。"""
    ordered = sorted(
        units,
        key=lambda unit: hashlib.sha256(
            f"{seed}|fold|{'|'.join(sorted(unit))}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ordered) < 2:
        return []
    k = max(2, min(k, len(ordered)))
    folds: list[list[str]] = [[] for _ in range(k)]
    for unit in ordered:
        target = min(range(k), key=lambda index: (len(folds[index]), index))
        folds[target].extend(unit)
    return [sorted(fold) for fold in folds]


def _fold_hash(bucket_id: str, folds: list[list[str]], units: list[list[str]], cfg: dict) -> str:
    return canonical_hash(
        {
            "bucket_id": bucket_id,
            "taxonomy_version": TAXONOMY_VERSION,
            "validation_mode": "kfold",
            "cv_folds": int(cfg.get("cv_folds", 10)),
            "split_seed": int(cfg["split_seed"]),
            "folds": [sorted(fold) for fold in folds],
            "units": [sorted(unit) for unit in units],
        }
    )


def _split_hash(bucket_id: str, train: list[str], holdout: list[str], units: list[list[str]], cfg: dict) -> str:
    return canonical_hash(
        {
            "bucket_id": bucket_id,
            "taxonomy_version": TAXONOMY_VERSION,
            "split_seed": int(cfg["split_seed"]),
            "holdout_ratio": float(cfg["holdout_ratio"]),
            "train": sorted(train),
            "holdout": sorted(holdout),
            "units": [sorted(unit) for unit in units],
        }
    )

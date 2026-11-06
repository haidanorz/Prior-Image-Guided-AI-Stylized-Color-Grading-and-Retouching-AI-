"""ProfileComposer：一个全局签名 + 每个桶的增量。

依据《离线建档后处理架构设计》第 9–11 节：

- 桶内稳健统计只用训练侧 core 成员（split 决定）；
- 全局基线按桶可靠性归一化，禁止大桶支配（w_ib = q_i × confidence_i × B_b / n_b，B_b = min(1, n_b/8)）；
- 桶档案只保存相对全局的 delta，不复制整套签名；
- 提供 merge(global, delta) 与留出集上「仅全局 / 全局+增量」的对比。

输出状态：profile.status=draft，桶增量 status=candidate；`published` 必须等
§13 的归一化误差 / P_bucket 全集 / Bootstrap / 留出充分性门槛通过后才可写。
加权中位数在累计权重恰好等于一半时取两侧中点，避免两个等权桶把全局拉向低值桶。

关于「全局签名」的两层含义（本实现的口径）：
每个可测参数都给出一个全局基线 central，作为该参数的增量零点；
但只有跨桶一致（同向一致率、加权一致性、桶心离散度均达标）的参数才标记
cross_scene_invariant=true。标记为 false 的参数强度随场景变化，禁止单独套用，
必须叠加桶增量。

用法：
    python -m offline_profile.profile_compose <build_root>
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from typing import Any, Iterable

from .version import PROFILE_SCHEMA_VERSION

CONF_MIN_DEFAULT = 0.30
CORE_FULL_MIN = 8
STABLE_N_VALID = 5
STABLE_SUPPORT = 0.50
STABLE_CONSISTENCY = 0.60
GLOBAL_MIN_BUCKETS = 2
GLOBAL_SIGN_AGREEMENT = 0.75
GLOBAL_WEIGHTED_CONSISTENCY = 0.70
HOLDOUT_MIN = 5
PARAM_MIN_VALID = 5
KEY_PARAMETERS = ("contrast", "highlights", "shadows", "blacks", "clarity", "saturation")


# ---------------------------------------------------------------- io helpers
def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _canonical_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
    os.replace(tmp, path)


# ------------------------------------------------------------ robust numeric
def weighted_median(pairs: Iterable[tuple[float, float]]) -> float | None:
    """加权中位数。半权重恰好落在两个样本之间时取两者中点。

    取中点而不是直接取靠前者，是为了避免「两个等权桶」这种常见情形
    把全局基线系统性地拉向数值较小的那个桶（否则全局基线会变成大桶/低值桶的复读）。
    """
    data = sorted((float(v), max(float(w), 0.0)) for v, w in pairs)
    data = [(v, w) for v, w in data if w > 0]
    if not data:
        return None
    half = sum(w for _, w in data) / 2.0
    acc = 0.0
    for index, (value, weight) in enumerate(data):
        acc += weight
        if acc > half + 1e-12:
            return value
        if abs(acc - half) <= 1e-12:
            if index + 1 < len(data):
                return round((value + data[index + 1][0]) / 2.0, 4)
            return value
    return data[-1][0]


def weighted_quantile(pairs: Iterable[tuple[float, float]], q: float) -> float | None:
    data = sorted((float(v), max(float(w), 0.0)) for v, w in pairs)
    data = [(v, w) for v, w in data if w > 0]
    if not data:
        return None
    target = sum(w for _, w in data) * q
    acc = 0.0
    for value, weight in data:
        acc += weight
        if acc >= target - 1e-12:
            return value
    return data[-1][0]


def weighted_mad(pairs: Iterable[tuple[float, float]], center: float | None) -> float | None:
    if center is None:
        return None
    return weighted_median((abs(v - center), w) for v, w in pairs)


def circular_median(pairs: Iterable[tuple[float, float]]) -> float | None:
    """加权圆形中位数：候选角度中取加权角距之和最小者（跨 0° 用最短弧）。"""
    data = [(math.radians(float(v) % 360.0), max(float(w), 0.0)) for v, w in pairs]
    data = [(a, w) for a, w in data if w > 0]
    if not data:
        return None

    def cost(candidate: float) -> float:
        return sum(
            abs((angle - candidate + math.pi) % (2 * math.pi) - math.pi) * weight
            for angle, weight in data
        )

    best = min((a for a, _ in data), key=cost)
    return round(math.degrees(best) % 360.0, 4)


def circular_dispersion(pairs: Iterable[tuple[float, float]]) -> float | None:
    """加权圆形离散度（mean resultant length，0..1；越大越集中）。"""
    data = [(math.radians(float(v) % 360.0), max(float(w), 0.0)) for v, w in pairs]
    data = [(a, w) for a, w in data if w > 0]
    total = sum(w for _, w in data)
    if total <= 0:
        return None
    x = sum(math.cos(a) * w for a, w in data) / total
    y = sum(math.sin(a) * w for a, w in data) / total
    return round(math.hypot(x, y), 4)


def circular_delta(a: float, b: float) -> float:
    diff = (float(a) - float(b)) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return round(diff, 4)


def circular_abs_diff(a: float, b: float) -> float:
    return abs(circular_delta(a, b))


# --------------------------------------------------------- classification
def _sample_class(value: float, spec: dict) -> int:
    """把单个估计归入 +1 / -1 / 0（零带内视为中性）。"""
    thr = float(spec.get("significance_threshold", 0.0))
    if spec.get("direction") == "positive":
        return 1 if value >= thr else 0
    if value >= thr:
        return 1
    if value <= -thr:
        return -1
    return 0


def _class_label(cls: int) -> str:
    return {1: "positive", -1: "negative", 0: "neutral"}[cls]


# --------------------------------------------------------------- bucket math
def _param_index(registry: dict) -> dict[str, dict]:
    return {item["key"]: item for item in registry["edit_parameters"]}


def _collect(
    analyses: dict[str, dict], members: list[str], key: str, conf_min: float
) -> list[tuple[str, float, float]]:
    out = []
    for image_id in members:
        payload = analyses[image_id]["edit_hypotheses"].get(key)
        if not payload:
            continue
        confidence = float(payload.get("confidence", 0.0))
        if confidence < conf_min:
            continue
        out.append((image_id, float(payload["estimate"]), confidence))
    return out


def bucket_param_stats(samples: list[tuple[str, float, float]], spec: dict, n_train: int) -> dict:
    kind = spec.get("kind", "numeric")
    pairs = [(value, weight) for _, value, weight in samples]
    if kind == "circular":
        central = circular_median(pairs)
        q25 = weighted_quantile(pairs, 0.25)
        q75 = weighted_quantile(pairs, 0.75)
        mad = None
        dispersion = circular_dispersion(pairs)
        consistency = round(dispersion, 4) if dispersion is not None else 0.0
        n_pos = n_neg = n_neutral = 0
        dominant_class = "circular"
        outliers: list[str] = []
    else:
        central = weighted_median(pairs)
        q25 = weighted_quantile(pairs, 0.25)
        q75 = weighted_quantile(pairs, 0.75)
        mad = weighted_mad(pairs, central)
        dispersion = None
        classes = [_sample_class(value, spec) for _, value, _ in samples]
        n_pos = classes.count(1)
        n_neg = classes.count(-1)
        n_neutral = classes.count(0)
        dominant = max(
            (n_pos, 1), (n_neg, -1), (n_neutral, 0), key=lambda kv: (kv[0], -abs(kv[1]))
        )[1]
        dominant_class = _class_label(dominant)
        consistency = round(max(n_pos, n_neg, n_neutral) / len(samples), 4) if samples else 0.0
        outliers = []
        if mad and mad > 0 and central is not None:
            for image_id, value, _ in samples:
                if abs(value - central) > 3.5 * 1.4826 * mad:
                    outliers.append(image_id)

    n_valid = len(samples)
    support = round(n_valid / n_train, 4) if n_train else 0.0
    stable = (
        n_valid >= STABLE_N_VALID
        and support >= STABLE_SUPPORT
        and consistency >= STABLE_CONSISTENCY
    )

    return {
        "kind": kind,
        "unit": spec.get("unit"),
        "n_train": n_train,
        "n_valid": n_valid,
        "support": support,
        "central": central,
        "p25": q25,
        "p75": q75,
        "mad": mad,
        "circular_dispersion": dispersion,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_neutral": n_neutral,
        "dominant_class": dominant_class,
        "consistency": consistency,
        "mean_confidence": round(sum(w for _, _, w in samples) / n_valid, 4) if n_valid else None,
        "outliers": outliers,
        "stable": stable,
        "significance_threshold": spec.get("significance_threshold"),
        "practical_zero_band": spec.get("practical_zero_band"),
    }


def visual_attributes(analyses: dict[str, dict], members: list[str], registry: dict) -> dict:
    out = {}
    for item in registry["visual_attributes"]:
        key = item["key"]
        counts: dict[str, float] = {}
        for image_id in members:
            value = analyses[image_id]["visual_observation"].get(key)
            if value is None:
                continue
            counts[value] = counts.get(value, 0.0) + 1.0
        total = sum(counts.values())
        if not total:
            continue
        mode = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        out[key] = {
            "mode": mode,
            "share": round(counts[mode] / total, 4),
            "proportions": {k: round(v / total, 4) for k, v in sorted(counts.items())},
            "n": int(total),
        }
    return out


def representatives(analyses: dict[str, dict], members: list[str], keys: list[str]) -> list[str]:
    """确定性 medoid：共同参数空间里到各维中位数距离最小者。"""
    centers: dict[str, float] = {}
    for key in keys:
        vals = sorted(
            float(analyses[mid]["edit_hypotheses"][key]["estimate"])
            for mid in members
            if analyses[mid]["edit_hypotheses"].get(key)
        )
        if vals:
            centers[key] = vals[len(vals) // 2]
    vectors = {
        mid: {
            key: float(analyses[mid]["edit_hypotheses"][key]["estimate"])
            for key in centers
            if analyses[mid]["edit_hypotheses"].get(key)
        }
        for mid in members
    }
    vectors = {k: v for k, v in vectors.items() if v}
    if not vectors:
        return []
    usable = [k for k in centers if all(k in v for v in vectors.values())]
    if not usable:
        return []
    scored = sorted(
        (round(sum(abs(vector[k] - centers[k]) for k in usable), 4), mid)
        for mid, vector in vectors.items()
    )
    return [mid for _, mid in scored[:3]]


# ------------------------------------------------------------------ pipeline
def build_bucket_profile(
    bucket_id: str,
    stats: dict,
    delta: dict,
    global_signature: dict,
    profile: dict,
    build_root: str | None = None,
) -> dict:
    """单个桶的风格指纹：只放增量与证据引用，不复制整套全局签名。"""
    outliers = sorted(
        {image_id for param in stats.get("params", {}).values() for image_id in param.get("outliers", [])}
    )
    narrative_ref = None
    if build_root:
        narrative_path = os.path.join(build_root, "buckets", bucket_id, "narrative.json")
        if os.path.isfile(narrative_path):
            with open(narrative_path, encoding="utf-8") as handle:
                narrative = json.load(handle)
            narrative_ref = {
                "file": f"buckets/{bucket_id}/narrative.json",
                "narrative_hash": narrative.get("narrative_hash"),
                "status": narrative.get("status", "draft"),
                "language": narrative.get("language", "zh"),
            }

    payload = {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_type": "bucket_style",
        "profile_id": f"bucket_{bucket_id}",
        "bucket_id": bucket_id,
        "taxonomy_version": global_signature.get("taxonomy_version"),
        "status": "draft",
        "bucket_status": delta.get("bucket_status") or stats.get("status"),
        "n_core": stats.get("n_core"),
        "n_train": stats.get("n_train"),
        "n_holdout": stats.get("n_holdout"),
        "base_profile_ref": {
            "profile_id": profile.get("profile_id"),
            "aggregate_hash": profile.get("aggregate_hash"),
            "global_signature_file_hash": profile.get("global_signature", {}).get("file_hash"),
        },
        "applicable_scope": {
            "scene_primary": bucket_id,
            "taxonomy_version": global_signature.get("taxonomy_version"),
        },
        "fallback": "global",
        "parameters": (profile.get("bucket_parameters") or {}).get(bucket_id, {}),
        "bucket_delta": delta.get("deltas", {}),
        "conditional_deltas": delta.get("conditional_deltas", []),
        "narrative_ref": narrative_ref,
        "unresolved": delta.get("unresolved", []),
        "dropped": delta.get("dropped", []),
        "representatives": stats.get("representatives", []),
        "outliers": outliers,
        "holdout_eval": profile.get("holdout_eval", {}).get(bucket_id),
        "cv_eval": profile.get("cv_eval", {}).get(bucket_id),
        "merge_check": profile.get("merge_check", {}).get(bucket_id),
        "evidence": {
            "stats_file": f"buckets/{bucket_id}/stats.json",
            "stats_hash": _canonical_hash(stats),
            "delta_file": f"buckets/{bucket_id}/delta.json",
            "delta_hash": _canonical_hash(delta),
            "global_signature_file": "global_signature.json",
            "global_signature_hash": profile.get("global_signature", {}).get("file_hash"),
        },
        "warnings": [
            "candidate_not_validated：尚未通过 §13 留出/Bootstrap 发布门槛。",
            "只保存桶增量，不复制全局签名；应用时必须先取 global 再叠加本桶 delta。",
        ],
    }
    payload["profile_hash"] = _canonical_hash(payload)
    return payload


def analysis_coverage(analyses: dict[str, dict], cfg: dict) -> dict:
    """检查编辑假设是否被叙事挤掉；返回可写入 qa 的覆盖度报告。"""
    if not analyses:
        return {
            "passed": False,
            "n": 0,
            "mean_hypotheses": 0.0,
            "key_parameters": list(KEY_PARAMETERS),
            "key_coverage": {},
            "min_key_coverage": 0.0,
            "thresholds": {},
            "reasons": ["no_analyses"],
        }
    counts = [len(item.get("edit_hypotheses", {})) for item in analyses.values()]
    n = len(counts)
    mean_hypotheses = round(sum(counts) / n, 4)
    key_coverage = {
        key: round(sum(1 for item in analyses.values() if key in item.get("edit_hypotheses", {})) / n, 4)
        for key in KEY_PARAMETERS
    }
    min_key_coverage = min(key_coverage.values())
    mean_min = float(cfg.get("analysis_min_mean_hypotheses", 6.0))
    key_min = float(cfg.get("analysis_min_key_coverage", 0.65))
    reasons = []
    if mean_hypotheses < mean_min:
        reasons.append("mean_hypotheses_below_threshold")
    if min_key_coverage < key_min:
        reasons.append("key_parameter_coverage_below_threshold")
    return {
        "passed": not reasons,
        "n": n,
        "mean_hypotheses": mean_hypotheses,
        "key_parameters": list(KEY_PARAMETERS),
        "key_coverage": key_coverage,
        "min_key_coverage": min_key_coverage,
        "thresholds": {
            "analysis_min_mean_hypotheses": mean_min,
            "analysis_min_key_coverage": key_min,
        },
        "reasons": reasons,
        "status": "pass" if not reasons else "blocked",
        "note": "数值假设覆盖不足时阻止 compose/package，避免产出空的风格指纹。",
    }


def _central_from(analyses: dict, ids: list[str], key: str, spec: dict, conf_min: float) -> float | None:
    samples = _collect(analyses, list(ids), key, conf_min)
    if not samples:
        return None
    pairs = [(value, weight) for _, value, weight in samples]
    if spec["kind"] == "circular":
        return circular_median(pairs)
    return weighted_median(pairs)


def _abs_err(predicted: float, observed: float, spec: dict) -> float:
    if spec["kind"] == "circular":
        return circular_abs_diff(predicted, observed)
    return abs(predicted - observed)


def _cross_validate(
    *,
    analyses: dict,
    stats: dict,
    bucket_members: dict,
    global_params: dict,
    bucket_deltas: dict,
    spec_by_key: dict,
    conf_min: float,
    iterations: int,
    seed: int,
    split: dict,
) -> dict:
    """K 折 out-of-fold 对比：仅全局 vs 桶中心（= 全局+增量）。

    因为 delta 的定义就是「桶心 − 全局心」，所以 global+delta 恒等于该折训练集上的
    桶心。每折只重算桶心与全局心（不重复做稳定性门禁），因此这里度量的是数值流程的
    样本外表现，不是发布门槛。
    """
    rng = random.Random(seed)
    result: dict[str, dict] = {}
    for bucket, info in stats.items():
        folds = [list(fold) for fold in (split.get("buckets", {}).get(bucket, {}).get("folds") or [])]
        if len(folds) < 2:
            result[bucket] = {
                "mode": "kfold",
                "n_folds": 0,
                "params": {},
                "note": "该桶没有可用折，未做交叉验证。",
            }
            continue
        core = set(bucket_members[bucket]["train"])
        reliability = info["reliability"]
        params: dict[str, dict] = {}
        not_evaluable: list[dict] = []
        published = bucket_deltas.get(bucket, {}).get("deltas", {})

        for key in spec_by_key:
            spec = spec_by_key[key]
            stat = info["params"].get(key)
            if not stat or stat["n_valid"] < PARAM_MIN_VALID:
                not_evaluable.append({"key": key, "reason": "bucket_param_missing_or_sparse"})
                continue

            # 其它桶的桶心在整轮 CV 中不变（它们不含本折图），只算一次
            others = []
            for other, other_info in stats.items():
                if other == bucket or key not in other_info["params"]:
                    continue
                central = other_info["params"][key]["central"]
                if central is not None and other_info["params"][key]["n_valid"] >= PARAM_MIN_VALID:
                    others.append((central, other_info["reliability"]))

            pairs: list[tuple[float, float]] = []
            for fold in folds:
                fold_set = set(fold)
                train_k = [image_id for image_id in core if image_id not in fold_set]
                bucket_k = _central_from(analyses, train_k, key, spec, conf_min)
                if bucket_k is None:
                    continue
                pool = list(others) + [(bucket_k, reliability)]
                global_k = circular_median(pool) if spec["kind"] == "circular" else weighted_median(pool)
                if global_k is None:
                    continue
                for image_id in fold:
                    payload = analyses[image_id]["edit_hypotheses"].get(key)
                    if not payload or float(payload.get("confidence", 0.0)) < conf_min:
                        continue
                    observed = float(payload["estimate"])
                    pairs.append(
                        (_abs_err(global_k, observed, spec), _abs_err(bucket_k, observed, spec))
                    )

            if not pairs:
                not_evaluable.append({"key": key, "reason": "no_out_of_fold_samples"})
                continue

            n = len(pairs)
            mae_global = sum(item[0] for item in pairs) / n
            mae_bucket = sum(item[1] for item in pairs) / n
            relative = (mae_global - mae_bucket) / mae_global if mae_global > 0 else 0.0
            ci_low = ci_high = None
            if iterations > 0:
                draws = []
                for _ in range(iterations):
                    sample = [pairs[rng.randrange(n)] for _ in range(n)]
                    g = sum(item[0] for item in sample) / n
                    b = sum(item[1] for item in sample) / n
                    draws.append((g - b) / g if g > 0 else 0.0)
                draws.sort()
                ci_low = round(draws[int(0.05 * (len(draws) - 1))], 4)
                ci_high = round(draws[int(0.95 * (len(draws) - 1))], 4)

            entry = {
                "n": n,
                "baseline_kind": "global" if key in global_params else "pooled",
                "mae_global_only": round(mae_global, 3),
                "mae_bucket_only": round(mae_bucket, 3),
                "relative_improvement": round(relative, 4),
                "ci90": [ci_low, ci_high] if ci_low is not None else None,
                "ci_excludes_zero": bool(ci_low is not None and (ci_low > 0 or ci_high < 0)),
                "bucket_stable": bool(stat["stable"]),
                "bucket_central": stat["central"],
                "global_central": (
                    global_params[key]["central"] if key in global_params else None
                ),
                "published_delta": published.get(key, {}).get("delta", 0.0) if key in published else None,
                "published": key in published,
            }
            params[key] = entry

        improved = [k for k, v in params.items() if v["relative_improvement"] > 0]
        worsened = [k for k, v in params.items() if v["relative_improvement"] < 0]
        result[bucket] = {
            "mode": "kfold",
            "n_folds": len(folds),
            "fold_sizes": [len(fold) for fold in folds],
            "n_core": len(core),
            "params": params,
            "params_below_min_valid": [k for k, v in params.items() if v["n"] < PARAM_MIN_VALID],
            "improved_params": improved,
            "worsened_params": worsened,
            "ci_excludes_zero_params": [
                k for k, v in params.items() if v["ci_excludes_zero"] and v["relative_improvement"] > 0
            ],
            "not_evaluable": not_evaluable,
            "gate_applied": False,
            "note": (
                "10 折 out-of-fold 对比：global+delta 恒等于该折训练集上的桶心；"
                "每折重算桶心与全局心，不重复稳定性门禁。Bootstrap 90% 区间为探索性证据，"
                "未做 scale_p 归一化与 P_bucket 全集门槛。"
            ),
        }
    return result


def _diagnostic_validate(
    *,
    analyses: dict,
    assignments: list[dict],
    stats: dict,
    global_params: dict,
    split: dict,
    spec_by_key: dict,
    conf_min: float,
    iterations: int,
    seed: int,
) -> dict:
    """小样本桶（provisional / exception）的只读诊断。

    这些桶不进入 stats，因此不影响全局基线，也不会产生可发布的 delta。
    这里只回答“如果把它当成一个桶，它的中心比全局更准吗”，用于判断是否值得补样本。
    """
    rng = random.Random(seed + 1)
    core_by_bucket: dict[str, list[str]] = {}
    for item in assignments:
        if item["level"] == "core" and item.get("bucket_id") and item["bucket_id"] not in stats:
            core_by_bucket.setdefault(item["bucket_id"], []).append(item["image_id"])
    result: dict[str, dict] = {}
    for bucket, core in sorted(core_by_bucket.items()):
        entry = split.get("buckets", {}).get(bucket, {})
        folds = [list(fold) for fold in (entry.get("diagnostic_folds") or [])]
        if len(folds) < 2:
            continue
        params: dict[str, dict] = {}
        not_evaluable: list[dict] = []
        for key in spec_by_key:
            if key not in global_params:
                continue
            spec = spec_by_key[key]
            samples = _collect(analyses, core, key, conf_min)
            if len(samples) < 3:
                not_evaluable.append({"key": key, "reason": "fewer_than_3_samples", "n": len(samples)})
                continue
            stat = bucket_param_stats(samples, spec, len(core))
            pairs: list[tuple[float, float]] = []
            fold_centrals: list[float] = []
            for fold in folds:
                fold_set = set(fold)
                train_k = [image_id for image_id in core if image_id not in fold_set]
                bucket_k = _central_from(analyses, train_k, key, spec, conf_min)
                if bucket_k is None:
                    continue
                fold_centrals.append(bucket_k)
                for image_id in fold:
                    payload = analyses[image_id]["edit_hypotheses"].get(key)
                    if not payload or float(payload.get("confidence", 0.0)) < conf_min:
                        continue
                    observed = float(payload["estimate"])
                    pairs.append(
                        (
                            _abs_err(global_params[key]["central"], observed, spec),
                            _abs_err(bucket_k, observed, spec),
                        )
                    )
            if not pairs:
                not_evaluable.append({"key": key, "reason": "no_out_of_fold_samples"})
                continue
            n = len(pairs)
            mae_global = sum(item[0] for item in pairs) / n
            mae_bucket = sum(item[1] for item in pairs) / n
            relative = (mae_global - mae_bucket) / mae_global if mae_global > 0 else 0.0
            ci_low = ci_high = None
            if iterations > 0:
                draws = []
                for _ in range(iterations):
                    sample = [pairs[rng.randrange(n)] for _ in range(n)]
                    g = sum(item[0] for item in sample) / n
                    b = sum(item[1] for item in sample) / n
                    draws.append((g - b) / g if g > 0 else 0.0)
                draws.sort()
                ci_low = round(draws[int(0.05 * (len(draws) - 1))], 4)
                ci_high = round(draws[int(0.95 * (len(draws) - 1))], 4)
            params[key] = {
                "n": n,
                "n_valid_full": stat["n_valid"],
                "support": stat["support"],
                "consistency": stat["consistency"],
                "stable": stat["stable"],
                "bucket_central": stat["central"],
                "global_central": (
                    global_params[key]["central"] if key in global_params else None
                ),
                "candidate_delta": (
                    round(circular_delta(stat["central"], global_params[key]["central"]), 4)
                    if spec["kind"] == "circular"
                    else round(stat["central"] - global_params[key]["central"], 4)
                ),
                "mae_global_only": round(mae_global, 3),
                "mae_bucket_only": round(mae_bucket, 3),
                "relative_improvement": round(relative, 4),
                "ci90": [ci_low, ci_high] if ci_low is not None else None,
                "fold_central_min": min(fold_centrals) if fold_centrals else None,
                "fold_central_max": max(fold_centrals) if fold_centrals else None,
                "fold_central_span": (
                    round(max(fold_centrals) - min(fold_centrals), 4) if fold_centrals else None
                ),
                "below_min_valid": n < PARAM_MIN_VALID,
            }
        result[bucket] = {
            "status": "diagnostic_only",
            "used_in_baseline": False,
            "publishable": False,
            "bucket_status": entry.get("status"),
            "n_core": len(core),
            "n_folds": len(folds),
            "fold_sizes": [len(fold) for fold in folds],
            "params": params,
            "not_evaluable": not_evaluable,
            "note": (
                "小样本桶只读诊断：不进入全局基线、不产生 delta；"
                "out-of-fold n 很小，区间仅供参考，不能据此发布。"
            ),
        }
    return result


def build_bucket_parameters(
    stats: dict,
    cv_eval: dict,
    spec_by_key: dict,
    *,
    min_valid: int = PARAM_MIN_VALID,
    min_support: float = STABLE_SUPPORT,
    prior_cv_min: float = 0.10,
    prior_span_max: float = 30.0,
) -> dict[str, dict]:
    """桶为主的参数表：每个桶给出绝对中心，不依赖全局零点。

    status 语义：
      published       桶内稳定，且留出对比不比池化差 → 可直接作为锚点
      wide_dispersion 桶内不稳但样本外优于池化 → 只作先验，带宽放宽
      withheld        证据不足或留出更差 → 不发布
    """
    out: dict[str, dict] = {}
    for bucket, info in stats.items():
        params: dict[str, dict] = {}
        for key, stat in info["params"].items():
            spec = spec_by_key[key]
            span = (
                round(stat["p75"] - stat["p25"], 4)
                if stat["p25"] is not None and stat["p75"] is not None
                else None
            )
            entry = {
                "value": stat["central"],
                "kind": spec.get("kind"),
                "unit": spec.get("unit"),
                "p25": stat["p25"],
                "p75": stat["p75"],
                "span": span,
                "n": stat["n_valid"],
                "support": stat["support"],
                "consistency": stat["consistency"],
                "practical_zero_band": spec.get("practical_zero_band"),
            }
            cv = ((cv_eval.get(bucket) or {}).get("params") or {}).get(key)
            if cv:
                entry["cv_vs_pooled"] = cv.get("relative_improvement")
                entry["cv_ci90"] = cv.get("ci90")
            if stat["n_valid"] < min_valid or stat["support"] < min_support:
                entry.update(status="withheld", reason="insufficient_samples", apply_policy="off")
            elif stat["stable"] and (not cv or cv["relative_improvement"] >= -0.05):
                if abs(stat["central"] or 0.0) < float(spec.get("practical_zero_band", 0.0)):
                    entry.update(status="neutral", reason="value_within_zero_band", apply_policy="off")
                else:
                    entry.update(status="published", reason="bucket_stable", apply_policy="anchor")
            elif stat["stable"] and cv and cv["relative_improvement"] < -0.05:
                entry.update(status="withheld", reason="cv_vs_pooled_negative", apply_policy="off")
            elif (
                cv
                and cv["relative_improvement"] >= prior_cv_min
                and (span is None or span <= prior_span_max)
            ):
                entry.update(
                    status="wide_dispersion",
                    reason="cv_beats_pooled_but_dispersed",
                    apply_policy="prior_only",
                )
            else:
                entry.update(status="withheld", reason="not_stable", apply_policy="off")
            params[key] = entry
        out[bucket] = params
    return out


def compose(build_root: str, *, write: bool = True, allow_underfilled: bool = False) -> dict:
    cfg_payload = _read_json(os.path.join(build_root, "config.resolved.json"))
    cfg = cfg_payload.get("config", cfg_payload)
    registry = _read_json(os.path.join(os.path.dirname(__file__), "schema", "parameters.json"))
    spec_by_key = _param_index(registry)

    analyses: dict[str, dict] = {}
    for task in _read_jsonl(os.path.join(build_root, "mcp", "tasks.jsonl")):
        rel = task.get("normalized_path")
        if task["status"] == "submitted" and rel:
            path = os.path.join(build_root, rel)
            if os.path.isfile(path):
                analyses[task["image_id"]] = _read_json(path)

    state_path = os.path.join(build_root, "mcp", "state.json")
    state = _read_json(state_path) if os.path.isfile(state_path) else {}
    input_fingerprint = state.get("input_fingerprint")
    config_hash = state.get("config_hash")
    response_set_hash = _canonical_hash({key: analyses[key] for key in sorted(analyses)})

    assignments = _read_jsonl(os.path.join(build_root, "assignments.jsonl"))
    split = _read_json(os.path.join(build_root, "evaluation_split.json"))
    conf_min = float(cfg.get("parameter_min_confidence", CONF_MIN_DEFAULT))
    coverage = analysis_coverage(analyses, cfg)
    if write:
        _write_json(os.path.join(build_root, "qa", "analysis_coverage.json"), coverage)
    if not coverage["passed"] and not allow_underfilled:
        return {
            "blocked": True,
            "analysis_coverage": coverage,
            "global_signature": None,
            "profile": None,
            "stats": {},
            "bucket_deltas": {},
        }

    bucket_members = {
        item["bucket_id"]: {"train": [], "holdout": []}
        for item in assignments
        if item["level"] == "core" and item.get("bucket_id")
    }
    for bucket, info in split.get("buckets", {}).items():
        if bucket in bucket_members:
            bucket_members[bucket]["train"] = list(info.get("train", []))
            bucket_members[bucket]["holdout"] = list(info.get("holdout", []))

    # ---- 桶内统计（训练侧 core）
    stats: dict[str, dict] = {}
    for bucket, members in bucket_members.items():
        train = members["train"]
        if not train:
            continue
        params = {}
        for key, spec in spec_by_key.items():
            samples = _collect(analyses, train, key, conf_min)
            if samples:
                params[key] = bucket_param_stats(samples, spec, len(train))
        stats[bucket] = {
            "bucket_id": bucket,
            "n_core": len(train) + len(members["holdout"]),
            "n_train": len(train),
            "n_holdout": len(members["holdout"]),
            "reliability": round(min(1.0, len(train) / CORE_FULL_MIN), 4),
            "params": params,
            "visual_attributes": visual_attributes(analyses, train, registry),
            "representatives": representatives(analyses, train, list(spec_by_key)),
            "members_train": sorted(train),
        }

    # ---- 全局基线 / 跨场景不变量判定
    global_params: dict[str, dict] = {}
    insufficient: list[dict] = []
    for key, spec in spec_by_key.items():
        contributors = [
            (bucket, info["params"][key], info["reliability"], info["n_train"])
            for bucket, info in stats.items()
            if key in info["params"] and info["params"][key]["stable"]
        ]
        if not contributors:
            insufficient.append({"key": key, "reason": "no_stable_bucket"})
            continue
        if len(contributors) < GLOBAL_MIN_BUCKETS:
            # 只有一个桶稳定：不能据此发布全局基线，否则另一桶会被迫套用错误零点。
            only_bucket, only_info, _, _ = contributors[0]
            insufficient.append(
                {
                    "key": key,
                    "reason": "single_bucket_evidence",
                    "bucket_id": only_bucket,
                    "bucket_central": only_info["central"],
                    "consistency": only_info["consistency"],
                    "n_valid": only_info["n_valid"],
                }
            )
            continue

        pairs = [(info["central"], reliability) for _, info, reliability, _ in contributors]
        central = circular_median(pairs) if spec["kind"] == "circular" else weighted_median(pairs)

        # 桶级优势方向投票
        vote: dict[str, float] = {}
        for _, info, reliability, _ in contributors:
            vote[info["dominant_class"]] = vote.get(info["dominant_class"], 0.0) + reliability
        global_class = max(vote.items(), key=lambda kv: kv[1])[0]
        sign_agreement = round(
            sum(reliability for _, info, reliability, _ in contributors if info["dominant_class"] == global_class)
            / sum(reliability for _, _, reliability, _ in contributors),
            4,
        )

        total_weight = sum(reliability for _, _, reliability, _ in contributors)
        weighted_consistency = round(
            sum(info["consistency"] * reliability for _, info, reliability, _ in contributors) / total_weight, 4
        )

        centers = [info["central"] for _, info, _, _ in contributors]
        band = float(spec.get("practical_zero_band", 0.0))
        spread = round(max(centers) - min(centers), 4)

        reasons = []
        if sign_agreement < GLOBAL_SIGN_AGREEMENT:
            reasons.append("cross_bucket_direction_disagreement")
        if weighted_consistency < GLOBAL_WEIGHTED_CONSISTENCY:
            reasons.append("weighted_consistency_below_threshold")
        if spread > band:
            reasons.append("bucket_centers_exceed_practical_band")

        image_pairs = []
        for bucket, _, reliability, n_train in contributors:
            for _, value, confidence in _collect(
                analyses, bucket_members[bucket]["train"], key, conf_min
            ):
                image_pairs.append((value, confidence * reliability / n_train))
        p25 = weighted_quantile(image_pairs, 0.25) if spec["kind"] != "circular" else None
        p75 = weighted_quantile(image_pairs, 0.75) if spec["kind"] != "circular" else None

        global_params[key] = {
            "kind": spec["kind"],
            "unit": spec.get("unit"),
            "central": central,
            "global_class": global_class,
            "cross_scene_invariant": not reasons,
            "invariance_reasons": reasons,
            "p25": p25,
            "p75": p75,
            "bucket_center_range": [min(centers), max(centers)],
            "bucket_center_spread": spread,
            "practical_zero_band": band,
            "n_buckets": len(contributors),
            "evidence_buckets": [bucket for bucket, _, _, _ in contributors],
            "bucket_dominant_classes": {b: i["dominant_class"] for b, i, _, _ in contributors},
            "sign_agreement": sign_agreement,
            "weighted_consistency": weighted_consistency,
            "total_support": sum(i["n_valid"] for _, i, _, _ in contributors),
        }

    evidence_buckets = sorted(
        {bucket for item in global_params.values() for bucket in item["evidence_buckets"]}
    )
    global_signature = {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "build_id": os.path.basename(build_root.rstrip("/\\")),
        "status": "draft",
        "shape": "global_signature",
        "taxonomy_version": split.get("taxonomy_version"),
        "registry_version": registry.get("registry_version"),
        "validation_scope": split.get("validation_scope"),
        "split_hash": split.get("split_hash"),
        "input_fingerprint": input_fingerprint,
        "config_hash": config_hash,
        "response_set_hash": response_set_hash,
        "evidence_scope": "cross_bucket" if len(evidence_buckets) >= 2 else "single_bucket",
        "evidence_bucket_count": len(evidence_buckets),
        "evidence_buckets": evidence_buckets,
        "parameters": global_params,
        "insufficient_evidence": insufficient,
        "invariant_parameters": [k for k, v in global_params.items() if v["cross_scene_invariant"]],
        "scene_dependent_parameters": [k for k, v in global_params.items() if not v["cross_scene_invariant"]],
        "weighted_median_rule": "midpoint_on_exact_half_weight",
        "single_bucket_policy": "single_bucket_evidence_never_publishes_global_or_delta",
        "baseline_note": (
            "每个参数都有全局基线 central，作为桶增量的零点；"
            "cross_scene_invariant=false 表示该参数强度随场景变化，"
            "禁止把基线单独套用到桶内，必须叠加该桶 delta。"
        ),
        "constraints": [],
        "local_adjustment_policy": "档案只覆盖全局倾向与桶级偏移，不虚构局部调整；local_adjustments_suspected 仅作证据保留。",
        "bucket_references": [
            {
                "bucket_id": b,
                "n_core": i["n_core"],
                "n_train": i["n_train"],
                "stats_hash": _canonical_hash(i),
            }
            for b, i in sorted(stats.items())
        ],
    }

    # ---- 桶增量
    bucket_deltas: dict[str, dict] = {}
    for bucket, info in stats.items():
        deltas, dropped, unresolved = {}, [], []
        for key, spec in spec_by_key.items():
            if key not in info["params"]:
                continue
            stat = info["params"][key]
            band = float(spec.get("practical_zero_band", 0.0))
            if not stat["stable"]:
                dropped.append({"key": key, "reason": "bucket_param_not_stable", "consistency": stat["consistency"]})
                continue
            if key not in global_params:
                # 无可用的跨桶基线：保留本桶绝对值备查，但不发布 delta，也不谎称全局值。
                unresolved.append(
                    {
                        "key": key,
                        "reason": "no_cross_bucket_baseline",
                        "bucket_central": stat["central"],
                        "range": [stat["p25"], stat["p75"]],
                        "n_valid": stat["n_valid"],
                        "support": stat["support"],
                        "consistency": stat["consistency"],
                        "note": "该参数只在本桶稳定，不能作为全局倾向传播。",
                    }
                )
                continue
            base = global_params[key]["central"]
            if base is None or stat["central"] is None:
                continue
            delta = (
                circular_delta(stat["central"], base)
                if spec["kind"] == "circular"
                else round(stat["central"] - base, 4)
            )
            magnitude = circular_abs_diff(stat["central"], base) if spec["kind"] == "circular" else abs(delta)
            if magnitude < band:
                dropped.append({"key": key, "reason": "below_practical_delta", "delta": delta, "band": band})
                continue
            deltas[key] = {
                "delta": delta,
                "kind": spec["kind"],
                "unit": spec.get("unit"),
                "global_central": base,
                "bucket_central": stat["central"],
                "range": [stat["p25"], stat["p75"]],
                "n_valid": stat["n_valid"],
                "support": stat["support"],
                "consistency": stat["consistency"],
                "mean_confidence": stat["mean_confidence"],
                "min_practical_delta": band,
                "status": "candidate",
            }
        bucket_deltas[bucket] = {
            "bucket_id": bucket,
            "status": "draft",
            "bucket_status": split.get("buckets", {}).get(bucket, {}).get("status"),
            "n_core": info["n_core"],
            "n_train": info["n_train"],
            "n_holdout": info["n_holdout"],
            "deltas": deltas,
            "dropped": dropped,
            "unresolved": unresolved,
            "conditional_deltas": [],
            "conditional_note": "条件子组（brightness/lighting/backlit）均未达最小 8 张，本轮不产出条件增量。",
        }

    # ---- 合并一致性 + 留出评估
    def merge(bucket: str) -> dict:
        merged = {k: v["central"] for k, v in global_params.items()}
        for key, item in bucket_deltas.get(bucket, {}).get("deltas", {}).items():
            if spec_by_key[key]["kind"] == "circular":
                merged[key] = round((merged[key] + item["delta"]) % 360.0, 4)
            else:
                merged[key] = round(merged[key] + item["delta"], 4)
        return merged

    merge_check: dict[str, dict] = {}
    holdout_eval: dict[str, dict] = {}
    for bucket, info in stats.items():
        merged = merge(bucket)
        max_err = 0.0
        failures = []
        for key, value in merged.items():
            stat = info["params"].get(key)
            if not stat or stat["central"] is None:
                continue
            err = (
                circular_abs_diff(value, stat["central"])
                if spec_by_key[key]["kind"] == "circular"
                else abs(value - stat["central"])
            )
            max_err = max(max_err, err)
            band = float(spec_by_key[key].get("practical_zero_band", 0.0))
            if err > band + 1e-9:
                failures.append({"key": key, "error": round(err, 4), "band": band})
        merge_check[bucket] = {
            "max_abs_error_vs_bucket_center": round(max_err, 4),
            "tolerance": "per_parameter_practical_zero_band",
            "within_tolerance": not failures,
            "failures": failures,
        }

        holdout = bucket_members[bucket]["holdout"]
        if not holdout:
            continue
        per_param = {}
        for key in merged:
            err_global, err_merged = [], []
            for image_id in holdout:
                payload = analyses[image_id]["edit_hypotheses"].get(key)
                if not payload or float(payload.get("confidence", 0.0)) < conf_min:
                    continue
                observed = float(payload["estimate"])
                if spec_by_key[key]["kind"] == "circular":
                    err_global.append(circular_abs_diff(global_params[key]["central"], observed))
                    err_merged.append(circular_abs_diff(merged[key], observed))
                else:
                    err_global.append(abs(global_params[key]["central"] - observed))
                    err_merged.append(abs(merged[key] - observed))
            if err_global:
                per_param[key] = {
                    "n": len(err_global),
                    "mae_global_only": round(sum(err_global) / len(err_global), 3),
                    "mae_global_plus_delta": round(sum(err_merged) / len(err_merged), 3),
                }
        holdout_eval[bucket] = {
            "n_holdout": len(holdout),
            "min_holdout": HOLDOUT_MIN,
            "holdout_sufficient": len(holdout) >= HOLDOUT_MIN,
            "params": per_param,
            "params_below_min_valid": [k for k, v in per_param.items() if v["n"] < PARAM_MIN_VALID],
            "improved_params": [k for k, v in per_param.items() if v["mae_global_plus_delta"] < v["mae_global_only"]],
            "worsened_params": [k for k, v in per_param.items() if v["mae_global_plus_delta"] > v["mae_global_only"]],
            "gate_applied": False,
            "note": "探索性 MAE；未按 scale_p 归一化、未使用 P_bucket 全集、未做 Bootstrap，不能作为发布门槛",
        }

    cv_eval: dict[str, dict] = {}
    if split.get("validation_mode", "holdout") == "kfold":
        cv_eval = _cross_validate(
            analyses=analyses,
            stats=stats,
            bucket_members=bucket_members,
            global_params=global_params,
            bucket_deltas=bucket_deltas,
            spec_by_key=spec_by_key,
            conf_min=conf_min,
            iterations=int(cfg.get("bootstrap_iterations", 200)),
            seed=int(cfg.get("split_seed", 0)),
            split=split,
        )

    diagnostic_cv: dict[str, dict] = {}
    if split.get("validation_mode", "holdout") == "kfold":
        diagnostic_cv = _diagnostic_validate(
            analyses=analyses,
            assignments=assignments,
            stats=stats,
            global_params=global_params,
            split=split,
            spec_by_key=spec_by_key,
            conf_min=conf_min,
            iterations=int(cfg.get("bootstrap_iterations", 200)),
            seed=int(cfg.get("split_seed", 0)),
        )

    bucket_parameters = build_bucket_parameters(stats, cv_eval, spec_by_key)
    published_counts = {
        bucket: sum(1 for item in params.values() if item["status"] == "published")
        for bucket, params in bucket_parameters.items()
    }

    aggregate_hash = _canonical_hash(
        {
            "global_signature": global_signature,
            "buckets": bucket_deltas,
            "bucket_parameters": bucket_parameters,
            "stats": stats,
        }
    )
    global_signature["aggregate_hash"] = aggregate_hash
    global_signature_hash = _canonical_hash(global_signature)

    profile = {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "profile_shape": cfg.get("profile_shape", "bucket_primary"),
        "profile_id": f"profile_{global_signature['build_id']}",
        "build_id": global_signature["build_id"],
        "status": "draft",
        "style_name": None,
        "shape": (
            "bucket_primary"
            if cfg.get("profile_shape", "bucket_primary") == "bucket_primary"
            else "global_signature+per_bucket_delta"
        ),
        "validation_scope": split.get("validation_scope"),
        "application_ready": False,
        "input_fingerprint": input_fingerprint,
        "config_hash": config_hash,
        "response_set_hash": response_set_hash,
        "aggregate_hash": aggregate_hash,
        "dataset": {
            "submitted_images": len(analyses),
            "global_parameters": len(global_params),
            "buckets": len(stats),
            "invariant_parameters": len(global_signature["invariant_parameters"]),
            "scene_dependent_parameters": len(global_signature["scene_dependent_parameters"]),
            "insufficient_parameters": len(insufficient),
        },
        "global_signature": {
            "file": "global_signature.json",
            "file_hash": global_signature_hash,
            "aggregate_hash": aggregate_hash,
            "parameter_count": len(global_params),
            "invariant_keys": global_signature["invariant_parameters"],
            "scene_dependent_keys": global_signature["scene_dependent_parameters"],
            "insufficient_keys": [i["key"] for i in insufficient],
            "evidence_scope": global_signature["evidence_scope"],
        },
        "buckets": dict(sorted(bucket_deltas.items())),
        "bucket_parameters": dict(sorted(bucket_parameters.items())),
        "bucket_parameter_counts": dict(sorted(published_counts.items())),
        "merge_check": merge_check,
        "holdout_eval": holdout_eval,
        "cv_eval": cv_eval,
        "diagnostic_cv": diagnostic_cv,
        "validation": {
            "status": "exploratory",
            "gate_applied": False,
            "mode": split.get("validation_mode", "holdout"),
            "reasons": (
                ["mae_not_normalized_by_scale_p", "no_p_bucket_ensemble_gate",
                 "cv_bootstrap_is_exploratory"]
                if split.get("validation_mode", "holdout") == "kfold"
                else ["holdout_mae_not_normalized_by_scale_p", "no_p_bucket_ensemble_gate",
                      "no_bootstrap_interval", "holdout_or_per_param_below_minimum"]
            ),
            "note": (
                "cv_eval / holdout_eval 仅用于探索；未达到 §13 的发布门槛，桶增量全部为 candidate。"
            ),
        },
        "release": {
            "numeric_profile": "draft",
            "bucket_deltas": "candidate",
            "application_ready": False,
        },
        "limits": [
            "validation_scope=analysis_consistency：留出仅验证分析一致性，不是感知/应用验证，application_ready 保持 false。",
            "样本均为单次 agent 判定，未做多模型复核。",
            "delta.status=candidate：尚未通过 Bootstrap/P_bucket 发布门槛，不能当作已验证增量。",
        ],
    }

    if write:
        _write_json(os.path.join(build_root, "global_signature.json"), global_signature)
        _write_json(os.path.join(build_root, "profile.json"), profile)
        for bucket, data in stats.items():
            _write_json(os.path.join(build_root, "buckets", bucket, "stats.json"), data)
        for bucket, data in bucket_deltas.items():
            _write_json(os.path.join(build_root, "buckets", bucket, "delta.json"), data)
        bucket_profiles = {}
        for bucket, data in stats.items():
            bucket_profile = build_bucket_profile(
                bucket, data, bucket_deltas[bucket], global_signature, profile, build_root
            )
            _write_json(os.path.join(build_root, "buckets", bucket, "profile.json"), bucket_profile)
            bucket_profiles[bucket] = bucket_profile
        index = {
            "buckets": [
                {
                    "bucket_id": bucket,
                    "profile_file": f"buckets/{bucket}/profile.json",
                    "profile_hash": bucket_profiles[bucket]["profile_hash"],
                    "narrative_file": bucket_profiles[bucket]["narrative_ref"]["file"]
                    if bucket_profiles[bucket]["narrative_ref"]
                    else None,
                    "narrative_hash": bucket_profiles[bucket]["narrative_ref"]["narrative_hash"]
                    if bucket_profiles[bucket]["narrative_ref"]
                    else None,
                }
                for bucket in sorted(stats)
            ],
            "note": "桶风格指纹索引；详细统计在 stats.json，增量在 delta.json；narrative 仅桶级。",
        }
        _write_json(os.path.join(build_root, "buckets", "index.json"), index)

    return {
        "global_signature": global_signature,
        "profile": profile,
        "stats": stats,
        "bucket_deltas": bucket_deltas,
    }


def main(argv: list[str]) -> int:
    args = [item for item in argv[1:] if not item.startswith("--")]
    allow = "--allow-underfilled" in argv[1:]
    if len(args) != 1:
        print("usage: python -m offline_profile.profile_compose <build_root> [--allow-underfilled]")
        return 2
    result = compose(args[0], allow_underfilled=allow)
    if result.get("blocked"):
        coverage = result["analysis_coverage"]
        print("analysis coverage blocked:")
        print(f"  mean_hypotheses={coverage['mean_hypotheses']} thresholds={coverage['thresholds']}")
        print(f"  key_coverage={coverage['key_coverage']}")
        print(f"  reasons={coverage['reasons']}")
        return 3
    signature = result["global_signature"]
    print(f"profile shape: {result['profile'].get('profile_shape')}")
    print("global baseline (fallback):")
    for key, item in signature["parameters"].items():
        flag = "invariant" if item["cross_scene_invariant"] else "scene-dependent"
        print(f"  {key:20s} {item['central']:>7} [{flag}] buckets={item['n_buckets']}")
    print(f"insufficient: {[i['key'] for i in signature['insufficient_evidence']]}")
    for bucket, data in result["profile"]["buckets"].items():
        pairs = ", ".join(f"{k}{v['delta']:+}" for k, v in data["deltas"].items())
        params = (result["profile"].get("bucket_parameters") or {}).get(bucket, {})
        published = [k for k, v in params.items() if v.get("status") == "published"]
        print(
            f"[{bucket}] n_core={data['n_core']} anchors={len(published)} "
            f"deltas: {pairs or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))

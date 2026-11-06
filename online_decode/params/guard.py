"""参数守卫：模型给的每一项都要过七道闸。

顺序即错误信息优先级：
  字段存在 → 可写性 → 证据等级 → 类型/值域 → 基线锚定 → 依赖/冲突 → 力度与末轮幅度

被拒项一律进 ledger.rejected，不静默丢弃——否则"这轮为什么没变化"无从解释。
"""

from __future__ import annotations

import math
from typing import Any

from ..util import clamp
from . import baseline as baseline_mod
from .fields import SHIFT_LOGICAL, SHIFT_SDKS, WB_PRESETS, Fields
from .ledger import EVIDENCE_LEVELS, FIRST_ROUND_ALLOWED, Entry, Ledger, Rejection

# 末轮：单值改动不超过上一轮改动的一半，且至少留 10 的余量（否则等于禁止微调）
LAST_ROUND_MIN_STEP = 10.0
# 参考带宽里额外计入桶内分散度：span 越大，允许偏离中心越多。
# ratio=1.0 表示允许走到 p25/p75 之外一个 IQR 宽度；cap 防止极端参数把带宽放大到失去约束。
BAND_SPAN_RATIO = 1.0
BAND_SPAN_CAP = 25.0
# 参考带宽只在前两轮生效：首轮定方向、次轮收敛；第 3 轮起允许模型按本图自由调整。
BAND_APPLY_MAX_ROUND = 2


class Guard:
    def __init__(self, fields: Fields) -> None:
        self.fields = fields

    def build_ledger(
        self,
        adjustments: list[dict[str, Any]],
        *,
        round_no: int,
        max_rounds: int,
        baseline: dict[str, Any],
        references: dict[str, float] | None = None,
        reference_spans: dict[str, float] | None = None,
        not_evaluable: set[str] | None = None,
        prev_values: dict[str, Any] | None = None,
        prev_step: dict[str, float] | None = None,
        strength: float = 1.0,
        user_intent: str | None = None,
    ) -> Ledger:
        ledger = Ledger(round_no=round_no)
        references = references or {}
        reference_spans = reference_spans or {}
        not_evaluable = not_evaluable or set()
        prev_values = prev_values or {}
        prev_step = prev_step or {}
        seen: dict[str, int] = {}

        for raw in adjustments or []:
            self._check_one(
                raw,
                ledger=ledger,
                round_no=round_no,
                max_rounds=max_rounds,
                baseline=baseline,
                references=references,
                reference_spans=reference_spans,
                not_evaluable=not_evaluable,
                prev_values=prev_values,
                prev_step=prev_step,
                strength=strength,
                seen=seen,
            )

        self._apply_white_balance_dependency(ledger, baseline)
        self._apply_grayscale_conflict(ledger)
        return ledger

    def build_ledger_from_xmp(
        self,
        values: dict[str, Any],
        *,
        notes: dict[str, dict[str, Any]] | None = None,
        round_no: int,
        max_rounds: int,
        baseline: dict[str, Any],
        references: dict[str, float] | None = None,
        reference_spans: dict[str, float] | None = None,
        not_evaluable: set[str] | None = None,
        prev_values: dict[str, Any] | None = None,
        prev_step: dict[str, float] | None = None,
        strength: float = 1.0,
        user_intent: str | None = None,
    ) -> Ledger:
        """模型直接给 XMP 时的入口。

        与 adjustments 路径走**同一套闸门**，区别只有一处：XMP 里没有理由和证据等级，
        所以元数据退化为可选（缺省 planner_judgement + ref=model:xmp），而不是直接拒。
        这样"模型交 XMP"不会因为缺注释而全盘失败，但字段名错、值越界照样拦得住。
        """
        notes = notes or {}
        adjustments: list[dict[str, Any]] = []
        for key, value in (values or {}).items():
            meta = notes.get(key) or {}
            adjustments.append({
                "key": key,
                "value": value,
                "evidence_level": meta.get("evidence_level") or "planner_judgement",
                "ref": meta.get("ref") or "model:xmp",
                "reason": meta.get("reason") or "",
            })
        ledger = self.build_ledger(
            adjustments,
            round_no=round_no,
            max_rounds=max_rounds,
            baseline=baseline,
            references=references,
            reference_spans=reference_spans,
            not_evaluable=not_evaluable,
            prev_values=prev_values,
            prev_step=prev_step,
            strength=strength,
            user_intent=user_intent,
        )
        if not notes:
            ledger.warn("模型未提供 param_notes：本轮参数只有字段级校验，没有逐项理由")
        return ledger

    # ------------------------------------------------------------------
    def _reject(self, ledger: Ledger, raw: Any, code: str, message: str) -> None:
        key = raw.get("key") if isinstance(raw, dict) else None
        value = raw.get("value") if isinstance(raw, dict) else None
        level = raw.get("evidence_level") if isinstance(raw, dict) else None
        ledger.rejected.append(
            Rejection(given_key=str(key or ""), value=value, code=code,
                      message=message, evidence_level=level)
        )

    def _check_one(
        self,
        raw: Any,
        *,
        ledger: Ledger,
        round_no: int,
        max_rounds: int,
        baseline: dict[str, Any],
        references: dict[str, float],
        reference_spans: dict[str, float],
        not_evaluable: set[str],
        prev_values: dict[str, Any],
        prev_step: dict[str, float],
        strength: float,
        seen: dict[str, int],
    ) -> None:
        if not isinstance(raw, dict):
            self._reject(ledger, raw, "ENTRY_NOT_OBJECT", "adjustments 的每一项必须是对象")
            return

        key = raw.get("key")
        if not isinstance(key, str) or not key:
            self._reject(ledger, raw, "KEY_MISSING", "缺少 key")
            return

        # 1) 字段存在
        sdk = self.fields.canonical(key)
        if sdk is None:
            self._reject(ledger, raw, "FIELD_UNKNOWN", f"未知字段 {key}（不在字段清单里）")
            return

        # 2) 可写性（tier=v1 才在插件默认写入范围内）
        if not self.fields.is_writable(key):
            spec = self.fields.spec(key) or {}
            self._reject(
                ledger, raw, "FIELD_UNSUPPORTED",
                f"{key} 属于 tier={spec.get('tier')}，插件默认不写入，接口层无法执行",
            )
            return

        # 3) 证据等级
        level = raw.get("evidence_level")
        if level is None:
            level = "planner_judgement"
            ledger.warn(f"{key} 未标 evidence_level，按 planner_judgement 处理")
        if level not in EVIDENCE_LEVELS:
            self._reject(ledger, raw, "EVIDENCE_LEVEL_UNKNOWN", f"未知证据等级 {level}")
            return
        if level == "unsupported":
            self._reject(ledger, raw, "EVIDENCE_UNSUPPORTED", f"{key} 被模型自己标为 unsupported")
            return
        if level not in FIRST_ROUND_ALLOWED:
            self._reject(ledger, raw, "EVIDENCE_NOT_ALLOWED", f"该轮不允许 {level}")
            return

        # 2026-11-06 用户决策：指纹未覆盖（not_evaluable）的字段不再禁写。
        # 解码者可对指纹没担保的参数按本图自主调整；证据等级照常逐项记在台账
        # entry 上（可追溯依据仍在），但不额外告警、不要求 target_adaptive 通道。
        # 仍然拒绝的只有：字段不在清单、tier != v1（接口写不进）、标 unsupported。
        # （not_evaluable 集合继续下传，仅用于报告统计，不再作为闸门。）

        value = raw.get("value")
        ref = str(raw.get("ref") or "")
        reason = str(raw.get("reason") or "")

        # narrative_only 是"无数值证据"的状态；模型仍给了数，就必须改标
        if level == "narrative_only" and isinstance(value, (int, float)):
            level = "planner_judgement"
            ledger.warn(f"{key} 以叙事为依据给数值，等级改为 planner_judgement")
        if level == "planner_judgement" and not ref:
            self._reject(ledger, raw, "REF_MISSING", f"{key} 属模型判断但没给 ref")
            return

        # 4) 类型 / 值域
        ftype = self.fields.type_of(key)
        if ftype == "toneCurve":
            self._reject(ledger, raw, "CURVE_UNSUPPORTED", f"{key} 是曲线，首版不由模型设定")
            return
        if ftype == "boolean":
            if not isinstance(value, bool):
                self._reject(ledger, raw, "VALUE_TYPE", f"{key} 期望布尔值，收到 {type(value).__name__}")
                return
        elif ftype == "enum":
            if not isinstance(value, str) or value not in WB_PRESETS:
                self._reject(ledger, raw, "VALUE_ENUM", f"{key} 取值非法：{value!r}")
                return
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                self._reject(ledger, raw, "VALUE_TYPE", f"{key} 期望数值，收到 {type(value).__name__}")
                return
            if not math.isfinite(float(value)):
                self._reject(ledger, raw, "VALUE_NAN", f"{key} 值为 NaN/Inf")
                return

        # 5) 基线锚定（只有 *_shift 语义的键才需要；直接写 Temperature 视为绝对值）
        is_shift = key in SHIFT_LOGICAL
        resolved, notes, err = baseline_mod.resolve(sdk, value, baseline, is_shift=is_shift)
        if err:
            self._reject(ledger, raw, "BASELINE_MISSING", f"{key}：{err}")
            return

        # 6) 值域（锚定后再查，避免钳制掩盖意图）
        rng = self.fields.range_of(key)
        if rng and isinstance(resolved, (int, float)):
            low, high = rng
            if resolved < low or resolved > high:
                clamped = clamp(float(resolved), low, high)
                notes.append(f"{key} {resolved} 越界 [{low:g}, {high:g}]，钳制到 {clamped:g}")
                resolved = int(clamped) if float(clamped).is_integer() else clamped

        # 7) 参考带宽（仅在指纹给了参考值时启用）
        if (
            isinstance(resolved, (int, float))
            and sdk in references
            and sdk not in SHIFT_SDKS
            and self.fields.group_of(key) != "whitebalance"
            and level != "target_adaptive"
        ):
            ref_value = float(references[sdk])
            span = float(reference_spans.get(sdk) or 0.0)
            span_term = min(span * BAND_SPAN_RATIO, BAND_SPAN_CAP) if span > 0 else 0.0
            allowance = max(abs(ref_value), span_term, 5.0) * (0.5 + clamp(strength, 0.3, 2.0))
            if abs(float(resolved) - ref_value) > allowance:
                if round_no <= BAND_APPLY_MAX_ROUND:
                    self._reject(
                        ledger, raw, "OUT_OF_BAND",
                        f"{key}={resolved} 偏离参考中心 {ref_value:g} 超过允许带宽 {allowance:.0f}",
                    )
                    return
                notes.append(
                    f"带宽已在第 {BAND_APPLY_MAX_ROUND} 轮后解除：{key}={resolved} 偏离参考 "
                    f"{ref_value:g} 超出原带宽 {allowance:.0f}，按本图自由调整"
                )

        # 8) 末轮阻尼
        if (
            round_no == max_rounds
            and isinstance(resolved, (int, float))
            and sdk in prev_values
            and isinstance(prev_values[sdk], (int, float))
        ):
            prev = float(prev_values[sdk])
            step = abs(float(resolved) - prev)
            prev_delta = abs(float(prev_step.get(sdk, 0.0)))
            allowed = max(LAST_ROUND_MIN_STEP, 0.5 * prev_delta)
            if step > allowed:
                target = prev + math.copysign(allowed, float(resolved) - prev)
                notes.append(
                    f"末轮阻尼：{key} 由 {resolved} 收到 {target:g}（上轮改动 {prev_delta:g}）"
                )
                resolved = int(target) if float(target).is_integer() else round(target, 2)

        # 9) 去重（同一字段出现多次以最后一次为准，但留痕）
        if sdk in seen:
            ledger.warn(f"{key} 重复出现，以前一条为准，后者已忽略")
            return
        seen[sdk] = len(ledger.entries)

        ledger.entries.append(
            Entry(
                given_key=key,
                sdk=sdk,
                intent=value,
                resolved=resolved,
                written=resolved,
                evidence_level=level,
                ref=ref,
                reason=reason,
                notes=notes,
            )
        )

    # ------------------------------------------------------------------
    def _apply_white_balance_dependency(self, ledger: Ledger, baseline: dict[str, Any]) -> None:
        touched = [e for e in ledger.entries if e.sdk in SHIFT_SDKS]
        if not touched:
            return
        if any(e.sdk == "WhiteBalance" for e in ledger.entries):
            return
        ledger.entries.append(
            Entry(
                given_key="WhiteBalance",
                sdk="WhiteBalance",
                intent="Custom",
                resolved="Custom",
                written="Custom",
                evidence_level="target_adaptive",
                ref="dependency:Temperature/Tint",
                reason="改了色温/色调必须显式切到 Custom，否则 LR 会忽略数值",
                notes=["守卫自动补齐"],
            )
        )
        ledger.auto_added.append("WhiteBalance=Custom")

    def _apply_grayscale_conflict(self, ledger: Ledger) -> None:
        grayscale = next((e for e in ledger.entries if e.sdk == "ConvertToGrayscale"), None)
        if not grayscale or grayscale.written is not True:
            return
        # 黑白模式下失效的不止 HSL：Vibrance/Saturation 在 basic 组里，单独点名
        color_sdks = {"Vibrance", "Saturation"}
        keep = []
        for entry in ledger.entries:
            group = self.fields.group_of(entry.sdk)
            if group in ("hsl", "colorgrading") or entry.sdk in color_sdks:
                ledger.rejected.append(
                    Rejection(
                        given_key=entry.given_key, value=entry.written,
                        code="CONFLICT_GRAYSCALE",
                        message="转为黑白后色彩通道无意义，已拒绝",
                        evidence_level=entry.evidence_level,
                    )
                )
                continue
            keep.append(entry)
        ledger.entries = keep

"""会话控制器：把"看图 → 写入 → 渲染 → 复核 → 收敛/回退"串成一条可审计的链。

它做三件事，且只做这三件：
  1. 守卫状态转移（谁能写 LR、下一轮还允不允许）；
  2. 每步落盘（XMP / 快照 / 预览 / 评价 / 指标），让任何一轮都能被回放；
  3. 失败时降级成"可交付的部分结果 + 明确错误"，不拿别的轮次冒充 final。
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from .config import Config
from .errors import Codes, DecodeError
from .eval import selector, verify_evidence
from .feedback import round as round_mod
from .feedback.lr_port import LrPort
from .fingerprint import brief as brief_mod
from .fingerprint.resolver import (
    decide_bucket,
    load_narrative,
    priors as fingerprint_priors,
    references as fingerprint_references,
    verify_package,
)
from .params.fields import LOGICAL_TO_SDK, Fields
from .params.guard import BAND_APPLY_MAX_ROUND, Guard
from .params import xmp_parser
from .ports import Driver, DriverError
from .state import SessionState, make_session_id, new_session
from .util import atomic_write_bytes, atomic_write_json, ensure_dir, is_ascii, now_iso, sha256_bytes, sha256_file

REQUIRED_CAPABILITIES = ("apply", "get_settings", "render_preview", "restore_settings", "export_final")

# 这几类错误表示"还在等模型"，不是"会话失败"；会话必须保持可 resume。
MODEL_WAIT_CODES = {
    Codes.MODEL_RESPONSE_TIMEOUT,
    Codes.MODEL_RESPONSE_INVALID,
    Codes.EVIDENCE_UNVERIFIED,
}


class SessionController:
    def __init__(self, cfg: Config, port: LrPort, driver: Driver, *, fields: Fields | None = None) -> None:
        self.cfg = cfg
        self.port = port
        self.driver = driver
        self.fields = fields or Fields.load()
        self.guard = Guard(self.fields)
        self.state: SessionState | None = None
        self._pending_warnings: list[str] = []

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def run(
        self,
        photo: str,
        package_dir: str,
        *,
        user_intent: str | None = None,
        strength: float | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        workspace = Path(self.cfg.workspace)
        strength_value = self.cfg.style_strength_default if strength is None else float(strength)

        try:
            pkg = self._preflight(photo, package_dir)
            session_id = session_id or make_session_id()
            state = new_session(
                workspace, session_id, photo,
                max_rounds=self.cfg.max_rounds,
                stop_on_single=self.cfg.effective_stop_on_single_no_progress,
                stop_source=self.cfg.stop_source,
                mode="pilot" if (not pkg.application_ready or pkg.status != "ready") else "standard",
            )
            self.state = state
            state.fingerprint = {
                "fingerprint_id": pkg.fingerprint_id,
                "fingerprint_hash": pkg.fingerprint_hash,
                "package_id": pkg.manifest.get("package_id"),
                "package_hash": pkg.manifest.get("package_hash"),
                "status": pkg.status,
                "application_ready": pkg.application_ready,
                "schema_version": pkg.schema_version,
                "path": str(pkg.path.resolve()),
            }
            state.user_intent = user_intent
            state.style_strength = strength_value
            for warning in self._pending_warnings:
                state.warn(warning)
            self._pending_warnings.clear()
            state.save()

            baseline = self._baseline(photo, session_id, pkg)
            mode = self._mode(pkg)
            state.fingerprint["decode_mode"] = mode
            state.save()
            scene = (
                self._scene(photo, session_id, pkg, baseline, user_intent, strength_value)
                if mode == "bucket_primary"
                else None
            )
            brief = self._brief(
                photo, session_id, pkg, baseline, user_intent, strength_value, scene=scene
            )
            self._run_rounds(photo, session_id, brief, baseline, user_intent, strength_value)
            self._finalize(photo, session_id)

        except DecodeError as exc:
            if not self.state:
                return {
                    "ok": False,
                    "session_id": None,
                    "error": exc.to_dict(),
                    "config": self.cfg.to_report(),
                }
            if exc.code in MODEL_WAIT_CODES:
                # 等模型不是"会话失败"：状态保持可恢复，resume 才能接上。
                self.state.warn(f"会话挂起等待模型：{exc.message}")
                self.state.errors.append({
                    "code": exc.code,
                    "message": exc.message,
                    "stage": exc.stage,
                    "resumable": True,
                })
                self.state.substep = "awaiting_model"
            else:
                self.state.fail(exc.to_dict())
        except DriverError as exc:
            if self.state:
                self.state.warn(f"会话挂起等待模型：{exc.message}")
                self.state.errors.append({"code": exc.code, "message": exc.message, "resumable": True})
                self.state.substep = "awaiting_model"
        finally:
            if self.state:
                self.state.save()

        if not self.state:
            return {"ok": False, "error": {"code": "UNKNOWN", "message": "会话未能建立"}}
        from .report import build as build_report

        report = build_report(self.state, self.cfg)
        atomic_write_json(self.state.root / "decode_report.json", report)
        return report

    # ------------------------------------------------------------------
    # 续跑
    # ------------------------------------------------------------------
    def resume(self, session_root: str | Path) -> dict[str, Any]:
        """从会话目录接着跑，已完成的轮次不重做、不重复渲染。

        重建依据全部来自落盘文件：session.json / creative_brief.json /
        rounds/rNN/*.json。中断在模型等待时最常见，此时已有响应文件会被复用，
        不会让人白写一次。

        只有"已成功渲染并评价过"的轮次算已完成；末尾那个中断轮次会从状态里剔除，
        但它的文件仍留在 rounds/ 供人工排查（只追加，不删档）。
        """
        import json

        from .report import build as build_report

        root = Path(session_root)
        try:
            state = SessionState.load(root)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            return {
                "ok": False,
                "session_id": None,
                "error": DecodeError(
                    Codes.SESSION_RESUME_INVALID,
                    f"读取会话失败：{exc}",
                    stage="resume",
                ).to_dict(),
            }
        self.state = state

        if state.status in ("completed", "failed", "aborted"):
            return {
                "ok": False,
                "session_id": state.session_id,
                "error": DecodeError(
                    Codes.SESSION_RESUME_INVALID,
                    f"会话状态为 {state.status}，不能续跑",
                    stage="resume",
                ).to_dict(),
            }

        try:
            package_path = (state.fingerprint or {}).get("path")
            if not package_path:
                raise DecodeError(
                    Codes.SESSION_RESUME_INVALID,
                    "会话里没有指纹包路径，无法重建上下文",
                    stage="resume",
                )
            photo = state.photo
            pkg = self._preflight(photo, package_path)
            for warning in self._pending_warnings:
                state.warn(warning)
            self._pending_warnings.clear()

            # 只保留完整轮次；中断轮的文件留在磁盘上，不删
            completed = [
                item for item in state.rounds
                if item.get("ok") and item.get("evaluation") and item.get("apply")
            ]
            if len(completed) != len(state.rounds):
                state.warn(
                    f"续跑时丢弃 {len(state.rounds) - len(completed)} 个未完成轮次"
                    "（文件仍保留在 rounds/ 供排查）"
                )
                state.rounds = completed
            state.stop_reason = None
            state.errors = []
            state.warn("会话已恢复，从下一轮继续")
            state.set_status("initialized", "resume")

            baseline = state.baseline
            if not baseline.get("settings_hash"):
                baseline = self._baseline(photo, state.session_id, pkg)

            brief_path = root / "creative_brief.json"
            if brief_path.is_file():
                brief = json.loads(brief_path.read_text(encoding="utf-8"))
            else:
                scene_path = root / "scene.json"
                scene = None
                if scene_path.is_file():
                    try:
                        scene = json.loads(scene_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        scene = None
                brief = self._brief(photo, state.session_id, pkg, baseline,
                                    state.user_intent, state.style_strength, scene=scene)

            self._run_rounds(photo, state.session_id, brief, baseline,
                             state.user_intent, state.style_strength,
                             existing_rounds=completed)
            self._finalize(photo, state.session_id)
        except DecodeError as exc:
            state.fail(exc.to_dict())
        finally:
            state.save()

        report = build_report(state, self.cfg)
        atomic_write_json(root / "decode_report.json", report)
        return report

    # ------------------------------------------------------------------
    # 预检
    # ------------------------------------------------------------------
    def _preflight(self, photo: str, package_dir: str):
        caps = self.port.capabilities()
        missing = [c for c in REQUIRED_CAPABILITIES if c not in (caps.get("capabilities") or [])]
        if missing:
            raise DecodeError(
                Codes.CAPABILITY_MISSING,
                f"插件缺少能力：{', '.join(missing)}（当前 {caps.get('plugin_version')}）",
                stage="preflight",
                detail=caps,
            )
        if caps.get("source") == "none":  # 全新安装还没写过 capabilities.json
            raise DecodeError(
                Codes.LISTENER_NOT_RUNNING,
                "没有检测到插件能力声明，监听器可能没开",
                stage="preflight",
            )

        path = Path(photo)
        if not path.is_file():
            raise DecodeError(Codes.PHOTO_NOT_FOUND, f"照片不存在：{photo}", stage="preflight")
        # 真机 AC-10 验证过：含中文的源照片路径，插件 PhotoResolver 能正常定位，
        # 所以这里不再硬拦。真正受 ANSI 限制的是**插件要读写的产物路径**：
        # XMP 由控制器写到 workspace，必须保证 ASCII。
        if not is_ascii(str(path)):
            self._pending_warnings.append(
                "源照片路径含非 ASCII 字符；真机已验证插件能定位，继续运行，"
                "但 XMP/快照产物路径必须保持 ASCII"
            )
        if not is_ascii(str(Path(self.cfg.workspace).resolve())):
            raise DecodeError(
                Codes.PHOTO_PATH_NOT_ASCII,
                "workspace 必须是 ASCII 路径：控制器生成的 XMP 要由插件用 Lua io 读取",
                stage="preflight",
            )

        pkg = verify_package(package_dir)
        if not pkg.fingerprint_id:
            raise DecodeError(Codes.FINGERPRINT_INVALID, "fingerprint.json 缺 fingerprint_id", stage="preflight")
        return pkg

    # ------------------------------------------------------------------
    # 基线
    # ------------------------------------------------------------------
    def _baseline(self, photo: str, session_id: str, pkg) -> dict[str, Any]:
        assert self.state
        self.state.set_status("initialized", "baseline")
        rounds_dir = ensure_dir(self.state.root / "rounds")

        snapshot = round_mod.stable_settings(self.port, photo)
        if not snapshot.ok:
            raise DecodeError(
                Codes.SNAPSHOT_INCOMPLETE,
                f"基线快照失败：{snapshot.message}",
                stage="baseline",
            )
        snap = snapshot.snapshot
        baseline_complete = bool(snap.get("complete", False))
        if not baseline_complete:
            self.state.warn(
                "基线设置读数持续不完整（LR 仍在物化 develop 数据）："
                "恢复 r00 时不能承诺精确一致，原图预览不带期望哈希（pilot 降级）"
            )

        baseline_hash = snapshot.settings_hash
        if not baseline_hash:
            raise DecodeError(Codes.SNAPSHOT_INCOMPLETE, "基线快照没有 settings_hash", stage="baseline")

        preview = self.port.render_preview(
            photo, session_id=session_id, round_no=1,
            expected_hash=baseline_hash if baseline_complete else None,
            long_edge=self.cfg.preview_long_edge,
        )
        if not preview.ok:
            raise DecodeError(
                Codes.RENDER_FAILED,
                f"原图预览失败：{preview.message}",
                stage="baseline",
            )

        src = Path((preview.entry.get("render") or {}).get("path", ""))
        local = rounds_dir / "r00_original.jpg"
        if src.is_file():
            shutil.copy2(src, local)

        baseline = {
            "settings_hash": baseline_hash,
            "snapshot": snap,
            "settings": (snapshot.entry.get("settings") or {}),
            "preview": str(local) if local.is_file() else str(src),
            "preview_source": str(src),
            "source": {
                "path": str(photo),
                "sha256": sha256_file(photo),
                "bytes": Path(photo).stat().st_size,
            },
            "captured_at": now_iso(),
        }
        self.state.baseline = baseline
        self.state.save()
        return baseline

    # ------------------------------------------------------------------
    # Brief
    # ------------------------------------------------------------------
    def _mode(self, pkg) -> str:
        if self.cfg.decode_mode != "auto":
            return self.cfg.decode_mode
        return "bucket_primary" if pkg.fingerprint.get("shape") == "bucket_primary" else "global_delta"

    def _scene(self, photo: str, session_id: str, pkg, baseline: dict[str, Any],
               user_intent: str | None, strength: float) -> dict[str, Any]:
        """第一段：只看原图判场景。桶参数此时不可见，避免先验污染分类。"""
        assert self.state
        scene_path = self.state.root / "scene.json"
        if scene_path.is_file():
            try:
                cached = json.loads(scene_path.read_text(encoding="utf-8"))
                if cached.get("observation"):
                    return cached
            except json.JSONDecodeError:
                pass
        request = {
            "request_schema_version": "1.0.0",
            "task": "scene",
            "session_id": session_id,
            "round": 0,
            "images": {"original": baseline["preview"], "current": None, "previous": None},
            "fingerprint": {
                "fingerprint_id": pkg.fingerprint_id,
                "status": pkg.status,
                "shape": pkg.fingerprint.get("shape"),
                "bucket_labels": pkg.bucket_ids,
            },
            "limits": self._limits(),
            "user_intent": user_intent,
            "style_strength": strength,
        }
        response = self._call_driver("scene", request)
        observation = response.get("scene_observation") or {}
        if not observation:
            raise DecodeError(
                Codes.MODEL_RESPONSE_INVALID,
                "scene 响应缺少 scene_observation",
                stage="scene",
            )
        decision = decide_bucket(pkg, observation)
        narrative = load_narrative(pkg, decision["bucket_id"]) if decision.get("bucket_id") else None
        payload = {
            "observation": observation,
            "decision": decision,
            "narrative": narrative,
            "image_id": response.get("image_id"),
        }
        atomic_write_json(scene_path, payload)
        return payload

    def _brief(self, photo: str, session_id: str, pkg, baseline: dict[str, Any],
               user_intent: str | None, strength: float, *, scene: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self.state
        self.state.set_status("initialized", "briefing")

        base_request = {
            "request_schema_version": "1.0.0",
            "task": "brief",
            "session_id": session_id,
            "round": 0,
            "images": {"original": baseline["preview"], "current": None, "previous": None},
            "fingerprint": {
                "fingerprint_id": pkg.fingerprint_id,
                "status": pkg.status,
                "application_ready": pkg.application_ready,
                "shape": pkg.fingerprint.get("shape"),
                "runtime_rule": pkg.fingerprint.get("runtime_rule"),
                "buckets": pkg.bucket_ids,
                "global": pkg.fingerprint.get("global"),
                "insufficient_evidence": pkg.fingerprint.get("insufficient_evidence"),
            },
            "baseline_settings": baseline.get("settings"),
            "limits": self._limits(),
            "user_intent": user_intent,
            "style_strength": strength,
        }

        if scene is None:
            request = base_request
            response = self._call_driver("brief", request)
            observation = response.get("scene_observation") or {}
            if not observation:
                raise DecodeError(
                    Codes.MODEL_RESPONSE_INVALID,
                    "单段式 brief 响应缺少 scene_observation",
                    stage="brief",
                )
            decision = decide_bucket(pkg, observation)
            narrative = load_narrative(pkg, decision["bucket_id"]) if decision.get("bucket_id") else None
        else:
            observation = scene.get("observation") or {}
            decision = scene.get("decision") or decide_bucket(pkg, observation)
            narrative = scene.get("narrative")
            bucket_id = decision.get("bucket_id")
            refs = fingerprint_references(pkg, bucket_id)
            request = dict(base_request)
            request["scene"] = {
                "observation": observation,
                "decision": decision,
                "anchors": {
                    sdk: {"value": item["value"], "role": item.get("role"),
                          "span": item.get("span"), "consistency": item.get("consistency")}
                    for sdk, item in refs.items()
                    if isinstance(item, dict) and item.get("value") is not None
                },
                "priors": fingerprint_priors(pkg, bucket_id),
                "narrative": {
                    key: (narrative or {}).get(key)
                    for key in ("summary", "mood", "color_language", "cautions", "exceptions")
                } if narrative else {},
            }
            response = self._call_driver("brief", request)

        brief = brief_mod.build(
            session_id=session_id,
            photo=str(photo),
            image_id=str(response.get("image_id") or Path(photo).stem),
            pkg=pkg,
            observation=observation,
            decision=decision,
            narrative=narrative,
            response=response,
            max_rounds=self.cfg.max_rounds,
            user_intent=user_intent,
            strength=strength,
        )
        from .fingerprint.resolver import priors as _priors

        brief["parameter_priors"] = _priors(pkg, decision.get("bucket_id"))
        bucket_id = decision.get("bucket_id")
        bucket = (pkg.fingerprint.get("buckets") or {}).get(bucket_id) or {}
        brief["decode"] = {
            "mode": self._mode(pkg),
            "bucket_id": bucket_id,
            "fallback": bool(decision.get("fallback")),
            "not_evaluable": [
                LOGICAL_TO_SDK[key]
                for key in (bucket.get("not_evaluable") or [])
                if key in LOGICAL_TO_SDK
            ],
        }
        brief["baseline"] = {
            "settings_hash": baseline["settings_hash"],
            "preview": baseline["preview"],
        }
        atomic_write_json(self.state.root / "creative_brief.json", brief)
        if decision.get("fallback"):
            self.state.warn(f"无可靠桶：回退 global（{decision.get('fallback_reason')}）")
        if self.state.mode == "pilot":
            self.state.warn("指纹为 draft / application_ready=false：本次按 pilot 运行")
        return brief

    # ------------------------------------------------------------------
    # 轮次
    # ------------------------------------------------------------------
    def _run_rounds(self, photo: str, session_id: str, brief: dict[str, Any],
                    baseline: dict[str, Any], user_intent: str | None, strength: float,
                    *, existing_rounds: list[dict[str, Any]] | None = None,
                    first_plan: dict[str, Any] | None = None) -> None:
        assert self.state
        rounds: list[dict[str, Any]] = list(existing_rounds or [])
        previous_record: dict[str, Any] | None = rounds[-1] if rounds else None
        # resume 时从已有轮次重建"上一轮写入了什么 / 改动多大"，末轮阻尼才不会失效
        prev_values, prev_step = _reconstruct_prev(rounds)

        default_plan: dict[str, Any] = (
            {"xmp": brief.get("initial_xmp"), "param_notes": brief.get("initial_param_notes") or {}}
            if brief.get("initial_xmp")
            else {"adjustments": brief.get("initial_plan") or []}
        )
        plan: dict[str, Any] | None = first_plan if first_plan is not None else (
            default_plan if not rounds else None
        )

        for round_no in range(len(rounds) + 1, self.cfg.max_rounds + 1):
            if plan is None:
                plan = self._next_plan(photo, session_id, brief, rounds, previous_record, user_intent)
                if plan.get("stop"):
                    if len(rounds) >= self.cfg.effective_min_rounds:
                        self.state.stop_reason = "model_stop"
                        break
                    # 强制最少轮数未满：把"停"解释为"保持上轮参数再验证一次"，
                    # 而不是绕过强制轮次。第 2 轮会照常 apply→render→evaluate。
                    prev = previous_record or {}
                    held = (
                        (prev.get("ledger") or {}).get("written_values")
                        or prev.get("written")
                        or (prev.get("xmp") or {}).get("values")
                        or {}
                    )
                    self.state.warn(
                        f"r{round_no:02d}: 模型请求停止，但未满强制 {self.cfg.effective_min_rounds} 轮，"
                        "按保留上轮参数继续验证"
                    )
                    plan = {"adjustments": [
                        {
                            "key": key, "value": value,
                            "reason": "模型请求停止，强制轮次内保持上轮参数",
                            "ref": "model:stop_hold",
                            "evidence_level": "planner_judgement",
                        }
                        for key, value in held.items()
                    ]}

            round_dir = ensure_dir(self.state.root / "rounds" / f"r{round_no:02d}")
            source, payload, notes, model_artifact = self._resolve_plan_values(plan, round_dir)

            guard_kwargs = dict(
                round_no=round_no,
                max_rounds=self.cfg.max_rounds,
                baseline=baseline,
                references={k: v["value"] for k, v in (brief.get("parameter_references") or {}).items()
                            if isinstance(v, dict) and isinstance(v.get("value"), (int, float))},
                reference_spans={k: v["span"] for k, v in (brief.get("parameter_references") or {}).items()
                                 if isinstance(v, dict) and isinstance(v.get("span"), (int, float))},
                not_evaluable=set((brief.get("decode") or {}).get("not_evaluable") or []),
                prev_values=prev_values,
                prev_step=prev_step,
                strength=strength,
                user_intent=user_intent,
            )
            if source == "xmp":
                ledger = self.guard.build_ledger_from_xmp(payload, notes=notes, **guard_kwargs)
            else:
                ledger = self.guard.build_ledger(payload, **guard_kwargs)

            atomic_write_json(round_dir / "plan.json", plan)
            atomic_write_json(round_dir / "param_ledger.json", ledger.to_dict())
            for warning in ledger.warnings:
                self.state.warn(f"r{round_no:02d}: {warning}")

            if ledger.is_empty:
                self.state.warn(
                    f"r{round_no:02d}: 所有参数都被守卫拒绝，本轮不写入 LR"
                )
                if not rounds:
                    raise DecodeError(
                        Codes.PLAN_REJECTED,
                        "首轮参数全部被拒，无法建立影调方向",
                        stage=f"round{round_no}",
                        detail=ledger.to_dict(),
                    )
                break

            self.state.set_status("applying", f"round{round_no}")
            record = round_mod.execute(
                self.port,
                photo=photo,
                session_id=session_id,
                round_no=round_no,
                values=ledger.values,
                round_dir=round_dir,
                long_edge=self.cfg.preview_long_edge,
                baseline_preview=baseline.get("preview"),
            )
            refs = brief.get("parameter_references") or {}
            for entry in ledger.entries:
                entry.role = (refs.get(entry.sdk) or {}).get("role") or "unreferenced"
            ledger_payload = ledger.to_dict()
            ledger_payload["anchor_usage"] = {
                "bucket": sum(1 for e in ledger.entries if e.role == "bucket"),
                "global_fallback": sum(1 for e in ledger.entries if e.role == "global_fallback"),
                "unreferenced": sum(1 for e in ledger.entries if e.role == "unreferenced"),
                "not_evaluable_written": sum(
                    1 for e in ledger.entries
                    if e.sdk in set((brief.get("decode") or {}).get("not_evaluable") or [])
                ),
            }
            record["ledger"] = ledger_payload
            record["planned"] = plan.get("adjustments")
            record["model_input"] = model_artifact

            if not record["ok"]:
                self.state.add_round(record)
                rounds.append(record)
                self._handle_round_error(record)
                break

            self.state.set_status("rendering", f"round{round_no}")
            atomic_write_json(round_dir / "snapshot.json", record.get("snapshot") or {})
            if record.get("metrics"):
                atomic_write_json(round_dir / "metrics.json", record["metrics"])
            _copy_preview(record, round_dir)

            self.state.set_status("evaluating", f"round{round_no}")
            evaluation = self._evaluate(photo, session_id, brief, round_no, record, rounds)
            record["evaluation"] = evaluation
            atomic_write_json(round_dir / "evaluation.json", evaluation)

            self.state.add_round(record)
            rounds.append(record)
            previous_record = record

            prev_step = {
                entry.sdk: float(entry.written) - float(prev_values.get(entry.sdk, 0))
                for entry in ledger.entries
                if isinstance(entry.written, (int, float))
            }
            prev_values = dict(ledger.values)

            best = selector.pick_best(rounds)
            if best and _all_worse(rounds):
                self.state.warn("所有轮次都不如原图，不选最佳轮（保留 r00 基线）")
                best = None
            self.state.best = best
            self.state.save()

            should_stop, reason = selector.should_stop(
                rounds,
                max_rounds=self.cfg.max_rounds,
                stop_on_single_no_progress=self.cfg.effective_stop_on_single_no_progress,
                has_best=bool(best),
                min_rounds=self.cfg.effective_min_rounds,
            )
            if should_stop:
                self.state.stop_reason = reason
                break
            plan = None  # 下一轮必须重新规划，不能复用本轮方案

        if not rounds:
            raise DecodeError(Codes.NO_ROUND_COMPLETED, "没有任何一轮形成有效反馈", stage="rounds")
        self.state.best = selector.pick_best(rounds)

    def _handle_round_error(self, record: dict[str, Any]) -> None:
        assert self.state
        error = record.get("error") or {}
        if (record.get("render") or {}).get("path") and error.get("code") == Codes.RENDER_HASH_MISMATCH:
            self.state.warn("预览哈希不匹配：该轮预览作废，不评价、不继续")
        self.state.stop_reason = "interface_failure"
        self.state.errors.append(error)

    def _restore_baseline_best_effort(self) -> None:
        """没有最佳轮时把照片放回 r00，避免把用户的照片留在更差的版本上。"""
        assert self.state
        snapshot = ((self.state.baseline or {}).get("snapshot") or {}).get("path")
        if not snapshot:
            return
        current = self.port.get_settings(self.state.photo)
        result = self.port.restore(
            self.state.photo, snapshot,
            expected_current_hash=current.settings_hash,
        )
        if result.ok:
            self.state.warn("已把照片恢复到 r00 基线")
        else:
            self.state.warn(f"r00 基线恢复失败，需人工处理：{result.message}")

    # ------------------------------------------------------------------
    # 模型交互
    # ------------------------------------------------------------------
    def _resolve_plan_values(self, plan: dict[str, Any], round_dir: Path):
        """从模型响应里取出本轮参数。

        优先按"XMP 文件"处理（需求 8.2 的原始口径）：原样存档为 model.xmp，
        解析后交给守卫逐字段校验；adjustments 路径仅作兼容保留。

        为什么不能原样送给 LR：Lightroom 对不认识的属性名是静默丢弃的，
        直接写入会得到"没报错但也没变化"的结果，而报告无法解释少了什么。
        """
        assert self.state
        xmp_text = plan.get("xmp")
        xmp_path = plan.get("xmp_path")
        if not xmp_text and xmp_path:
            path = Path(str(xmp_path))
            if not path.is_file():
                raise DecodeError(
                    Codes.MODEL_RESPONSE_INVALID,
                    f"响应里给了 xmp_path，但文件不存在：{path}",
                    stage="plan",
                )
            xmp_text = path.read_text(encoding="utf-8")

        if xmp_text:
            raw_path = round_dir / "model.xmp"
            atomic_write_bytes(raw_path, str(xmp_text).encode("utf-8"))
            parsed = xmp_parser.parse(str(xmp_text))
            artifact = {
                "source": "xmp",
                "path": str(raw_path),
                "sha256": sha256_bytes(str(xmp_text).encode("utf-8")),
                "fields": len(parsed.values),
                "metadata_keys": sorted(parsed.metadata),
                "issues": parsed.issues,
            }
            if parsed.issues:
                raise DecodeError(
                    Codes.MODEL_RESPONSE_INVALID,
                    "模型给的 XMP 不能用：" + "；".join(parsed.issues),
                    stage="plan",
                    detail=artifact,
                )
            if parsed.duplicates:
                self.state.warn("模型 XMP 有重复属性（以后一个为准）：" + ", ".join(parsed.duplicates))
            return "xmp", parsed.values, dict(plan.get("param_notes") or {}), artifact

        adjustments = plan.get("adjustments")
        if isinstance(adjustments, list) and adjustments:
            return "adjustments", adjustments, {}, {"source": "adjustments", "count": len(adjustments)}

        raise DecodeError(
            Codes.MODEL_RESPONSE_INVALID,
            "本轮响应既没有 xmp，也没有 adjustments",
            stage="plan",
        )

    def _limits(self) -> dict[str, Any]:
        allowed = sorted(self.fields.v1)
        ranges = {k: self.fields.range_of(k) for k in allowed}
        return {
            "max_rounds": self.cfg.max_rounds,
            "allowed_keys": allowed,
            "value_ranges": {k: v for k, v in ranges.items() if v},
            "unsupported": ["local_masks", "brush", "people_masking", "tone_curve"],
            "decode_mode": self.cfg.decode_mode,
            "band_until_round": BAND_APPLY_MAX_ROUND,
            "long_edge": self.cfg.preview_long_edge,
        }

    def _call_driver(self, task: str, request: dict[str, Any], *, attempts: int = 2) -> dict[str, Any]:
        from . import contracts

        # 请求是我们自己生成的，违反契约说明代码改坏了——直接失败，别把坏请求
        # 发给模型再让它去猜怎么填。
        if self.cfg.strict_schema:
            contracts.validate_or_raise(contracts.REQUEST, request, stage=task)

        response_schema = contracts.RESPONSE_SCHEMA.get(task)
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                response = getattr(self.driver, task)(request)
            except DriverError as exc:
                last_error = exc
                continue
            if isinstance(response, dict):
                if self.cfg.strict_schema and response_schema:
                    problems = contracts.validate(response_schema, response)
                    if problems:
                        last_error = DecodeError(
                            Codes.MODEL_RESPONSE_INVALID,
                            f"{task} 响应不符合契约：{'；'.join(problems[:3])}",
                            stage=task,
                            detail={"problems": problems},
                        )
                        continue
                return response
            last_error = DecodeError(Codes.MODEL_RESPONSE_INVALID, "模型响应不是对象", stage=task)
        if isinstance(last_error, DriverError):
            raise DecodeError(last_error.code, last_error.message, stage=task)
        raise DecodeError(
            Codes.MODEL_RESPONSE_INVALID,
            f"{task} 连续 {attempts} 次无效响应：{last_error}",
            stage=task,
        )

    def _evaluate(self, photo: str, session_id: str, brief: dict[str, Any], round_no: int,
                  record: dict[str, Any], rounds: list[dict[str, Any]]) -> dict[str, Any]:
        assert self.state
        request = {
            "request_schema_version": "1.0.0",
            "task": "evaluate",
            "session_id": session_id,
            "round": round_no,
            "images": {
                "original": brief.get("baseline", {}).get("preview"),
                "current": (record.get("render") or {}).get("path"),
                "previous": (rounds[-1].get("render") or {}).get("path") if rounds else None,
            },
            "brief": brief,
            "metrics": record.get("metrics"),
            "adjustments": record.get("ledger", {}).get("written_values"),
            "history": [_history_item(item) for item in rounds],
        }

        problems: list[str] = []
        for attempt in range(2):
            response = dict(self._call_driver("evaluate", request))
            ok, problems = verify_evidence.verify(response, record.get("metrics") or {})
            if ok:
                response.setdefault("round", round_no)
                response.setdefault("evaluated_at", now_iso())
                response["evidence_verified"] = True
                return response
            if attempt == 0:
                self.state.warn(
                    f"r{round_no:02d} 评价未通过核验，允许一次重做：{'；'.join(problems[:3])}"
                )

        raise DecodeError(
            Codes.EVIDENCE_UNVERIFIED,
            "评价连续两次无法核验：" + "；".join(problems[:3]),
            stage=f"evaluate{round_no}",
            detail={"problems": problems},
        )

    def _next_plan(self, photo: str, session_id: str, brief: dict[str, Any],
                   rounds: list[dict[str, Any]], previous: dict[str, Any] | None,
                   user_intent: str | None) -> dict[str, Any]:
        assert self.state
        request = {
            "request_schema_version": "1.0.0",
            "task": "plan",
            "session_id": session_id,
            "round": len(rounds) + 1,
            "images": {
                "original": brief.get("baseline", {}).get("preview"),
                "current": (previous or {}).get("render", {}).get("path"),
                "previous": (rounds[-2].get("render") or {}).get("path") if len(rounds) > 1 else None,
            },
            "brief": brief,
            "history": [_history_item(item) for item in rounds],
            "last_evaluation": (previous or {}).get("evaluation"),
            "limits": self._limits(),
            "user_intent": user_intent,
        }
        return self._call_driver("plan", request)

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------
    def _finalize(self, photo: str, session_id: str) -> None:
        assert self.state
        best = self.state.best
        if not best or not best.get("settings_hash"):
            self.state.stop_reason = self.state.stop_reason or "no_best"
            self._restore_baseline_best_effort()
            self.state.set_status("failed", "no_best")
            self.state.errors.append({
                "code": Codes.NO_ROUND_COMPLETED,
                "message": "没有可恢复的最佳轮，未导出成片",
                "hint": "照片已尝试恢复到 r00 基线；报告里保留了每轮预览供人工判断",
            })
            return

        target = best["settings_hash"]
        self.state.set_status("restoring", "best")

        current = self.port.get_settings(photo)
        current_hash = current.settings_hash
        if not current_hash:
            self.state.warn("恢复前读不到当前哈希，按无并发保护恢复")

        snapshot_path = (best.get("snapshot") or {}).get("path")
        if not snapshot_path:
            self.state.set_status("failed", "restore")
            self.state.errors.append({"code": Codes.RESTORE_FAILED, "message": "最佳轮缺快照路径"})
            return

        restored = self.port.restore(photo, snapshot_path, expected_current_hash=current_hash)
        if not restored.ok:
            self.state.set_status("failed", "restore")
            self.state.errors.append({
                "code": restored.error_code or Codes.RESTORE_FAILED,
                "message": restored.message or "恢复最佳轮失败",
                "hint": "已保留最佳轮预览，可人工恢复后重新导出",
            })
            return

        restored_hash = restored.settings_hash
        verify_block = (restored.entry.get("restore") or {}).get("verify") or {}
        mismatches = int(verify_block.get("mismatches") or 0)
        if mismatches > 0:
            self.state.set_status("failed", "restore")
            self.state.errors.append({
                "code": Codes.RESTORE_VERIFY_MISMATCH,
                "message": f"恢复后 {mismatches} 个字段读回不一致，禁止导出",
            })
            return

        export_hash = restored_hash or target
        if restored_hash and restored_hash != target:
            self.state.warn(
                "恢复后的哈希与最佳快照哈希不一致（字段集合漂移），按 pilot 用恢复后状态导出"
            )

        self.state.set_status("exporting", "final")
        exported = self.port.export_final(photo, session_id=session_id, expected_hash=export_hash)
        if not exported.ok:
            self.state.set_status("failed", "export")
            self.state.errors.append({
                "code": exported.error_code or Codes.EXPORT_FAILED,
                "message": exported.message or "导出成片失败",
                "hint": "最佳轮预览仍可用于人工出片",
                "best_preview": best.get("preview"),
            })
            return

        block = exported.entry.get("export") or {}
        final_dir = ensure_dir(self.state.root / "final")
        local_final = final_dir / "final.jpg"
        if block.get("path") and Path(block["path"]).is_file():
            shutil.copy2(block["path"], local_final)

        # 插件对超过 hashLimitBytes（默认 64KB）的产物会写 hash_skipped 而不算哈希。
        # 真机成片 25MB 就属于这种情况；AC-07 要求报告里哈希可复算，
        # 所以控制器侧补算一份，并标明来源，绝不伪造"插件算过"。
        file_sha = block.get("sha256")
        sha_source = "plugin"
        if not file_sha and local_final.is_file():
            file_sha = "sha256:" + sha256_file(local_final)
            sha_source = "controller_fallback"
            self.state.warn("插件跳过了大文件哈希，成片哈希由控制器补算")

        self.state.final = {
            "path": str(local_final) if local_final.is_file() else block.get("path"),
            "source_path": block.get("path"),
            "format": block.get("format"),
            "width": block.get("width"),
            "height": block.get("height"),
            "bytes": block.get("bytes"),
            "sha256": file_sha,
            "sha256_source": sha_source,
            "settings_hash": block.get("settings_hash"),
            "restored_hash": restored_hash,
            "best_round": best.get("round"),
            "exported_at": now_iso(),
        }
        atomic_write_json(final_dir / "best_settings.snapshot.json", best.get("snapshot") or {})
        atomic_write_json(final_dir / "restore.result.json", dict(restored.entry.get("restore") or {}))

        if restored_hash and restored_hash != target:
            self.state.warn("最终成片的设置哈希与最佳快照不完全一致，已在报告中标注")
        self.state.set_status("completed", "done")


# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------


def _history_item(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation") or {}
    return {
        "round": record.get("round"),
        "written": (record.get("ledger") or {}).get("written_values"),
        "improvement": evaluation.get("improvement"),
        "decision": evaluation.get("decision"),
        "stop_reason": record.get("stop_reason"),
    }


def _copy_preview(record: dict[str, Any], round_dir: Path) -> None:
    src = (record.get("render") or {}).get("path")
    if not src or not Path(src).is_file():
        return
    dest = round_dir / "preview.jpg"
    shutil.copy2(src, dest)
    record["render"]["local_path"] = str(dest)


def _all_worse(rounds: list[dict[str, Any]]) -> bool:
    improvements = [(r.get("evaluation") or {}).get("improvement") for r in rounds]
    return bool(improvements) and all(item == "worse" for item in improvements)


def _reconstruct_prev(rounds: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
    """从落盘的账本重建上一轮写入值与改动量，让续跑后的末轮阻尼仍然正确。"""
    if not rounds:
        return {}, {}
    written = dict((rounds[-1].get("ledger") or {}).get("written_values") or {})
    previous = dict((rounds[-2].get("ledger") or {}).get("written_values") or {}) if len(rounds) > 1 else {}
    step: dict[str, float] = {}
    for key, value in written.items():
        if isinstance(value, (int, float)):
            prior = previous.get(key, 0)
            if isinstance(prior, (int, float)):
                step[key] = float(value) - float(prior)
    return written, step

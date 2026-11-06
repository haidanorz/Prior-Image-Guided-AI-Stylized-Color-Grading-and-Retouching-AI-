"""单图分析执行器：缓存命中、重试、原始响应落盘、失败隔离。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import AdapterError, AnalysisStatus, EpisodeType, ErrorCode, InvalidAnalysis
from ..intake import IntakeImage, image_fingerprint
from ..store import BuildStore
from ..util import now_utc_iso
from ..version import ANALYSIS_SCHEMA_VERSION, PROMPT_VERSION
from .adapter import AnalysisAdapter, AnalysisRequest
from .cache import ResponseCache
from .normalize import normalize_analysis
from .prompt import build_prompt


@dataclass
class RunResult:
    image_id: str
    status: str
    payload: dict | None = None
    error_code: str | None = None
    error_message: str | None = None
    details: list[str] = field(default_factory=list)
    attempts: int = 0
    cached: bool = False
    warnings: list[str] = field(default_factory=list)


class AnalysisRunner:
    def __init__(
        self,
        *,
        adapter: AnalysisAdapter,
        cache: ResponseCache,
        store: BuildStore,
        episodes,
        cfg: dict,
        cfg_hash: str,
        build_id: str,
    ):
        self.adapter = adapter
        self.cache = cache
        self.store = store
        self.episodes = episodes
        self.cfg = cfg
        self.cfg_hash = cfg_hash
        self.build_id = build_id

    def run(self, image: IntakeImage) -> RunResult:
        fingerprint = image_fingerprint(
            image,
            analysis_config_hash=self.cfg_hash,
            prompt_version=PROMPT_VERSION,
            model_info=self.adapter.model_info.as_dict(),
            schema_version=ANALYSIS_SCHEMA_VERSION,
        )
        prompt = build_prompt(image, input_fingerprint=fingerprint, cfg=self.cfg)
        cache_key = self.cache.key(fingerprint, prompt.sha256)

        attempts: list[dict[str, Any]] = []
        recovered = False

        cached = self.cache.get(cache_key)
        if cached is not None:
            try:
                normalized = normalize_analysis(
                    cached.text,
                    image=image,
                    expected_fingerprint=fingerprint,
                    config=self.cfg,
                    model_info=self.adapter.model_info,
                )
                attempts.append({"attempt": 0, "cached": True, "text": cached.text, "error": None})
                self._write_raw(image, fingerprint, prompt, attempts)
                self.store.write_normalized(image.image_id, normalized.payload)
                return RunResult(
                    image_id=image.image_id,
                    status=AnalysisStatus.NORMALIZED,
                    payload=normalized.payload,
                    attempts=0,
                    cached=True,
                    warnings=normalized.warnings,
                )
            except InvalidAnalysis as exc:
                attempts.append(
                    {"attempt": 0, "cached": True, "text": cached.text, "error": exc.message}
                )
                self.episodes.add(
                    EpisodeType.ANALYSIS_INVALID,
                    image_id=image.image_id,
                    context={"attempt": 0, "cached": True, "details": exc.details[:20]},
                    action="normalize_cached_response",
                    result="rejected",
                )
                recovered = True

        max_attempts = 1 + int(self.cfg["analysis_max_retries"])
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.adapter.analyze(
                    AnalysisRequest(image=image, prompt=prompt, input_fingerprint=fingerprint)
                )
            except AdapterError as exc:
                attempts.append({"attempt": attempt, "cached": False, "text": None, "error": exc.message})
                self.episodes.add(
                    EpisodeType.ANALYSIS_ERROR,
                    image_id=image.image_id,
                    context={"attempt": attempt, "error": exc.message},
                    action="adapter_call",
                    result="failed",
                )
                if exc.retryable and attempt < max_attempts:
                    self.episodes.add(
                        EpisodeType.ANALYSIS_RETRY,
                        image_id=image.image_id,
                        context={"attempt": attempt, "error": exc.message},
                        action="retry_adapter",
                        result="retrying",
                    )
                    recovered = recovered or bool(attempts)
                    continue
                self._write_raw(image, fingerprint, prompt, attempts)
                return RunResult(
                    image_id=image.image_id,
                    status=AnalysisStatus.ERROR,
                    error_code=ErrorCode.ANALYSIS_TRANSPORT_ERROR,
                    error_message=exc.message,
                    attempts=attempt,
                )
            except Exception as exc:  # 单图隔离：适配器实现错误也不终止整批
                attempts.append(
                    {"attempt": attempt, "cached": False, "text": None, "error": repr(exc)}
                )
                self.episodes.add(
                    EpisodeType.ANALYSIS_ERROR,
                    image_id=image.image_id,
                    context={"attempt": attempt, "error": repr(exc)},
                    action="adapter_call",
                    result="failed",
                )
                self._write_raw(image, fingerprint, prompt, attempts)
                return RunResult(
                    image_id=image.image_id,
                    status=AnalysisStatus.ERROR,
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_message=f"适配器异常：{exc}",
                    attempts=attempt,
                )

            try:
                normalized = normalize_analysis(
                    response.text,
                    image=image,
                    expected_fingerprint=fingerprint,
                    config=self.cfg,
                    model_info=self.adapter.model_info,
                )
            except InvalidAnalysis as exc:
                attempts.append(
                    {"attempt": attempt, "cached": False, "text": response.text, "error": exc.message}
                )
                self.episodes.add(
                    EpisodeType.ANALYSIS_INVALID,
                    image_id=image.image_id,
                    context={"attempt": attempt, "details": exc.details[:20]},
                    action="normalize_response",
                    result="rejected",
                )
                if attempt < max_attempts:
                    self.episodes.add(
                        EpisodeType.ANALYSIS_RETRY,
                        image_id=image.image_id,
                        context={"attempt": attempt, "reason": "invalid_analysis"},
                        action="retry_adapter",
                        result="retrying",
                    )
                    recovered = True
                    continue
                self._write_raw(image, fingerprint, prompt, attempts)
                return RunResult(
                    image_id=image.image_id,
                    status=AnalysisStatus.INVALID,
                    error_code=ErrorCode.ANALYSIS_INVALID,
                    error_message=exc.message,
                    details=exc.details[:20],
                    attempts=attempt,
                )

            attempts.append({"attempt": attempt, "cached": False, "text": response.text, "error": None})
            self.cache.put(
                cache_key,
                response,
                input_fingerprint=fingerprint,
                prompt_version=prompt.version,
            )
            self._write_raw(image, fingerprint, prompt, attempts)
            self.store.write_normalized(image.image_id, normalized.payload)
            if recovered:
                self.episodes.add(
                    EpisodeType.ANALYSIS_RECOVERED,
                    image_id=image.image_id,
                    context={"attempt": attempt},
                    action="normalize_response",
                    result="accepted_after_retry",
                )
            return RunResult(
                image_id=image.image_id,
                status=AnalysisStatus.NORMALIZED,
                payload=normalized.payload,
                attempts=attempt,
                cached=False,
                warnings=normalized.warnings,
            )

        self._write_raw(image, fingerprint, prompt, attempts)
        return RunResult(
            image_id=image.image_id,
            status=AnalysisStatus.ERROR,
            error_code=ErrorCode.INTERNAL_ERROR,
            error_message="重试循环意外结束",
            attempts=max_attempts,
        )

    def _write_raw(self, image: IntakeImage, fingerprint: str, prompt, attempts: list[dict]) -> None:
        self.store.write_raw(
            image.image_id,
            {
                "image_id": image.image_id,
                "input_fingerprint": fingerprint,
                "prompt_version": prompt.version,
                "prompt_sha256": prompt.sha256,
                "model_info": self.adapter.model_info.as_dict(),
                "attempts": attempts,
                "finished_at": now_utc_iso(),
            },
        )


"""控制器级错误码与异常。

为什么不用裸 Exception：报告与 CLI 要给出**可执行建议**，而不是一句"失败了"。
每个码都必须能被 `errors.HINTS` 映射到一句人话，测试里会检查没有漏网的码。
"""

from __future__ import annotations

from typing import Any


class Codes:
    # 预检
    PACKAGE_MISSING = "PACKAGE_MISSING"
    PACKAGE_HASH_MISMATCH = "PACKAGE_HASH_MISMATCH"
    PACKAGE_INVALID = "PACKAGE_INVALID"
    FINGERPRINT_INVALID = "FINGERPRINT_INVALID"
    CAPABILITY_MISSING = "CAPABILITY_MISSING"
    LISTENER_NOT_RUNNING = "LISTENER_NOT_RUNNING"
    PHOTO_NOT_FOUND = "PHOTO_NOT_FOUND"
    PHOTO_PATH_NOT_ASCII = "PHOTO_PATH_NOT_ASCII"
    SESSION_LOCKED = "SESSION_LOCKED"
    SESSION_RESUME_INVALID = "SESSION_RESUME_INVALID"
    CONFIG_INVALID = "CONFIG_INVALID"
    # 模型交互
    MODEL_RESPONSE_INVALID = "MODEL_RESPONSE_INVALID"
    MODEL_RESPONSE_TIMEOUT = "MODEL_RESPONSE_TIMEOUT"
    EVIDENCE_UNVERIFIED = "EVIDENCE_UNVERIFIED"
    PLAN_REJECTED = "PLAN_REJECTED"
    # LR 往返
    APPLY_STATE_UNKNOWN = "APPLY_STATE_UNKNOWN"
    APPLY_FAILED = "APPLY_FAILED"
    RENDER_FAILED = "RENDER_FAILED"
    RENDER_HASH_MISMATCH = "RENDER_HASH_MISMATCH"
    SNAPSHOT_INCOMPLETE = "SNAPSHOT_INCOMPLETE"
    RESTORE_FAILED = "RESTORE_FAILED"
    RESTORE_VERIFY_MISMATCH = "RESTORE_VERIFY_MISMATCH"
    EXPORT_FAILED = "EXPORT_FAILED"
    EXPORT_HASH_MISMATCH = "EXPORT_HASH_MISMATCH"
    # 会话
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    NO_ROUND_COMPLETED = "NO_ROUND_COMPLETED"


HINTS: dict[str, str] = {
    Codes.PACKAGE_MISSING: "指纹包目录不存在或不完整，先跑离线建档导出包。",
    Codes.PACKAGE_HASH_MISMATCH: "包内文件被改动过。拒绝进入 LR 阶段，请重新导出指纹包。",
    Codes.PACKAGE_INVALID: "manifest / fingerprint.json 结构不符，按 fingerprint_schema_version 检查。",
    Codes.FINGERPRINT_INVALID: "fingerprint.json 缺指纹 id 或哈希，或状态与 mode 不匹配。",
    Codes.CAPABILITY_MISSING: "插件能力不足：在 Lightroom 里点一次 “AI Color: Start Automation Listener…”。",
    Codes.LISTENER_NOT_RUNNING: "队列无人消费：自动化监听未开启，job 会一直躺在 inbox。",
    Codes.PHOTO_NOT_FOUND: "照片不在当前 catalog，或路径与 LR 记录不一致。",
    Codes.PHOTO_PATH_NOT_ASCII: "插件队列与 io 只支持 ASCII 路径；请改用 ASCII 的工作副本。",
    Codes.SESSION_LOCKED: "同一张照片已有进行中的会话，先结束或删除锁文件。",
    Codes.SESSION_RESUME_INVALID: "会话已结束或状态不可恢复；可用 inspect 查看，必要时重开新会话。",
    Codes.CONFIG_INVALID: "配置越界：max_rounds 只接受 1–5，不做宽容钳制。",
    Codes.MODEL_RESPONSE_INVALID: "模型响应不符合 schema；允许重做一次且不消耗 LR 轮次。",
    Codes.MODEL_RESPONSE_TIMEOUT: "等待模型响应超时；会话可 resume，已完成的轮次不会作废。",
    Codes.EVIDENCE_UNVERIFIED: "评价里的 evidence 无法对上本轮 metrics.json，拒绝该评价。",
    Codes.PLAN_REJECTED: "本轮参数全部被守卫拒绝（越界或不支持），未写入 LR。",
    Codes.APPLY_STATE_UNKNOWN: "apply 超时且状态未知：先用原 job id 查结果，不得盲目重写。",
    Codes.APPLY_FAILED: "apply 失败，本轮不渲染；有最佳轮则恢复并导出。",
    Codes.RENDER_FAILED: "预览导出失败；模型未看到图，不得进入下一轮。",
    Codes.RENDER_HASH_MISMATCH: "预览对应的设置不是本轮写入的那一版，该轮作废。",
    Codes.SNAPSHOT_INCOMPLETE: "快照读回不完整，禁止承诺精确恢复；报告标 pilot。",
    Codes.RESTORE_FAILED: "恢复最佳轮失败，返回已有最佳预览并明确报错。",
    Codes.RESTORE_VERIFY_MISMATCH: "恢复后校验不一致，不得导出成片。",
    Codes.EXPORT_FAILED: "最终导出失败，不拿其它轮次冒充 final。",
    Codes.EXPORT_HASH_MISMATCH: "当前设置不是最佳快照，拒绝导出。",
    Codes.BUDGET_EXHAUSTED: "轮次预算用尽，正常收敛到最佳轮。",
    Codes.NO_ROUND_COMPLETED: "没有任何一轮形成有效反馈，无法选最佳轮。",
}


class DecodeError(Exception):
    """带稳定错误码、阶段和建议的异常。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str | None = None,
        detail: Any = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.detail = detail
        self.retryable = retryable

    @property
    def hint(self) -> str:
        return HINTS.get(self.code, "未收录的错误码，查看 detail。")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "stage": self.stage,
            "retryable": self.retryable,
            "detail": self.detail,
        }

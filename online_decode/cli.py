"""命令行入口。

    python -m online_decode run     --photo <path> --package <dir> [--max-rounds 5] [--intent ...]
    python -m online_decode inspect --session sessions/s_xxx
    python -m online_decode verify-package --package examples/sample_package
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load as load_config
from .errors import DecodeError
from .fingerprint.resolver import verify_package
from .util import read_json


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_verify_package(args) -> int:
    try:
        pkg = verify_package(args.package)
    except DecodeError as exc:
        _print(exc.to_dict())
        return 2
    _print({
        "ok": True,
        "fingerprint_id": pkg.fingerprint_id,
        "fingerprint_hash": pkg.fingerprint_hash,
        "status": pkg.status,
        "application_ready": pkg.application_ready,
        "buckets": pkg.bucket_ids,
        "files_checked": len(pkg.file_hashes),
    })
    return 0


def _build_runtime(cfg, args, *, reuse_existing: bool = False):
    from .drivers.agent_driver import AgentDriver
    from .feedback.lr_port import QueueLrPort
    from .session import SessionController

    if cfg.driver == "agent":
        def announce(task, round_no, req, resp):
            print(f"[agent] {task} r{round_no:02d} 请求包：{req}")
            print(f"        写回响应：{resp}", flush=True)

        driver = AgentDriver(
            cfg.workspace, timeout_sec=cfg.request_timeout_sec,
            on_request=announce, reuse_existing=reuse_existing,
        )
    else:
        raise DecodeError(
            "CONFIG_INVALID",
            f"driver={cfg.driver} 需要显式注入，CLI 只实现了 agent",
        )

    if not cfg.mcp_repo:
        raise DecodeError("CONFIG_INVALID", "缺少 --mcp-repo（ai_color_mcp 所在目录）")
    port = QueueLrPort(cfg.mcp_repo, timeout=cfg.job_timeout_sec)
    return SessionController(cfg, port, driver)


def cmd_run(args) -> int:
    cfg = load_config(args.config, {
        "max_rounds": args.max_rounds,
        "workspace": args.workspace,
        "driver": args.driver,
        "mcp_repo": args.mcp_repo,
        "request_timeout_sec": args.timeout,
    })
    controller = _build_runtime(cfg, args)
    report = controller.run(args.photo, args.package, user_intent=args.intent)
    _print(report)
    return 0 if report.get("ok") else 1


def cmd_resume(args) -> int:
    cfg = load_config(args.config, {
        "workspace": args.workspace,
        "driver": args.driver,
        "mcp_repo": args.mcp_repo,
        "request_timeout_sec": args.timeout,
    })
    controller = _build_runtime(cfg, args, reuse_existing=True)
    report = controller.resume(args.session)
    _print(report)
    return 0 if report.get("ok") else 1


def cmd_inspect(args) -> int:
    root = Path(args.session)
    session = read_json(root / "session.json")
    report = read_json(root / "decode_report.json")
    if session is None:
        print(f"找不到会话：{root}", file=sys.stderr)
        return 2
    _print({
        "session_id": session.get("session_id"),
        "status": session.get("status"),
        "stop_reason": session.get("stop_reason"),
        "max_rounds": session.get("max_rounds"),
        "stop_on_single_no_progress": session.get("stop_on_single_no_progress"),
        "rounds": [
            {
                "round": item.get("round"),
                "ok": item.get("ok"),
                "improvement": (item.get("evaluation") or {}).get("improvement"),
                "decision": (item.get("evaluation") or {}).get("decision"),
                "error": (item.get("error") or {}).get("code"),
            }
            for item in session.get("rounds") or []
        ],
        "best": (session.get("best") or {}).get("round"),
        "final": (session.get("final") or {}).get("path"),
        "has_report": report is not None,
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="online_decode", description="在线个性化解码")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="跑一次单图会话")
    run.add_argument("--photo", required=True)
    run.add_argument("--package", required=True)
    run.add_argument("--intent", default=None)
    run.add_argument("--max-rounds", type=int, default=None)
    run.add_argument("--config", default=None)
    run.add_argument("--workspace", default=".")
    run.add_argument("--driver", default="agent")
    run.add_argument("--mcp-repo", default=None)
    run.add_argument("--timeout", type=float, default=None)
    run.set_defaults(func=cmd_run)

    inspect = sub.add_parser("inspect", help="查看会话摘要")
    inspect.add_argument("--session", required=True)
    inspect.set_defaults(func=cmd_inspect)

    resume = sub.add_parser("resume", help="从中断处继续一次会话")
    resume.add_argument("--session", required=True)
    resume.add_argument("--config", default=None)
    resume.add_argument("--workspace", default=".")
    resume.add_argument("--driver", default="agent")
    resume.add_argument("--mcp-repo", default=None)
    resume.add_argument("--timeout", type=float, default=None)
    resume.set_defaults(func=cmd_resume)

    verify = sub.add_parser("verify-package", help="校验指纹包完整性")
    verify.add_argument("--package", required=True)
    verify.set_defaults(func=cmd_verify_package)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DecodeError as exc:
        _print(exc.to_dict())
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

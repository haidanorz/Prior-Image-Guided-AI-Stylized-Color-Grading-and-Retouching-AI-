"""命令行入口。

用法：
    python -m offline_preprocess --input-dir 原图目录 --output-dir 输出目录
    python -m offline_preprocess --input-dir ... --output-dir ... --config my.json
    python -m offline_preprocess verify --output-dir 输出目录
"""

from __future__ import annotations

import argparse
import json
import sys

from .errors import FatalError
from .run import run
from .verify import verify_output


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "verify":
        return _verify_command(args[1:])
    return _run_command(args)


def _run_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m offline_preprocess",
        description="离线建档预处理：原片 -> sRGB 预览 + 客观特征 JSON + 可追溯清单",
    )
    parser.add_argument("--input-dir", required=True, help="原始照片目录（只读）")
    parser.add_argument("--output-dir", required=True, help="输出目录（不得等于输入目录）")
    parser.add_argument("--config", default=None, help="可选 JSON 配置文件")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--recursive", dest="recursive", action="store_true", help="递归扫描（默认）")
    group.add_argument("--no-recursive", dest="recursive", action="store_false", help="只扫描顶层目录")
    parser.set_defaults(recursive=None)
    parser.add_argument("--overwrite", action="store_true", help="忽略既有清单，全部重新处理")
    parser.add_argument("--fail-on-error", action="store_true", help="存在 error 记录时以退出码 1 结束")
    parser.add_argument("--quiet", action="store_true", help="不打印逐图进度")
    parser.add_argument("--json", action="store_true", help="以 JSON 打印运行摘要")
    args = parser.parse_args(argv)

    overrides: dict = {}
    if args.recursive is not None:
        overrides["recursive"] = bool(args.recursive)
    if args.overwrite:
        overrides["overwrite"] = True

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    try:
        result = run(
            args.input_dir,
            args.output_dir,
            config_path=args.config,
            overrides=overrides,
            progress=progress,
        )
    except FatalError as exc:
        print(f"致命错误 [{exc.code}]：{exc.message}", file=sys.stderr)
        return 2

    report = result["report"]
    counts = report["status_counts"]
    summary = {
        "output_dir": result["output_dir"],
        "input_file_count": report["input_file_count"],
        "status_counts": counts,
        "warning_count": report["warning_count"],
        "error_code_counts": report["error_code_counts"],
        "near_duplicate_group_count": report["near_duplicate_group_count"],
        "counts_consistent": report["counts_consistent"],
        "manifest": result["manifest_path"],
        "run_report": result["report_path"],
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("")
        print(f"输入文件：{report['input_file_count']}")
        print(
            "状态：ready={ready} duplicate_exact={duplicate_exact} skipped={skipped} error={error}".format(
                **counts
            )
        )
        print(f"警告：{report['warning_count']}    近重复候选组：{report['near_duplicate_group_count']}")
        print(f"清单：{result['manifest_path']}")
        print(f"报告：{result['report_path']}")
        if not report["counts_consistent"]:
            print("警告：状态计数未闭环，请检查报告", file=sys.stderr)

    if args.fail_on_error and counts.get("error", 0) > 0:
        return 1
    return 0


def _verify_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m offline_preprocess verify", description="校验输出目录完整性（只读）")
    parser.add_argument("--output-dir", required=True, help="待校验的输出目录")
    parser.add_argument("--json", action="store_true", help="以 JSON 打印完整结果")
    args = parser.parse_args(argv)

    result = verify_output(args.output_dir)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"目录：{result['output_dir']}")
        print(f"清单记录：{result.get('checked', 0)}    ready：{result.get('ready', 0)}")
        if result["ok"]:
            print("校验通过")
        else:
            print(f"校验失败，问题 {len(result['problems'])} 项：")
            for problem in result["problems"][:50]:
                location = f" [{problem['image_id']}]" if problem.get("image_id") else ""
                print(f"  - {problem['code']}{location}: {problem['message']}")
            if len(result["problems"]) > 50:
                print(f"  ... 其余 {len(result['problems']) - 50} 项省略")
    return 0 if result["ok"] else 1

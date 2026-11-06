"""命令行入口。

用法：
    python -m offline_profile run --input 预处理输出 --build-root profile_build --mock-dir mock/
    python -m offline_profile verify --build-root profile_build
"""

from __future__ import annotations

import argparse
import json
import sys

from .errors import FatalError
from .agent_flow import preview_build
from .pipeline import run
from .fingerprint_package import export_fingerprint_package
from .narrative import collect_bucket_descriptions, submit_bucket_narrative
from .xmp_export import export_pilot_xmp
from .verify import verify_build


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "verify":
        return _verify_command(args[1:])
    if args and args[0] == "checkpoint":
        return _checkpoint_command(args[1:])
    if args and args[0] == "xmp-export":
        return _xmp_command(args[1:])
    if args and args[0] == "package":
        return _package_command(args[1:])
    if args and args[0] == "narrative-collect":
        return _narrative_collect(args[1:])
    if args and args[0] == "narrative-submit":
        return _narrative_submit(args[1:])
    if args and args[0] == "run":
        args = args[1:]
    return _run_command(args)


def _run_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m offline_profile",
        description="离线建档后处理：单图分析 → 场景分桶 → 训练/留出划分",
    )
    parser.add_argument("--input", required=True, help="预处理输出目录（只读）")
    parser.add_argument("--build-root", required=True, help="构建输出目录")
    parser.add_argument("--mock-dir", default=None, help="响应目录（mock 或 replay 共用）")
    parser.add_argument("--adapter", choices=("mock", "replay"), default=None, help="覆盖配置中的 adapter")
    parser.add_argument("--config", default=None, help="可选 JSON 配置文件")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖旧构建")
    parser.add_argument("--strict", action="store_true", help="覆盖率不足或存在失败分析时退出码为 1")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    try:
        result = run(
            args.input,
            args.build_root,
            config_path=args.config,
            mock_dir=args.mock_dir,
            overrides={
                **({"overwrite": True} if args.overwrite else {}),
                **({"adapter": args.adapter} if args.adapter else {}),
            }
            or None,
            progress=progress,
        )
    except FatalError as exc:
        print(f"致命错误 [{exc.code}]：{exc.message}", file=sys.stderr)
        return 2

    report = result["report"]
    analysis = report["analysis"]
    summary = {
        "build_id": report["build_id"],
        "build_root": result["build_root"],
        "input_ready": report["preprocess"]["manifest_total"] - report["preprocess"]["non_ready"],
        "analysis_counts": analysis["counts"],
        "coverage": analysis["coverage"],
        "coverage_ok": analysis["coverage_ok"],
        "assignments": report["assignments"],
        "split_buckets": report["split"]["buckets"],
        "warnings": report["warnings"],
        "manifest": result["manifest_path"],
        "run_report": result["report_path"],
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("")
        print(f"构建：{report['build_id']}")
        print(
            "分析：normalized={normalized} invalid={invalid} error={error}  覆盖率={coverage:.3f}".format(
                **analysis["counts"], coverage=analysis["coverage"]
            )
        )
        levels = report["assignments"]["levels"]
        print(f"分桶：core={levels.get('core', 0)} associate={levels.get('associate', 0)} unassigned={levels.get('unassigned', 0)}")
        for bucket_id, entry in report["split"]["buckets"].items():
            print(
                f"  {bucket_id}: n_core={entry['n_core']} status={entry['status']} "
                f"split={entry['split_status']} train={entry['n_train']} holdout={entry['n_holdout']}"
            )
        for warning in report["warnings"]:
            print(f"警告：{warning}")
        print(f"清单：{result['manifest_path']}")

    if args.strict and (not analysis["coverage_ok"] or analysis["counts"]["error"] or analysis["counts"]["invalid"]):
        return 1
    return 0


def _narrative_collect(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m offline_profile narrative-collect", description="生成桶级逐图描述与归纳 prompt")
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = collect_bucket_descriptions(args.build_root, args.bucket)
    except Exception as exc:
        print(f"收集失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        payload = dict(result)
        payload.pop("prompt", None)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{args.bucket}: {result['n']} 张训练成员")
        print(f"描述：{result['descriptions_file']}")
        print(f"prompt：{result['prompt_file']}")
    return 0


def _narrative_submit(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m offline_profile narrative-submit", description="校验并写入桶级语言描述")
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--file", required=True, help="父代理生成的 narrative JSON")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        with open(args.file, encoding="utf-8") as handle:
            payload = json.load(handle)
        narrative = submit_bucket_narrative(args.build_root, args.bucket, payload)
    except Exception as exc:
        print(f"提交失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(narrative, ensure_ascii=False, indent=2))
    else:
        print(f"已写入 buckets/{args.bucket}/narrative.json")
        print(f"narrative_hash={narrative['narrative_hash']}")
    return 0


def _package_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m offline_profile package", description="导出离线编码器的特征指纹包")
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = export_fingerprint_package(args.build_root, args.out_dir, overwrite=args.overwrite)
    except Exception as exc:
        print(f"打包失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(f"输出：{args.out_dir}")
        print(f"package_id={manifest['package_id']}  files={manifest['counts']['files']}  buckets={manifest['counts']['buckets']}")
        print(f"fingerprint_id={manifest['fingerprint_id']}  status={manifest['status']}")
    return 0


def _xmp_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m offline_profile xmp-export", description="导出 pilot XMP（仅试看）")
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--strict", action="store_true", help="不写入不稳定的 contrast/clarity")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = export_pilot_xmp(args.build_root, args.out_dir, include_unstable=not args.strict)
    except Exception as exc:
        print(f"导出失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(f"输出：{args.out_dir}")
        for preset in manifest["presets"]:
            print(f"  {preset['file']}  candidate={preset['candidate_params']}  pilot_only={preset['pilot_only_params']}")
        print("状态：pilot_only（application_ready=false）")
    return 0


def _checkpoint_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m offline_profile checkpoint", description="只读检查点：试算分桶与划分")
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = preview_build(args.build_root)
    except FatalError as exc:
        print(f"致命错误 [{exc.code}]：{exc.message}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        counts = result["counts"]
        levels = result["assignments"]["levels"]
        print(f"构建：{result['build_id']}  状态：{result['status']}")
        print(f"任务：submitted={counts['submitted']} pending={counts['pending']} invalid={counts['invalid']}")
        print(f"等级：core={levels['core']} associate={levels['associate']} unassigned={levels['unassigned']}")
        for bucket_id, entry in result["buckets"].items():
            fold_note = f" folds={entry.get('n_folds')}" if entry.get("split_status") == "kfold" else ""
            print(
                f"  {bucket_id}: n_core={entry['n_core']} status={entry['status']} "
                f"split={entry['split_status']}{fold_note} reason={entry['reason']}"
            )
        print(f"split_hash={result['split_hash']}")
        print(result["note"])
    return 0


def _verify_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m offline_profile verify", description="校验构建目录（只读）")
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = verify_build(args.build_root)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"目录：{result['build_root']}")
        print(f"记录：{result.get('checked', 0)}  normalized：{result.get('normalized', 0)}")
        if result["ok"]:
            print("校验通过")
        else:
            print(f"校验失败，问题 {len(result['problems'])} 项：")
            for problem in result["problems"][:50]:
                location = f" [{problem['image_id']}]" if problem.get("image_id") else ""
                print(f"  - {problem['code']}{location}: {problem['message']}")
    return 0 if result["ok"] else 1

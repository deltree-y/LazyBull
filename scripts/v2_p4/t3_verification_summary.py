# -*- coding: utf-8 -*-
"""P4-T3：由原始日志与门 2b verdict 生成机器汇总（阶段专用）。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_log(path: Path) -> str:
    raw = path.read_bytes()
    return raw.decode("utf-16" if b"\x00" in raw else "utf-8", errors="replace")


def _pytest_result(path: Path):
    matches = re.findall(r"=+\s*([\d,]+) passed(?:, (\d+) warnings)? in ([\d.]+)s", _read_log(path))
    if not matches:
        raise AssertionError(f"未找到 pytest 通过行: {path}")
    passed, warnings, seconds = matches[-1]
    return {
        "passed": int(passed.replace(",", "")),
        "warnings": int(warnings or 0),
        "seconds": float(seconds),
        "log": str(path),
    }


def _complexity_result(path: Path):
    matches = re.findall(
        r"检查文件 (\d+) 个；软超限 (\d+) 条（拆分义务），新增硬超限 (\d+) 条，"
        r"存量基线 (\d+) 条（登记不阻塞），豁免 (\d+) 条",
        _read_log(path),
    )
    if not matches:
        raise AssertionError(f"未找到复杂度汇总: {path}")
    files, soft, new_hard, baseline, exempt = matches[-1]
    return {
        "files": int(files),
        "soft": int(soft),
        "new_hard": int(new_hard),
        "baseline": int(baseline),
        "exempt": int(exempt),
        "log": str(path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--t3-log", required=True)
    parser.add_argument("--related-log", required=True)
    parser.add_argument("--full-log", required=True)
    parser.add_argument("--complexity-src-log", required=True)
    parser.add_argument("--complexity-all-log", required=True)
    parser.add_argument("--gate2b-verdict", required=True)
    parser.add_argument("--expected-t3", type=int, default=61)
    parser.add_argument("--expected-related", type=int, default=None)
    parser.add_argument("--min-full", type=int, default=4219)
    args = parser.parse_args(argv)
    evidence = Path(args.evidence_dir)
    t3 = _pytest_result(Path(args.t3_log))
    related = _pytest_result(Path(args.related_log))
    full = _pytest_result(Path(args.full_log))
    complexity_src = _complexity_result(Path(args.complexity_src_log))
    complexity_all = _complexity_result(Path(args.complexity_all_log))
    inuse = _load(evidence / "inuse_closure.json")
    normalize = _load(evidence / "normalize_divergence.json")
    gate2b = _load(Path(args.gate2b_verdict))
    passed = (
        t3["passed"] == args.expected_t3
        and (args.expected_related is None or related["passed"] == args.expected_related)
        and full["passed"] >= args.min_full
        and complexity_src["new_hard"] == 0
        and complexity_all["new_hard"] == 0
        and inuse["conservation_ok"]
        and gate2b["verdict"] == "PASS"
        and gate2b["worktree_binding"]["before_after_same"]
        and gate2b["worktree_binding"]["current_matches_after"]
        and normalize["read_only_inputs"]
        and not normalize["input_identity"]["after_drift"]
    )
    payload = {
        "run_id": evidence.name,
        "verdict": "PASS" if passed else "FAIL",
        "tests": {"t3": t3, "related": related, "full": full},
        "complexity": {"src": complexity_src, "src_and_scripts": complexity_all},
        "inuse": {
            key: inuse[key]
            for key in (
                "static_closure_count",
                "inuse_count",
                "all_modules_count",
                "notin_count",
                "conservation_ok",
            )
        },
        "normalize": {
            "sample": normalize["sample"],
            "actual_stats": normalize["actual_stats"],
            "hypothetical_kelly_window_stats": normalize["hypothetical_kelly_window_stats"],
            "input_files": len(normalize["input_identity"]["files"]),
            "after_drift": normalize["input_identity"]["after_drift"],
            "attribution": normalize["attribution"],
        },
        "gate2b": gate2b,
    }
    out = evidence / "verification_summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "out": str(out)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

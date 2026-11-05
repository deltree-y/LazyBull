# -*- coding: utf-8 -*-
"""P6 T0.5 生成时间测量入口（设计 §7.2：完整进程 5 次，每次 < 10,000ms）。

被测对象固定为 ``reference_generator.py``（参考生成器）：只验证**计时协议**，不宣称 P6 正式
报告达成性能目标。每次在**新子进程**中执行完整 source→projection→HTML/CSV→manifest 链路，
父进程用 monotonic 墙钟计时；5 次输出必须逐字节一致（同一 ``generated_at``）。

结果写独立 validation 目录的 ``generation_performance.json``；不写生产数据根。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）；pytest 导入时不改全局
    sys.dont_write_bytecode = True
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p6 import env_probe  # noqa: E402
from src.lazybull.v2.reporting import contracts, perf_gate, validation  # noqa: E402

__all__ = ["GenerationSetupError", "main", "run"]

SCHEMA = "p6_t05_generation_performance/v1"
REFERENCE_SCRIPT = Path(__file__).resolve().parent / "reference_generator.py"
DEFAULT_GENERATED_AT = "20260101T000000Z"
SUBPROCESS_TIMEOUT = 180
#: 生成测量允许写的 reports 根（模块常量：测试可注入 tmp 根，禁止真实生产根写入）
_REPORTS_ROOT = (ROOT / contracts.REPORTS_ROOT).resolve()


class GenerationSetupError(RuntimeError):
    """生成测量输入/输出不合法或子进程失败。"""


def _digest_tree(root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": validation.sha256_file(path),
                }
            )
    return entries


def _assert_new_validation_out(out: Path) -> Path:
    reports_root = _REPORTS_ROOT
    target = out.resolve()
    if target.exists():
        raise GenerationSetupError(f"输出目录已存在，拒绝覆盖：{target}")
    validations_root = reports_root / "validations"
    try:
        relative = target.relative_to(validations_root)
    except ValueError as exc:
        raise GenerationSetupError(
            f"生成测量只允许写 {validations_root} 之下：{target}"
        ) from exc
    evidence_class = relative.parts[0] if relative.parts else ""
    if evidence_class not in contracts.EVIDENCE_CLASSES:
        raise GenerationSetupError(
            f"validation 根形态不合法（evidence_class 须 ∈ "
            f"{sorted(contracts.EVIDENCE_CLASSES)}）：{target}"
        )
    return target


def _run_once(corpus: Path, run_dir: Path, generated_at: str) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(REFERENCE_SCRIPT),
        "--corpus",
        str(corpus),
        "--out",
        str(run_dir),
        "--generated-at",
        generated_at,
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=SUBPROCESS_TIMEOUT,
        env=environment,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "elapsed_ms": elapsed_ms,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip()[-2000:],
    }


def run(
    corpus: Path | str,
    out: Path | str,
    *,
    case: str = "reference",
    repeat: int = contracts.THRESHOLDS["generation_repeats"],
    generated_at: str = DEFAULT_GENERATED_AT,
) -> dict[str, Any]:
    """执行 5 次完整进程生成测量并写 ``generation_performance.json``。"""
    if case != "reference":
        raise GenerationSetupError(f"T0.5 只支持 reference 计时协议，收到 case={case!r}")
    if repeat != contracts.THRESHOLDS["generation_repeats"]:
        raise GenerationSetupError(
            f"生成测量次数固定为 {contracts.THRESHOLDS['generation_repeats']}，收到 {repeat}"
        )
    corpus_root = Path(corpus).resolve()
    if not (corpus_root / "standard").is_dir():
        raise GenerationSetupError(f"语料缺少 standard/ 目录：{corpus_root}")
    target = _assert_new_validation_out(Path(out))
    validation.ensure_new_dir(target)
    work_root = target / "work"
    work_root.mkdir()

    runs: list[dict[str, Any]] = []
    for index in range(repeat):
        result = _run_once(corpus_root, work_root / f"run_{index}", generated_at)
        result["index"] = index
        result["outputs"] = (
            _digest_tree(work_root / f"run_{index}") if result["returncode"] == 0 else []
        )
        runs.append(result)

    reference_outputs = runs[0]["outputs"]
    deterministic = all(
        run_result["returncode"] == 0 and run_result["outputs"] == reference_outputs
        for run_result in runs
    )
    gate = perf_gate.judge_generation_series([run_result["elapsed_ms"] for run_result in runs])
    environment = env_probe.capture_environment(None)
    environment_gate = perf_gate.validate_environment(environment, require_browser=False)
    versions_gate = perf_gate.verify_acceptance_versions(
        {"jinja2": environment.get("jinja2_version")}, names=("jinja2",)
    )
    all_succeeded = all(run_result["returncode"] == 0 for run_result in runs)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "case": case,
        "protocol_reference_only": True,
        "generated_at": generated_at,
        "corpus": str(corpus_root),
        "repeat": repeat,
        "runs": runs,
        "deterministic": deterministic,
        "gates": {
            "generation": gate,
            "environment": environment_gate,
            "acceptance_versions": versions_gate,
            "subprocess": {"all_succeeded": all_succeeded},
        },
        "environment": environment,
    }
    payload["verdict"] = (
        "pass"
        if all(
            (
                gate["passed"],
                deterministic,
                all_succeeded,
                environment_gate["passed"],
                versions_gate["passed"],
            )
        )
        else "fail"
    )
    validation.write_json_new(target / "generation_performance.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P6 T0.5 生成时间测量（参考生成器，5 次）")
    parser.add_argument("--corpus", required=True, help="语料根（须含 standard/）")
    parser.add_argument(
        "--out", required=True, help="新的 validation 目录（须位于 data/reports/p6/validations/）"
    )
    parser.add_argument("--case", default="reference")
    parser.add_argument("--repeat", type=int, default=contracts.THRESHOLDS["generation_repeats"])
    parser.add_argument("--generated-at", default=DEFAULT_GENERATED_AT)
    args = parser.parse_args(argv)
    try:
        payload = run(
            args.corpus,
            args.out,
            case=args.case,
            repeat=args.repeat,
            generated_at=args.generated_at,
        )
    except GenerationSetupError as exc:
        print(f"[P6-T0.5] 生成测量未执行：{exc}", file=sys.stderr)
        return 2
    print(json.dumps({"verdict": payload["verdict"], "out": str(args.out)}, ensure_ascii=False))
    return 0 if payload["verdict"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""P6 T0.5 验收驱动（P6-M0.5：用户可复跑固定命令并查看原始 JSON）。

一次运行产出完整证据包 ``data/reports/p6/validations/research/<validation_id>/``：

1. 合成语料双跑（standard + stress）并比对 SHA；
2. 固定合成参考生成器产出 HTML/CSV/manifest；
3. 打开测量 5 冷 + 5 热（Playwright/Chromium）；
4. 生成测量 5 次完整进程（参考生成器）；
5. 契约冻结件 + 工作树/复杂度/依赖基线；
6. 汇总 ``t05_acceptance.json``（全部检查项与原始子证据路径）。

本驱动不提交、不改生产数据、不写报告根之外路径。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）；pytest 导入时不改全局
    sys.dont_write_bytecode = True
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p6 import (  # noqa: E402
    build_corpus,
    freeze_t05_contracts,
    measure_generation,
    measure_report,
    reference_generator,
)
from src.lazybull.v2.reporting import contracts, perf_gate, validation  # noqa: E402

__all__ = ["main", "run_acceptance"]

SCHEMA = "p6_t05_acceptance/v1"
REFERENCE_GENERATED_AT = "20260101T000000Z"
DOCS_FREEZE = ROOT / "docs" / "contracts" / "p6_t05_contract_freeze_20261105.json"


def _corpus_files(manifest: dict[str, Any]) -> dict[str, str]:
    files = {}
    for profile in manifest["profiles"].values():
        for entry in profile["files"]:
            files[entry["path"]] = entry["sha256"]
    return files


def _check(check_id: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "detail": detail}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _docs_contract_check() -> dict[str, Any]:
    if not DOCS_FREEZE.is_file():
        return _check("contract_docs_freeze", False, {"path": str(DOCS_FREEZE), "error": "missing"})
    record = _load_json(DOCS_FREEZE)
    passed = (
        record.get("contract_sha256") == contracts.frozen_contract_sha256()
        and record.get("contract") == contracts.as_frozen_payload()
    )
    return _check(
        "contract_docs_freeze",
        passed,
        {"path": str(DOCS_FREEZE), "contract_sha256": record.get("contract_sha256")},
    )


def _default_validation_id() -> str:
    """默认 validation_id：全小写数字形态，生成即通过冻结 pattern（评审 R1-1/R2-T05-05）。

    ``utc_stamp()`` 的大写 ``T``/``Z`` 只用于 ``generated_at`` 时间字段，不得进入路径 id。
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"p6_t05_acceptance_{stamp}"


def _run_open_measurement(
    html_path: Path, open_dir: Path
) -> tuple[dict[str, Any], subprocess.CompletedProcess, list[str]]:
    """经 CLI 子进程执行打开测量并载入原始记录（评审 R2-T05-03 续）。

    返回（performance.json 内容, 子进程结果, 实际命令）；执行层保存实际退出码，
    与 ``performance.json`` 内的退出码约定字段相区分。
    """
    command = [
        sys.executable,
        str(ROOT / "scripts" / "v2_p6" / "measure_report.py"),
        "--html",
        str(html_path),
        "--out",
        str(open_dir),
    ]
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise measure_report.MeasureSetupError(f"打开测量超时：{exc}") from exc
    if completed.returncode == 2:
        raise measure_report.MeasureSetupError(
            f"打开测量未执行：{completed.stderr.strip()[-500:]}"
        )
    if completed.returncode not in (0, 3):
        raise measure_report.MeasureSetupError(
            f"打开测量异常退出（{completed.returncode}）：{completed.stderr.strip()[-500:]}"
        )
    payload = json.loads((open_dir / "performance.json").read_text(encoding="utf-8"))
    return payload, completed, command


def run_acceptance(validation_id: str | None = None) -> dict[str, Any]:
    """执行 T0.5 全链验收，返回汇总 JSON（同时写 ``t05_acceptance.json``）。"""
    reports_root = ROOT / contracts.REPORTS_ROOT
    identifier = validation_id or _default_validation_id()
    out = validation.resolve_validation_dir(reports_root, "research", identifier)
    validation.ensure_new_dir(out)

    corpus_a = build_corpus.build(out / "corpus_a", profile="all")
    corpus_b = build_corpus.build(out / "corpus_b", profile="all")
    corpus_equal = corpus_a["content_sha256"] == corpus_b["content_sha256"] and _corpus_files(
        corpus_a
    ) == _corpus_files(corpus_b)
    checks = [
        _check(
            "corpus_double_run",
            corpus_equal,
            {
                "content_sha256": corpus_a["content_sha256"],
                "second_run": corpus_b["content_sha256"],
                "files": len(_corpus_files(corpus_a)),
            },
        )
    ]

    reference_dir = out / "reference"
    reference_generator.generate(
        out / "corpus_a", reference_dir, generated_at=REFERENCE_GENERATED_AT
    )
    html_path = reference_dir / "report.html"
    size_gate = perf_gate.judge_size(html_path.stat().st_size)
    checks.append(_check("reference_size", size_gate["passed"], size_gate))

    # R2-T05-03 续：打开测量经 CLI 子进程执行，由执行层保存实际命令/实际退出码/原始记录路径
    open_payload, open_process, open_command = _run_open_measurement(
        html_path, out / "open"
    )
    open_gates = open_payload["gates"]
    checks.append(
        _check(
            "open_5cold_5hot",
            open_payload["verdict"] == "pass",
            {
                "verdict": open_payload["verdict"],
                "exit_code": open_process.returncode,
                "command": open_command,
                "raw_record": "open/performance.json",
                "cold": open_gates["open"]["cold_count"],
                "hot": open_gates["open"]["hot_count"],
                "max_ms": open_gates["open"]["max_ms"],
                "request_offenders": open_gates["requests"]["offenders"],
                "external_references": open_payload["html"]["external_references"],
                "environment": open_gates["environment"],
                "versions": open_gates["acceptance_versions"],
            },
        )
    )

    generation_payload = measure_generation.run(out / "corpus_a", out / "generation")
    checks.append(
        _check(
            "generation_5runs",
            generation_payload["verdict"] == "pass",
            {
                "verdict": generation_payload["verdict"],
                "deterministic": generation_payload["deterministic"],
                "max_ms": generation_payload["gates"]["generation"]["max_ms"],
                "runs": len(generation_payload["runs"]),
            },
        )
    )

    contract_record = freeze_t05_contracts.build_contract_record()
    baseline = freeze_t05_contracts.collect_baseline(ROOT)
    validation.write_json_new(out / "contract_freeze.json", contract_record)
    validation.write_json_new(out / "baseline.json", baseline)

    html_sha = validation.sha256_file(html_path)
    isolation_passed = (
        open_payload["html"]["unchanged"]
        and open_payload["html"]["sha256_after"] == html_sha
        and _load_json(out / "corpus_a" / "corpus_manifest.json")["content_sha256"]
        == corpus_a["content_sha256"]
    )
    checks.append(
        _check(
            "validation_isolation",
            isolation_passed,
            {
                "html_sha256_after_measure": open_payload["html"]["sha256_after"],
                "html_sha256_now": html_sha,
                "corpus_content_sha256": corpus_a["content_sha256"],
            },
        )
    )

    checks.append(
        _check(
            "contract_freeze",
            contract_record["contract_sha256"] == contracts.frozen_contract_sha256(),
            {"contract_sha256": contract_record["contract_sha256"]},
        )
    )
    checks.append(_docs_contract_check())

    complexity = baseline["complexity"]
    checks.append(
        _check(
            "complexity_both_scopes",
            all(record["exit_code"] == 0 for record in complexity.values()),
            {name: record["exit_code"] for name, record in complexity.items()},
        )
    )

    installed = baseline["dependency"]["installed"]
    env = open_payload["environment"]
    checks.append(
        _check(
            "dependency_pins",
            installed.get("jinja2") == contracts.ACCEPTANCE_VERSIONS["jinja2"]
            and installed.get("playwright") == contracts.ACCEPTANCE_VERSIONS["playwright"]
            and str(env.get("chromium_version")) == contracts.ACCEPTANCE_VERSIONS["chromium"],
            {
                "installed": installed,
                "chromium": env.get("chromium_version"),
                "expected": dict(contracts.ACCEPTANCE_VERSIONS),
            },
        )
    )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "validation_id": identifier,
        "generated_at": validation.utc_stamp(),
        "verdict": "pass" if all(check["passed"] for check in checks) else "fail",
        "checks": checks,
        "artifacts": {
            "corpus_a": "corpus_a/corpus_manifest.json",
            "reference": "reference/report.html",
            "open": "open/performance.json",
            "generation": "generation/generation_performance.json",
            "contract": "contract_freeze.json",
            "baseline": "baseline.json",
        },
        "environment": env,
    }
    validation.write_json_new(out / "t05_acceptance.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P6 T0.5 全链验收（P6-M0.5 可复跑命令）")
    parser.add_argument(
        "--validation-id", default=None, help="可选 validation_id（默认 UTC 时间戳）"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = run_acceptance(args.validation_id)
    except (
        build_corpus.CorpusError,
        reference_generator.ReferenceError,
        measure_report.MeasureSetupError,
        measure_generation.GenerationSetupError,
        validation.ValidationError,
    ) as exc:
        print(f"[P6-T0.5] 验收未完成：{exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"verdict": payload["verdict"], "validation_id": payload["validation_id"]},
            ensure_ascii=False,
        )
        if args.json
        else f"T0.5 验收 {payload['verdict']}：{payload['validation_id']}"
    )
    return 0 if payload["verdict"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())

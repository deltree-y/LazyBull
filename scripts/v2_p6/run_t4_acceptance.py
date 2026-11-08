# -*- coding: utf-8 -*-
"""P6 T4 验收驱动（P6-M4 可复跑证据包）。

一次运行产出 ``data/reports/p6/validations/research/<validation_id>/t4_acceptance.json``：

1. T4 八组 pytest + T3 十一组 + T1 七组 + T2 七组 + T0.5 四组 + 治理面（前序阶段不回归）；
2. 复杂度 hook 双口径（``src/lazybull`` 与 ``src/lazybull scripts``）；
3. 语料确定性与增量语义（tiny 档双跑 ``content_sha256`` 一致；上一快照 + 新 run）；
4. 标准/压力两档：
   - 上一快照**全量批任务**发布一次（耗时只登记，不入 10s 交互门）；
   - **完整进程** 5 次增量生成计时（``--previous`` + 单新 run），逐次 <10,000ms 且五件字节一致；
   - 真实发布 + 精确写审计（``audit/readonly_audit_<profile>.json``）；
   - 两份 CSV 守恒（行数 = run 数 / 5×run 数、键唯一、``index_mode=full``）；
   - 已发布 run 的 manifest 自封存 + 链复算；
   - 5 冷 + 5 热打开测量（``measure_report``）；
   - 页面体积门 ≤3,000,000 字节；
5. canonical Linux 验收包（固定四文件 + 期望哈希 + 复跑命令；真实 Linux 执行门归 T5）；
6. 汇总 verdict（任一检查失败即 fail，原始终端输出保留在载荷中）。

写域：本驱动只写本次新 validation 目录、其兄弟 ``_audit_*``/``_open_*``/``_linux_*`` 目录、
``temp/`` 语料与工作目录；不写生产数据根。
"""

# 复杂度豁免：阶段专用验收驱动（scripts/v2_p6，不入生产链路，见 CLAUDE.md §7.5）

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback

if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）
    sys.dont_write_bytecode = True
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p6 import build_t4_corpus, env_probe, measure_report  # noqa: E402
from src.lazybull.v2.reporting import (  # noqa: E402
    contracts,
    manifest,
    perf_gate,
    publish,
    trend_contract,
    trend_csv,
    trend_portability,
    validation,
)

__all__ = ["AcceptanceSetupError", "main", "run"]

T4_TEST_FILES: tuple[str, ...] = (
    "tests/test_v2_p6_t4_metrics.py",
    "tests/test_v2_p6_t4_identity.py",
    "tests/test_v2_p6_t4_index.py",
    "tests/test_v2_p6_t4_order.py",
    "tests/test_v2_p6_t4_portability.py",
    "tests/test_v2_p6_t4_isolation.py",
    "tests/test_v2_p6_t4_history.py",
    "tests/test_v2_p6_t4_perf_recompute.py",
    "tests/test_v2_p6_t4_boundaries.py",
)
T3_TEST_FILES: tuple[str, ...] = (
    "tests/test_v2_p6_t3_chain_delta.py",
    "tests/test_v2_p6_t3_comparability.py",
    "tests/test_v2_p6_t3_topk.py",
    "tests/test_v2_p6_t3_stability.py",
    "tests/test_v2_p6_t3_regime.py",
    "tests/test_v2_p6_t3_attribution.py",
    "tests/test_v2_p6_t3_cost.py",
    "tests/test_v2_p6_t3_csv.py",
    "tests/test_v2_p6_t3_missing.py",
    "tests/test_v2_p6_t3_perf.py",
    "tests/test_v2_p6_t3_guard.py",
    "tests/test_v2_p6_t3_reviewer3_20261106.py",
)
T1_TEST_FILES: tuple[str, ...] = (
    "tests/test_v2_p6_t1_snapshot.py",
    "tests/test_v2_p6_t1_authority.py",
    "tests/test_v2_p6_t1_instructions.py",
    "tests/test_v2_p6_t1_config_model.py",
    "tests/test_v2_p6_t1_sources.py",
    "tests/test_v2_p6_t1_manifest.py",
    "tests/test_v2_p6_t1_negatives.py",
)
T2_TEST_FILES: tuple[str, ...] = (
    "tests/test_v2_p6_t2_render_states.py",
    "tests/test_v2_p6_t2_evidence.py",
    "tests/test_v2_p6_t2_security.py",
    "tests/test_v2_p6_t2_selfcontained.py",
    "tests/test_v2_p6_t2_publish.py",
    "tests/test_v2_p6_t2_perf.py",
    "tests/test_v2_p6_t2_determinism.py",
)
GOVERNANCE_TEST_FILES: tuple[str, ...] = (
    "tests/test_v2_p6_t05_corpus.py",
    "tests/test_v2_p6_t05_deps.py",
    "tests/test_v2_p6_t05_isolation.py",
    "tests/test_v2_p6_t05_perf_gate.py",
    "tests/test_check_inuse_closure.py",
    "tests/test_check_complexity.py",
)
#: 最少通过数（实测 285：T4 20 + T3 77 + T1 95 + T2 47 + T0.5 13 + 棘轮 33）
EXPECTED_MIN_PASSED = 280
BASELINE = Path("scripts/check/complexity_baseline.json")
REPORT_SCRIPT = Path("scripts/p6_report.py")
#: 计时用固定 generated_at（同输入可比字节；发布用新墙钟）
WORKFLOW_GENERATED_AT = "20261106T090000Z"
#: 两档语料（上一快照 run 数 = runs + legacy - 1，新 run 数 = 1）
PROFILE_PLAN: dict[str, dict[str, int]] = {
    "standard": {"runs": 400, "legacy": 100, "runs_previous": 399, "total": 500},
    "stress": {"runs": 1600, "legacy": 400, "runs_previous": 1599, "total": 2000},
}


class AcceptanceSetupError(RuntimeError):
    """验收驱动前置条件不满足（输出目录形态非法/已存在）。"""


def _default_validation_id() -> str:
    return "p6_t4_acceptance_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _assert_new_validation_out(path: Path) -> Path:
    """只允许 ``validations/<evidence_class>/<id>`` 且必须不存在。"""
    target = Path(path).resolve()
    validations_root = (ROOT / contracts.REPORTS_ROOT / "validations").resolve()
    try:
        relative = target.relative_to(validations_root)
    except ValueError as exc:
        raise AcceptanceSetupError(f"输出目录不在 validation 根下：{target}") from exc
    if len(relative.parts) != 2 or relative.parts[0] not in contracts.EVIDENCE_CLASSES:
        raise AcceptanceSetupError(f"输出目录形态非法：{relative}")
    validation.validate_validation_id(relative.parts[1])
    if target.exists():
        raise AcceptanceSetupError(f"validation 目录已存在，拒绝覆盖：{target}")
    return target


def _run(command: Sequence[str], *, timeout: int = 3600) -> dict[str, Any]:
    """执行子进程并记录实际命令/退出码/首尾输出（不吞错、不重定向到 logs）。"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        list(command),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
        timeout=timeout,
    )
    return {
        "command": [str(item) for item in command],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr_tail": [line for line in completed.stderr.splitlines() if line.strip()][-5:],
    }


def _parse_pytest_counts(lines: Sequence[str]) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for line in lines:
        for key in counts:
            match = re.search(rf"(\d+)\s+{key}", line)
            if match:
                counts[key] = int(match.group(1))
    return counts


def run_tests() -> dict[str, Any]:
    """运行 T4 + T3 + T1 + T2 + 治理面定向集（禁字节码与 pytest 缓存）。"""
    files = list(T4_TEST_FILES) + list(T3_TEST_FILES) + list(T1_TEST_FILES)
    files += list(T2_TEST_FILES) + list(GOVERNANCE_TEST_FILES)
    result = _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files])
    tail = [line for line in result["stdout"].splitlines() if line.strip()][-3:]
    counts = _parse_pytest_counts(tail)
    return {
        "command": result["command"],
        "exit_code": result["exit_code"],
        "stdout_tail": tail,
        "stderr_tail": result["stderr_tail"],
        "counts": counts,
        "expected_min_passed": EXPECTED_MIN_PASSED,
        "passed": (
            result["exit_code"] == 0
            and counts["failed"] == 0
            and counts["skipped"] == 0
            and counts["passed"] >= EXPECTED_MIN_PASSED
        ),
    }


def run_complexity() -> dict[str, Any]:
    """复杂度 hook 双口径（均须 exit=0、新增硬超限 0）。

    判定只认**终端原文**：``[hard]`` 行中不带“（存量基线）”的即新增硬超限；
    不允许“解析失败 ⇒ 默认空 ⇒ 假 pass”（退出码之外的第二道判据）。
    """
    suites: dict[str, Any] = {}
    for name, paths in (
        ("src_only", ["src/lazybull"]),
        ("with_scripts", ["src/lazybull", "scripts"]),
    ):
        command = [sys.executable, "scripts/check/check_complexity.py", "--baseline", str(BASELINE)]
        for path in paths:
            command.extend(["--paths", path])
        result = _run(command)
        hard_lines = [
            line for line in result["stdout"].splitlines() if line.strip().startswith("[hard]")
        ]
        # 基线内条目带“（存量基线）”或“（已豁免）”；新增硬超限两者都不带
        new_hard = [
            line for line in hard_lines if "存量基线" not in line and "已豁免" not in line
        ]
        summary = [
            line for line in result["stdout"].splitlines() if "新增硬超限" in line
        ][-1:]
        declared = None
        if summary:
            match = re.search(r"新增硬超限\s*(\d+)\s*条", summary[0])
            declared = int(match.group(1)) if match else None
        suites[name] = {
            "command": command,
            "exit_code": result["exit_code"],
            "hard_total": len(hard_lines),
            "new_hard": new_hard[:10],
            "new_hard_count": len(new_hard),
            "declared_new_hard": declared,
            "summary": summary,
            "parsed_clean": declared is not None,
            "passed": (
                result["exit_code"] == 0
                and not new_hard
                and declared == 0
                and len(new_hard) == declared
            ),
        }
    return {
        "suites": suites,
        "passed": all(item["passed"] for item in suites.values()),
    }


def corpus_determinism(work_root: Path) -> dict[str, Any]:
    """tiny 语料双跑：``content_sha256`` 逐字节一致（构建器确定性）。"""
    digests = []
    for suffix in ("a", "b"):
        target = work_root / f"determinism_{suffix}"
        target.mkdir(parents=True, exist_ok=True)
        payload = build_t4_corpus.build(target, "tiny")
        digests.append(payload["content_sha256"])
    return {
        "digests": digests,
        "equal": digests[0] == digests[1],
        "passed": digests[0] == digests[1],
    }


def _profile_roots(work_root: Path, profile: str) -> dict[str, Path]:
    """构建上一快照语料（前 N-1 个 runs）与报告根。"""
    plan = PROFILE_PLAN[profile]
    corpus = work_root / f"corpus_{profile}"
    corpus.mkdir(parents=True, exist_ok=True)
    build_t4_corpus.build(corpus, profile, upto=plan["runs_previous"])
    return {
        "corpus": corpus,
        "reports": work_root / f"reports_previous_{profile}",
        # 计时轮的落盘旁证写独立根：不能污染上一快照根（否则上一快照不再唯一）
        "reports_timed": work_root / f"reports_timed_{profile}",
    }


def publish_previous(roots: dict[str, Path], profile: str) -> dict[str, Any]:
    """上一快照全量发布（**批任务**；耗时只登记，不冒充秒级交互）。"""
    plan = PROFILE_PLAN[profile]
    command = [
        sys.executable,
        str(REPORT_SCRIPT),
        "generate",
        "--kind",
        "trend",
        "--evidence-class",
        "research",
        "--source",
        f"runs:{_rel_text(roots['corpus'] / 'runs')}",
        "--source",
        f"legacy_wf:{_rel_text(roots['corpus'] / 'legacy')}",
        "--reports-root",
        _rel_text(roots["reports"]),
        "--attempt-id",
        f"t4_prev_{profile}",
        "--generated-at",
        WORKFLOW_GENERATED_AT,
        "--json",
    ]
    started = time.perf_counter()
    result = _run(command)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    payload = _last_json(result["stdout"]) if result["exit_code"] == 0 else {}
    run_dir = roots["reports"] / "research" / "trend" / str(payload.get("report_id"))
    return {
        "command": _posix_command(_relativize(command)),
        "command_args": _posix_command(_relativize(command[1:])),
        "interpreter": _interpreter_record(),
        "exit_code": result["exit_code"],
        "elapsed_ms": elapsed_ms,
        "batch_gate": "diagnostic_only（全历史重建是单独授权批任务，不冒充秒级交互）",
        "report_id": payload.get("report_id"),
        "run_dir": str(run_dir) if run_dir.is_dir() else None,
        "expected_total": plan["total"] - 1,
        "stderr_tail": result["stderr_tail"],
        "passed": result["exit_code"] == 0 and run_dir.is_dir(),
    }


def _interpreter_record() -> dict[str, Any]:
    """解释器登记（basename + 版本）：命令参数逐字入库，解释器不写绝对路径。"""
    return {
        "basename": Path(sys.executable).name,
        "version": platform.python_version(),
        "note": "实际执行解释器；命令/参数以仓库相对形态登记，异机可由 basename 重建",
    }


def _last_json(text: str) -> dict[str, Any]:
    """取子进程输出的最后一行 JSON（``--json`` 单行协议）。"""
    for line in reversed([item for item in text.splitlines() if item.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise AcceptanceSetupError("子进程未输出 JSON 载荷")


def _new_run_dir(corpus: Path, profile: str) -> Path:
    plan = PROFILE_PLAN[profile]
    return corpus / "runs" / f"p6t4_run_{plan['runs_previous']:04d}"


def _incremental_args(
    previous: Path, new_run: Path, attempt: str, *, generated_at: str | None, publish: bool,
    audit_out: Path | None, reports_root: Path | None,
) -> list[str]:
    """增量生成 CLI 参数（单新 run + 上一快照）。"""
    args = [
        sys.executable,
        str(REPORT_SCRIPT),
        "generate",
        "--kind",
        "trend",
        "--evidence-class",
        "research",
        "--source",
        f"runs:{_rel_text(new_run)}",
        "--previous",
        _rel_text(previous),
        "--attempt-id",
        attempt,
        "--json",
    ]
    if generated_at is not None:
        args += ["--generated-at", generated_at]
    if reports_root is not None:
        args += ["--reports-root", _rel_text(reports_root)]
    if audit_out is not None:
        args += ["--audit-out", str(audit_out)]
    if not publish:
        args.append("--no-publish")
    return args


def _payload_paths(payload: Mapping[str, Any]) -> dict[str, Any]:
    """载荷绝对路径巡检（B2-01/R3B-01）：除 ``environment`` 外不得出现盘符/URI/仓库绝对前缀。

    盘符用**无锚** ``re.search``（覆盖 ``runs:D:\\…`` 这类 ``kind:abs`` 形态）；
    仅要求 ``X:`` 后接反斜杠/正斜杠，避免误伤 ``12:30`` 之类文本。
    """
    leaks: list[str] = []

    def walk(value: Any, where: str) -> None:
        if isinstance(value, str):
            text = value
            if (
                re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text)
                or "file:///" in text
                or str(ROOT) in text
                or str(ROOT).replace("\\", "/") in text
            ):
                leaks.append(f"{where}: {text[:80]}")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key) == "environment" and where in ("", "checks"):
                    continue
                walk(item, f"{where}.{key}" if where else str(key))
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{where}[{index}]")

    walk(payload, "")
    return {"leaks": leaks[:20], "count": len(leaks), "passed": not leaks}


def _run_load_check(new_run: Path, profile: str) -> dict[str, Any]:
    """计时新 run 的负载断言（R3-07）：行数/折数/成交条数必须等于声明档位。"""
    import pyarrow.parquet as pq

    expected = dict(build_t4_corpus.PROFILE_LOAD[profile])
    chain_rows = pq.ParquetFile(new_run / "chain_nav.parquet").metadata.num_rows
    summary_rows = pq.ParquetFile(new_run / "summary.parquet").metadata.num_rows
    folds = sorted(path for path in (new_run / "folds").iterdir() if path.is_dir())
    trades = sum(pq.ParquetFile(fold / "trades.parquet").metadata.num_rows for fold in folds)
    actual = {
        "chain_rows": chain_rows,
        "summary_rows": summary_rows,
        "folds": len(folds),
        "trades": trades,
    }
    loader_ok = True
    loader_error = None
    try:
        from src.lazybull.v2.evidence.runs_loader import load_runs_batch

        load_runs_batch(new_run)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        loader_ok = False
        loader_error = f"{type(exc).__name__}: {exc}"
    return {
        "expected": expected,
        "actual": actual,
        "loader_ok": loader_ok,
        "loader_error": loader_error,
        "passed": (
            chain_rows == expected["chain_rows"]
            and summary_rows == expected["folds"]
            and len(folds) == expected["folds"]
            and trades == expected["trades"]
            and loader_ok
        ),
    }


def _new_run_metrics_check(landed_dir: Path | None, new_run: Path) -> dict[str, Any]:
    """计时新 run 的指标有效性（R3-07-R1）：核心四项必须 available（TopK 按契约不可用）。"""
    if landed_dir is None or not landed_dir.is_dir():
        return {"passed": False, "error": "无落盘产物"}
    index = trend_csv.parse_run_index(
        (landed_dir / "run_index.csv").read_text(encoding="utf-8")
    )
    trend = trend_csv.parse_trend((landed_dir / "trend.csv").read_text(encoding="utf-8"))
    matched = [row for row in index if str(row.get("run_id")) == new_run.name]
    if len(matched) != 1:
        return {"passed": False, "error": f"新 run 索引行数={len(matched)}"}
    run_key = str(matched[0]["run_key"])
    rows = {str(row["metric_name"]): row for row in trend if str(row["run_key"]) == run_key}
    core = ("annualized_return", "max_drawdown", "turnover_amount", "cost_amount")
    statuses = {name: str((rows.get(name) or {}).get("report_status")) for name in core}
    topk = rows.get("topk_positive_day_ratio") or {}
    return {
        "index_status": matched[0].get("report_status"),
        "core_statuses": statuses,
        "topk_reason": topk.get("degradation_reason_code"),
        "passed": (
            matched[0].get("report_status") == "available"
            and all(word == "available" for word in statuses.values())
            and topk.get("degradation_reason_code") == "no_unique_implementation"
        ),
    }


def timed_incremental(roots: dict[str, Path], previous: Path, profile: str) -> dict[str, Any]:
    """完整进程 5 次增量生成计时（父进程墙钟；逐次 <10,000ms 且五件字节一致）。"""
    new_run = _new_run_dir(roots["corpus"], profile)
    load_check = _run_load_check(new_run, profile)
    records: list[dict[str, Any]] = []
    for index in range(perf_gate.GENERATION_REPEATS):
        attempt = f"t4_inc_{profile}_{index}"
        started = time.perf_counter()
        result = _run(
            _incremental_args(
                previous,
                new_run,
                attempt,
                generated_at=WORKFLOW_GENERATED_AT,
                publish=False,
                audit_out=None,
                reports_root=roots["reports"],
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload = _last_json(result["stdout"]) if result["exit_code"] == 0 else {}
        records.append(
            {
                "index": index,
                "elapsed_ms": elapsed_ms,
                "exit_code": result["exit_code"],
                "verdict": payload.get("verdict"),
                "report_id": payload.get("report_id"),
                "outputs": payload.get("outputs"),
                "manifest_sha256": payload.get("manifest_sha256"),
                "file_bytes": payload.get("file_bytes"),
                "stderr_tail": result["stderr_tail"],
            }
        )
    series = perf_gate.judge_generation_series([item["elapsed_ms"] for item in records])
    identities = {
        json.dumps(
            {"outputs": item["outputs"], "manifest": item["manifest_sha256"]}, sort_keys=True
        )
        for item in records
    }
    succeeded = all(item["exit_code"] == 0 and item["verdict"] == "pass" for item in records)
    # 字节一致的**落盘**旁证：再发布一次并把五件实际字节与第 5 次自报哈希对齐
    # （B-06：只信子进程 stdout 的话，证据无法独立复核）。
    roots["reports_timed"].mkdir(parents=True, exist_ok=True)
    landed = _run(
        _incremental_args(
            previous,
            new_run,
            f"t4_inc_landed_{profile}",
            generated_at=WORKFLOW_GENERATED_AT,
            publish=True,
            audit_out=None,
            reports_root=roots["reports_timed"],
        )
    )
    landed_payload = _last_json(landed["stdout"]) if landed["exit_code"] == 0 else {}
    landed_dir = (
        Path(str(landed_payload.get("report_path"))) if landed_payload.get("report_path") else None
    )
    landed_hashes = _dir_hashes(landed_dir) if landed_dir and landed_dir.is_dir() else {}
    declared = dict(records[-1].get("outputs") or {})
    if records[-1].get("manifest_sha256"):
        declared["report_manifest.json"] = records[-1]["manifest_sha256"]
    landed_matches = bool(landed_hashes) and landed_hashes == declared
    metrics_check = _new_run_metrics_check(landed_dir, new_run)
    return {
        "profile": profile,
        "records": records,
        "gate": series,
        "gate_mode": "incremental_full_process（generate 阶段；不含 publish/审计，见 publish_scope）",
        "publish_scope": "10s 门只覆盖 generate；真实发布含两次整树审计，单独登记 elapsed_ms",
        "new_run": str(new_run),
        "load_check": load_check,
        "metrics_check": metrics_check,
        "byte_identical": len(identities) == 1,
        "landed": {
            "dir": str(landed_dir) if landed_dir else None,
            "exit_code": landed["exit_code"],
            "manifest_sha256": landed_hashes.get("report_manifest.json"),
            "matches_declared": landed_matches,
        },
        "all_succeeded": succeeded and landed["exit_code"] == 0,
        "passed": bool(
            series["passed"]
            and succeeded
            and len(identities) == 1
            and landed_matches
            and load_check["passed"]
            and metrics_check["passed"]
        ),
    }


def _dir_hashes(directory: Path | None) -> dict[str, str]:
    """目录内单层文件的 SHA-256（证据侧独立回读，不信子进程自报）。"""
    if directory is None or not directory.is_dir():
        return {}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _tree_digest(roots: Sequence[Path]) -> dict[str, Any]:
    """多根树的文件数/字节数/聚合摘要（输入侧 before/after 比对用）。"""
    files = 0
    size = 0
    entries: list[dict[str, str]] = []
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        prefix = f"{base.name}/"
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            files += 1
            size += path.stat().st_size
            entries.append(
                {
                    "path": prefix + path.relative_to(base).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return {
        "roots": [str(item) for item in roots],
        "files": files,
        "bytes": size,
        "digest": hashlib.sha256(
            json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def publish_with_audit(previous: Path, profile: str, validation_id: str) -> dict[str, Any]:
    """真实发布一次增量 run 到 canonical 报告根并落精确写审计。"""
    new_run = _new_run_dir(
        ROOT / "temp" / f"p6_t4_acceptance_{validation_id}" / f"corpus_{profile}", profile
    )
    audit_out = (
        ROOT
        / contracts.REPORTS_ROOT
        / "validations"
        / "research"
        / f"{validation_id}_audit_{profile}"
    )
    if audit_out.exists():
        raise AcceptanceSetupError(f"审计目录已存在，拒绝覆盖：{audit_out}")
    # 输入侧字节证据（B-02）：仓库内输入根在审计 EXCLUDES（temp/、data/walk_forward、
    # data/paper_shadow）内，产品审计不覆盖 ⇒ 由驱动在发布前后各做一次 receipts 全树哈希。
    input_before = _tree_digest([previous, new_run])
    command = _incremental_args(
        previous,
        new_run,
        f"t4_publish_{profile}",
        generated_at=None,
        publish=True,
        audit_out=audit_out,
        reports_root=None,
    )
    started = time.perf_counter()
    result = _run(command)
    audited_ms = (time.perf_counter() - started) * 1000.0
    input_after = _tree_digest([previous, new_run])
    payload = _last_json(result["stdout"]) if result["exit_code"] == 0 else {}
    audit: dict[str, Any] = {}
    audit_file = audit_out / "readonly_audit.json"
    if audit_file.is_file():
        raw = json.loads(audit_file.read_text(encoding="utf-8"))
        scope = dict(raw.get("scan_scope") or {})
        audit = {
            "verdict": raw.get("verdict"),
            "uncovered": list(scope.get("uncovered") or []),
            "reparse_checked": scope.get("reparse_checked"),
            "reparse_points": list(scope.get("reparse_points") or []),
            "missing_expected": list(raw.get("missing_expected") or []),
            "unexpected_inside": list(raw.get("unexpected_inside") or []),
            "outside_whitelist": list(raw.get("outside_whitelist") or []),
            "foreign_root_paths": list(raw.get("foreign_root_paths") or []),
            "overwrote_existing": list(raw.get("overwrote_existing") or []),
            "path": audit_out.relative_to(ROOT).as_posix() + "/readonly_audit.json",
        }
    run_dir = Path(payload.get("report_path")) if payload.get("report_path") else None
    return {
        "profile": profile,
        "command": _posix_command(_relativize(command)),
        "command_args": _posix_command(_relativize(command[1:])),
        "interpreter": _interpreter_record(),
        "exit_code": result["exit_code"],
        "verdict": payload.get("verdict"),
        "elapsed_ms": audited_ms,
        "report_id": payload.get("report_id"),
        "generated_at": payload.get("generated_at"),
        "report_path": str(run_dir) if run_dir else None,
        "audit": audit,
        "audit_included": True,
        "publish_scope": (
            "两次 worktree 有界子树哈希（输入根若在 EXCLUDES 内另由驱动前后 digest 覆盖），"
            "不吃 10s 生成门"
        ),
        "input_evidence": {
            "files": input_before["files"],
            "bytes": input_before["bytes"],
            "digest_before": input_before["digest"],
            "digest_after": input_after["digest"],
            "unchanged": input_before == input_after,
        },
        "stderr_tail": result["stderr_tail"],
        "passed": (
            result["exit_code"] == 0
            and payload.get("verdict") == "pass"
            and run_dir is not None
            and run_dir.is_dir()
            and audit.get("verdict") == "pass"
            and not audit.get("uncovered")
            and audit.get("reparse_checked") is True
            and input_before == input_after
        ),
    }


def csv_conservation(run_dir: Path) -> dict[str, Any]:
    """两份 CSV 守恒：行数 = run 数 / 5×run 数；键唯一；排序；``index_mode=full``。"""
    projection = json.loads((run_dir / "projection.json").read_text(encoding="utf-8"))
    index_text = (run_dir / "run_index.csv").read_text(encoding="utf-8")
    trend_text = (run_dir / "trend.csv").read_text(encoding="utf-8")
    index_rows = trend_csv.parse_run_index(index_text)
    trend_rows = trend_csv.parse_trend(trend_text)
    counts = dict(projection.get("counts") or {})
    conservation_ok = True
    try:
        trend_csv.assert_key_conservation(index_rows, trend_rows)
    except Exception:  # noqa: BLE001 - 守恒失败要在载荷里可见，而不是抛裸异常
        conservation_ok = False
    sorted_ok = trend_csv.sort_index_rows(index_rows) == index_rows
    visible = dict(projection.get("visible") or {})
    visible_ids = [str(row.get("run_key")) for row in projection.get("runs") or []]
    expected_visible = [row["run_key"] for row in index_rows][-trend_contract.VISIBLE_ROWS_MAX:]
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    size = perf_gate.judge_size(len(html.encode("utf-8")))
    # 页面渲染证据（与 CSV 守恒证据并列）：折线条数 + 合法空图 + 状态行文案
    # 空图与断线都会改变 <polyline> 条数（一条曲线可被缺值切成多段），
    # 因此图表数量以容器 ``chart-box`` 为准，不假设“一条曲线 = 一个 polyline”。
    render = {
        "polyline": html.count("<polyline"),
        "chart_empty": html.count('aria-label="chart-empty"'),
        "chart_boxes": html.count('class="chart-box"'),
        "state_cell": html.count("不可用（见状态列）"),
        "charts_ok": html.count('class="chart-box"') == len(trend_contract.METRICS)
        and html.count("<polyline") >= 1,
    }
    return {
        "run_rows": len(index_rows),
        "trend_rows": len(trend_rows),
        "counts_run_total": counts.get("run_total"),
        "projection_rows": len(projection.get("runs") or []),
        "projection_metric_rows": counts.get("metric_row_total"),
        "expected_metric_rows": counts.get("expected_metric_row_total"),
        "index_mode_all_full": all(row["index_mode"] == "full" for row in index_rows),
        "sorted": sorted_ok,
        "key_conservation": conservation_ok,
        "visible_rows": visible.get("visible_rows"),
        "visible_matches_projection": visible_ids == expected_visible,
        "render_evidence": render,
        "size": size,
        "passed": (
            conservation_ok
            and sorted_ok
            and len(index_rows) == counts.get("run_total")
            and len(trend_rows) == counts.get("metric_row_total")
            == counts.get("expected_metric_row_total")
            == len(index_rows) * len(trend_contract.METRICS)
            and all(row["index_mode"] == "full" for row in index_rows)
            and visible.get("visible_rows") == min(
                trend_contract.VISIBLE_ROWS_MAX, len(index_rows)
            )
            and visible_ids == expected_visible
            and render["charts_ok"]
            and render["polyline"] >= 1
            and size["passed"]
        ),
    }


def _verify_inputs_on_disk(
    record: Mapping[str, Any], previous_dir: Path, new_run_dir: Path
) -> dict[str, Any]:
    """按 manifest 的 ``inputs`` 回到**实际来源根**逐文件 re-hash（B-01）。

    ``recompute_chain`` 只比较 manifest 内部两个字段（seal 已覆盖），输入环恒真；
    这里独立解析 ``root_id → 目录`` 并回读字节，缺失/不等即 ``source_changed``。

    边界（T4-R2-B2-04）：增量轮只有一个新来源根；多 ``--source`` 的 manifest 在这里
    fail-closed（``error``），T5 多源 smoke 需按 root_map 逐根建映射后再复用本函数。
    """
    inputs = [dict(item) for item in record.get("inputs") or []]
    roots = {str(item.get("path") or "").split(":", 1)[0] for item in inputs}
    non_prev = {root for root in roots if root and root != "prev"}
    if len(non_prev) != 1:
        return {
            "verified": 0,
            "missing": [],
            "mismatch": [],
            "error": f"输入根不可唯一解析（多源 manifest 归 T5）：{sorted(roots)}",
            "passed": False,
        }
    locations = {"prev": Path(previous_dir), next(iter(non_prev)): Path(new_run_dir)}
    missing: list[str] = []
    mismatch: list[str] = []
    verified = 0
    for item in inputs:
        material = str(item.get("path") or "")
        root_id, _, relative = material.partition(":")
        declared = str(item.get("sha256") or "")
        base = locations.get(root_id)
        target = (base / relative) if base is not None and relative else None
        if target is None or not target.is_file():
            missing.append(material)
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != declared:
            mismatch.append(material)
            continue
        verified += 1
    return {
        "verified": verified,
        "total": len(inputs),
        "missing": missing[:10],
        "mismatch": mismatch[:10],
        "error": None,
        "passed": not missing and not mismatch and verified == len(inputs) and bool(inputs),
    }


def recompute_published(
    run_dir: Path, previous_dir: Path, new_run_dir: Path
) -> dict[str, Any]:
    """已发布 run 的 manifest 自封存 + 链复算 + **磁盘输入**回读（逐环 7 环）。"""
    record = json.loads((run_dir / "report_manifest.json").read_text(encoding="utf-8"))
    projection = json.loads((run_dir / "projection.json").read_text(encoding="utf-8"))
    outputs = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "report_manifest.json"
    }
    seal = manifest.verify_manifest(record)
    chain = manifest.recompute_chain(
        manifest=record, projection=projection, outputs=outputs, inputs=record["inputs"]
    )
    links = [item for item in chain.get("checks") or [] if item.get("ok")]
    inputs = _verify_inputs_on_disk(record, previous_dir, new_run_dir)
    # 期望 7 环：input_manifest / projection / outputs.keys / 4 个产出（不含 manifest 本体）
    expected_links = 3 + len(publish.TREND_FILENAMES) - 1
    return {
        "seal_status": seal.get("report_status"),
        "seal_reason": seal.get("degradation_reason_code"),
        "chain_status": chain.get("report_status"),
        "chain_reason": chain.get("degradation_reason_code"),
        "links_ok": len(links),
        "expected_links": expected_links,
        "outputs": len(outputs),
        "expected_outputs": len(publish.TREND_FILENAMES) - 1,
        "input_manifest_sha256": record.get("input_manifest_sha256"),
        "inputs": inputs,
        "root_map": dict(projection.get("root_map") or {}),
        "inherited_roots": list(projection.get("inherited_roots") or []),
        "passed": (
            seal.get("report_status") == "available"
            and chain.get("report_status") == "available"
            and len(links) == expected_links
            and len(outputs) == len(publish.TREND_FILENAMES) - 1
            and bool(inputs.get("passed"))
        ),
    }


def open_gate(run_dir: Path, validation_id: str, profile: str) -> dict[str, Any]:
    """5 冷 + 5 热打开测量（真实 Chromium；输出落 validation 兄弟目录）。"""
    out = (
        ROOT
        / contracts.REPORTS_ROOT
        / "validations"
        / "research"
        / f"{validation_id}_open_{profile}"
    )
    if out.exists():
        raise AcceptanceSetupError(f"打开测量目录已存在，拒绝覆盖：{out}")
    try:
        performance = measure_report.measure(run_dir / "report.html", out, repeat=5)
    except (measure_report.MeasureSetupError, OSError, RuntimeError) as exc:
        return {"profile": profile, "passed": False, "detail": f"{type(exc).__name__}: {exc}"}
    gates = dict(performance.get("gates") or {})
    opened = dict(gates.get("open") or {})
    return {
        "profile": profile,
        "out": out.relative_to(ROOT).as_posix(),
        "verdict": performance.get("verdict"),
        "report_bytes": (performance.get("html") or {}).get("bytes"),
        "size_gate": gates.get("size"),
        "open_max_ms": opened.get("max_ms"),
        "cold_count": opened.get("cold_count"),
        "hot_count": opened.get("hot_count"),
        "threshold_ms": perf_gate.MAX_OPEN_MS,
        "note": (
            "浏览器测量产物（performance.json 与 profile）含 file:// 导航 URL 与绝对 profile 路径，"
            "属验收侧测量证据，不进 canonical 报告、不参与异机复算"
        ),
        "passed": performance.get("verdict") == "pass",
    }


def linux_package(
    run_dir: Path,
    validation_id: str,
    profile: str,
    generated_at: str | None = None,
    previous_run_dir: Path | None = None,
) -> dict[str, Any]:
    """生成 canonical Linux 验收包（固定四文件 + 期望哈希 + 复跑命令）。"""
    out = (
        ROOT
        / contracts.REPORTS_ROOT
        / "validations"
        / "research"
        / f"{validation_id}_linux_{profile}"
    )
    if out.exists():
        raise AcceptanceSetupError(f"Linux 包目录已存在，拒绝覆盖：{out}")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }
    corpus = ROOT / "temp" / f"p6_t4_acceptance_{validation_id}" / f"corpus_{profile}"
    stamp = generated_at or WORKFLOW_GENERATED_AT
    if previous_run_dir is None:
        raise AcceptanceSetupError("Linux 包必须给出上一快照 run 目录（用于可执行复跑命令）")
    corpus_rel = corpus.relative_to(ROOT).as_posix()
    new_run_rel = _new_run_dir(corpus, profile).relative_to(ROOT).as_posix()
    previous_rel = Path(previous_run_dir).relative_to(ROOT).as_posix()
    previous_files = _dir_hashes(Path(previous_run_dir))
    # 复跑命令必须**真能执行**（B-04）：按 upto 分两步建语料 → 上一快照全量 → 增量（显式 reports-root）
    prev_reports_rel = previous_rel.rsplit("/research/trend/", 1)[0]
    commands = [
        "cd <repo_root>",
        f"python scripts/v2_p6/build_t4_corpus.py --root {corpus_rel} --profile {profile} "
        f"--upto {PROFILE_PLAN[profile]['runs_previous']}",
        "python scripts/p6_report.py generate --kind trend --evidence-class research "
        f"--source runs:{corpus_rel}/runs --source legacy_wf:{corpus_rel}/legacy "
        f"--generated-at {WORKFLOW_GENERATED_AT} --attempt-id prev "
        f"--reports-root {prev_reports_rel}",
        f"python scripts/v2_p6/build_t4_corpus.py --root {corpus_rel} --profile {profile}",
        "python scripts/p6_report.py generate --kind trend --evidence-class research "
        f"--source runs:{new_run_rel} --previous {previous_rel} "
        f"--generated-at {stamp} --attempt-id <attempt> --reports-root <out>",
    ]
    package = trend_portability.build_linux_package(
        report_id=run_dir.name,
        generated_at=stamp,
        evidence_class="research",
        files=hashes,
        commands=commands,
        corpus_sha256=build_t4_corpus.content_digest(corpus) if corpus.is_dir() else None,
        steps=[
            "1) 按 upto=上一快照 run 数构建确定性语料（标准/压力档见 corpus.content_sha256）",
            "2) 全量发布上一快照（批任务；--generated-at 固定，--reports-root 显式）",
            "3) 补齐语料到全量（同根增量物化，确定性不变）",
            "4) 单新 run 增量发布（--previous 指第 2 步产物，--reports-root <out>）",
            "5) sh acceptance.sh <report_dir> 逐文件比对预期哈希（包内不复算生成）",
        ],
        previous={
            "report_id": previous_rel.rsplit("/", 1)[-1],
            "run_dir": previous_rel,
            "manifest_sha256": previous_files.get("report_manifest.json"),
            "files": previous_files,
        },
    )
    out.mkdir(parents=True)
    for name, payload in package.items():
        (out / name).write_bytes(payload)
    manifest_payload = json.loads(package["package_manifest.json"].decode("utf-8"))
    recorded = {row["path"]: row["sha256"] for row in manifest_payload["files"]}
    actual = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in package.items()
        if name != "package_manifest.json"
    }
    return {
        "profile": profile,
        "out": out.relative_to(ROOT).as_posix(),
        "files": sorted(package),
        "file_count": len(package),
        "package_record_sha256": manifest_payload["record_sha256"],
        "self_check_ok": all(recorded.get(name) == digest for name, digest in actual.items()),
        "executed_on_linux": False,
        "note": "本轮只产出 canonical 包；真实 Linux 执行门归 T5",
        "passed": (
            tuple(sorted(package)) == tuple(sorted(trend_portability.LINUX_PACKAGE_FILES))
            and all(recorded.get(name) == digest for name, digest in actual.items())
            and all("D:" not in (out / name).read_text(encoding="utf-8") for name in
                    ("expected_hashes.json", "package_manifest.json"))
        ),
    }


def _profile_evidence(validation_id: str, profile: str, roots: dict[str, Path],
                      previous: dict[str, Any], timed: dict[str, Any],
                      published: dict[str, Any]) -> dict[str, Any]:
    """单档证据汇总（发布/守恒/复算/打开/包）。"""
    run_dir = Path(str(published.get("report_path"))) if published.get("report_path") else None
    new_run = _new_run_dir(roots["corpus"], profile)
    previous_dir = Path(str(previous.get("run_dir"))) if previous.get("run_dir") else None
    conservation = csv_conservation(run_dir) if run_dir and run_dir.is_dir() else {"passed": False}
    recompute = (
        recompute_published(run_dir, previous_dir, new_run)
        if run_dir and run_dir.is_dir() and previous_dir is not None
        else {"passed": False}
    )
    opened = open_gate(run_dir, validation_id, profile) if run_dir else {"passed": False}
    package = (
        linux_package(
            run_dir,
            validation_id,
            profile,
            published.get("generated_at"),
            previous_run_dir=previous_dir,
        )
        if run_dir
        else {"passed": False}
    )
    return {
        "profile": profile,
        "corpus": {
            "build_previous_runs": PROFILE_PLAN[profile]["runs_previous"],
            "build_total_runs": PROFILE_PLAN[profile]["total"],
            "content_sha256": build_t4_corpus.content_digest(roots["corpus"]),
        },
        "previous": previous,
        "timed": timed,
        "published": published,
        "conservation": conservation,
        "recompute": recompute,
        "open": opened,
        "linux_package": package,
        "passed": all(
            item.get("passed")
            for item in (previous, timed, published, conservation, recompute, opened, package)
        ),
    }


def _collect(validation_id: str) -> dict[str, Any]:
    """采集全部检查（顺序执行；任何一步失败都留在载荷里，不中断后续取证）。"""
    work_root = ROOT / "temp" / f"p6_t4_acceptance_{validation_id}"
    work_root.mkdir(parents=True, exist_ok=True)
    profiles: dict[str, Any] = {}
    for profile in ("standard", "stress"):
        roots = _profile_roots(work_root, profile)
        previous = publish_previous(roots, profile)
        if not previous["passed"]:
            profiles[profile] = {"profile": profile, "passed": False, "previous": previous}
            continue
        build_t4_corpus.build(roots["corpus"], profile, upto=PROFILE_PLAN[profile]["runs"])
        timed = timed_incremental(roots, Path(str(previous["run_dir"])), profile)
        published = publish_with_audit(Path(str(previous["run_dir"])), profile, validation_id)
        profiles[profile] = _profile_evidence(
            validation_id, profile, roots, previous, timed, published
        )
    return profiles


def _sanitize(value: Any) -> Any:
    """载荷归一：Path → POSIX 文本、非有限浮点置 None（JSON 严格）。"""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def _posix_command(command: Sequence[str]) -> list[str]:
    """命令记录归一为 POSIX 分隔符（R3A-04/R3B-03）：执行仍用原列表，记录可跨平台复制。"""
    return [str(item).replace("\\", "/") for item in command]


def _rel_text(path: Path | str) -> str:
    """仓库内路径 → 相对 POSIX（命令参数用）；仓库外保持绝对（显式边界）。"""
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _relativize(value: Any) -> Any:
    """证据载荷路径归一（B-07/B2-01）：绝对路径（含 ``kind:path``）→ 相对 POSIX。

    机器/用户名信息只保留在 ``environment`` 段；命令与产物路径异机可复用。
    解释器单独登记为 basename（``sys.executable`` → ``python``），不再混入绝对路径。
    """
    if isinstance(value, str):
        if value == sys.executable:
            return "python"
        if ":" in value:
            # ``runs:<abs>`` / ``legacy_wf:<abs>`` / ``A:`` 盘符 → 只归一冒号后的路径段
            head, _, tail = value.partition(":")
            if tail:
                try:
                    candidate = Path(tail)
                except (OSError, ValueError):
                    candidate = None
                if candidate is not None and candidate.is_absolute():
                    return f"{head}:{_rel_text(candidate)}"
        try:
            path = Path(value)
        except (OSError, ValueError):
            return value
        if path.is_absolute():
            return _rel_text(path)
        return value
    if isinstance(value, dict):
        return {str(key): _relativize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_relativize(item) for item in value]
    return value


def _guarded(stage: str, action: Any) -> dict[str, Any]:
    """执行单个检查：异常收敛为 fail 记录（现场保留优先于裸 traceback）。"""
    try:
        return action()
    except BaseException as exc:  # noqa: BLE001 - 登记异常类名/信息/调用栈尾，仍写汇总载荷
        return {
            "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-3000:],
            "passed": False,
        }


def run(validation_id: str | None = None) -> dict[str, Any]:
    """执行一次完整 T4 验收并写 ``t4_acceptance.json``（拒绝覆盖既有目录）。"""
    name = validation_id or _default_validation_id()
    out = _assert_new_validation_out(
        ROOT / contracts.REPORTS_ROOT / "validations" / "research" / name
    )
    started = time.perf_counter()
    checks: dict[str, Any] = {}
    checks["pytest"] = _guarded("pytest", run_tests)
    checks["complexity"] = _guarded("complexity", run_complexity)
    checks["corpus_determinism"] = _guarded(
        "corpus_determinism",
        lambda: corpus_determinism(ROOT / "temp" / f"p6_t4_acceptance_{name}"),
    )
    checks["profiles"] = _guarded("collect", lambda: _collect(name))
    environment = {
        "captured": env_probe.capture_environment(),
        "note": "环境/依赖版本仅登记，不进 canonical 报告（不参与 verdict）",
    }
    profiles = checks.get("profiles") or {}
    profiles_ok = (
        isinstance(profiles, dict)
        and {"standard", "stress"} <= set(profiles)
        and all(isinstance(item, dict) and bool(item.get("passed")) for item in profiles.values())
    )
    verdict_ok = (
        bool(checks["pytest"].get("passed"))
        and bool(checks["complexity"].get("passed"))
        and bool(checks["corpus_determinism"].get("passed"))
        and profiles_ok
    )
    payload = _relativize(
        _sanitize(
            {
                "schema": "p6_t4_acceptance/v1",
                "validation_id": name,
                "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "checks": checks,
                "environment": environment,
                "thresholds": dict(contracts.THRESHOLDS),
                "command_cwd": "<repo_root>",
                "path_style": "仓库相对 POSIX（命令记录已归一正斜杠）",
                "verdict": "pass" if verdict_ok else "fail",
            }
        )
    )
    payload["checks"]["payload_paths"] = _payload_paths(payload)
    if not payload["checks"]["payload_paths"]["passed"]:
        payload["verdict"] = "fail"
    out.mkdir(parents=True)
    (out / "t4_acceptance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口（``--validation-id`` 缺省按 UTC 时间生成新 id）。"""
    parser = argparse.ArgumentParser(description="P6 T4 验收驱动（P6-M4）")
    parser.add_argument("--validation-id", default=None)
    args = parser.parse_args(argv)
    try:
        payload = run(args.validation_id)
    except AcceptanceSetupError as exc:
        print(f"[P6-T4] 设置错误：{exc}", file=sys.stderr)
        return 2
    print(json.dumps({"verdict": payload["verdict"],
                      "validation_id": payload["validation_id"]}, ensure_ascii=False))
    return 0 if payload["verdict"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())

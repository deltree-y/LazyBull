# -*- coding: utf-8 -*-
"""P6 T3 验收驱动（P6-M3 可复跑证据包）。

一次运行产出 ``data/reports/p6/validations/research/<validation_id>/t3_acceptance.json``：

1. T3 十一组 pytest + T1 七组 + T2 七组 + T0.5 四组 + 治理面（前序阶段不回归）；
2. 复杂度 hook 双口径（``src/lazybull`` 与 ``src/lazybull scripts``）；
3. 语料双跑一致性（标准/压力语料 ``content_sha256`` 逐字节一致）；
4. 标准/压力两档：**完整进程** 5 次生成计时 + 真实发布 + 精确写审计
   （``audit/readonly_audit_*.json``）+ 五份 CSV 守恒（``evidence/csv_conservation_*.json``）
   + 已发布 run 的链复算（``recompute_chain``）+ 5 冷 5 热打开测量；
5. Δ 方向与稳定性正例的成品侧复核（读已发布 ``arm_metrics.csv`` / ``stability.csv``）；
6. bootstrap 拦截与“无裁决字段”检查（生成链只读投影，不含 verdict/decision）；
7. 汇总 verdict（任一检查失败即 fail，原始终端输出保留在载荷中）。

写域：本驱动只写本次新 validation 目录、其兄弟 ``_audit_*``/``_open_*`` 目录与 ``temp/`` 语料；
不写生产数据根。语料与计时工作目录都在 ``temp/``（已 gitignore）。
"""

# 复杂度豁免：阶段专用验收驱动（scripts/v2_p6，不入生产链路，见 CLAUDE.md §7.5）

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）
    sys.dont_write_bytecode = True
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p6 import build_t3_corpus, env_probe, measure_report  # noqa: E402
from src.lazybull.v2.reporting import (  # noqa: E402
    contracts,
    experiment_csv,
    manifest,
    perf_gate,
    validation,
)

__all__ = ["AcceptanceSetupError", "main", "run"]

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
#: 最少通过数（T3 29 + T1 95 + T2 47 + T0.5 13 + 棘轮）
EXPECTED_MIN_PASSED = 200
BASELINE = Path("scripts/check/complexity_baseline.json")
REPORT_SCRIPT = Path("scripts/p6_report.py")
WORKFLOW_GENERATED_AT = "20261106T090000Z"
FORBIDDEN_PROJECTION_KEYS = ("verdict", "decision", "adopt", "gate_pass")


class AcceptanceSetupError(RuntimeError):
    """验收驱动前置条件不满足（输出目录形态非法/已存在）。"""


def _default_validation_id() -> str:
    return "p6_t3_acceptance_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


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


def _sibling(target: Path, suffix: str) -> Path:
    """同层兄弟目录（``<id>_audit_<profile>`` 形态；同样不得预先存在）。"""
    path = target.parent / f"{target.name}{suffix}"
    if path.exists():
        raise AcceptanceSetupError(f"兄弟目录已存在，拒绝覆盖：{path}")
    return path


def _run(command: Sequence[str], *, timeout: int = 1800) -> dict[str, Any]:
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
        "command": list(command),
        "exit_code": completed.returncode,
        "stdout_tail": [line for line in completed.stdout.splitlines() if line.strip()][-5:],
        "stderr_tail": [line for line in completed.stderr.splitlines() if line.strip()][-5:],
        "stdout": completed.stdout,
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
    """运行 T3 + T1 + T2 + 治理面定向集（禁字节码与 pytest 缓存）。"""
    files = list(T3_TEST_FILES) + list(T1_TEST_FILES) + list(T2_TEST_FILES)
    files += list(GOVERNANCE_TEST_FILES)
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
    """复杂度 hook 双口径（均须 exit=0、新增硬超限 0）。"""
    suites: dict[str, Any] = {}
    for name, paths in (
        ("src_only", ["src/lazybull"]),
        ("with_scripts", ["src/lazybull", "scripts"]),
    ):
        result = _run(
            [
                sys.executable,
                "scripts/check/check_complexity.py",
                "--paths",
                *paths,
                "--baseline",
                str(BASELINE),
            ]
        )
        tail = [line for line in result["stdout"].splitlines() if line.strip()][-3:]
        hard_new = 0
        for line in result["stdout"].splitlines():
            match = re.search(r"新增硬超限\s*(\d+)", line)
            if match:
                hard_new = int(match.group(1))
        suites[name] = {
            "command": result["command"],
            "exit_code": result["exit_code"],
            "tail": tail,
            "new_hard_overruns": hard_new,
            "passed": result["exit_code"] == 0 and hard_new == 0,
        }
    return {"suites": suites, "passed": all(item["passed"] for item in suites.values())}


def _digest_tree(root: Path) -> dict[str, str]:
    """目录内全部文件 → SHA-256（相对路径排序；不含 mtime）。"""
    return {
        path.relative_to(root).as_posix(): validation.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def corpus_determinism(work_root: Path) -> dict[str, Any]:
    """语料双跑：标准档构建两次，逐文件 SHA 与 content_sha256 全等。"""
    first = work_root / "corpus_a"
    second = work_root / "corpus_b"
    manifest_first = build_t3_corpus.build(first, profile="standard")
    manifest_second = build_t3_corpus.build(second, profile="standard")
    digests_first = _digest_tree(first)
    digests_second = _digest_tree(second)
    return {
        "content_sha256": {
            "first": manifest_first["profiles"]["standard"]["content_sha256"],
            "second": manifest_second["profiles"]["standard"]["content_sha256"],
        },
        "file_count": len(digests_first),
        "files_equal": digests_first == digests_second,
        "content_equal": (
            manifest_first["profiles"]["standard"]["content_sha256"]
            == manifest_second["profiles"]["standard"]["content_sha256"]
        ),
        "passed": digests_first == digests_second,
    }


def build_corpora(work_root: Path) -> dict[str, Any]:
    """构建标准/压力语料到 ``temp/``（已存在则复用，不覆盖）。"""
    profiles: dict[str, Any] = {}
    for profile in ("standard", "stress"):
        target = work_root / f"corpus_{profile}"
        target.mkdir(parents=True, exist_ok=True)
        payload = build_t3_corpus.build(target, profile=profile)
        profiles[profile] = payload["profiles"][profile]
    return {"profiles": profiles, "passed": bool(profiles)}


def _case_args(
    profile_root: Path,
    attempt: str,
    *,
    publish: bool,
    audit_out: Path | None,
    generated_at: str | None = WORKFLOW_GENERATED_AT,
) -> list[str]:
    """CLI 参数（两臂 runs；显式基线与证据类）。

    计时用固定 ``generated_at``（同输入可比字节）；**发布不传**该参数（新墙钟即新 run，
    否则同一语料在新 validation 里会撞上既有 report_id 而被拒绝覆盖）。
    """
    args = [
        sys.executable,
        str(REPORT_SCRIPT),
        "generate",
        "--kind",
        "experiment",
        "--evidence-class",
        "research",
        "--arm",
        f"base=runs:{profile_root / 'runs' / 'base'}",
        "--arm",
        f"arm=runs:{profile_root / 'runs' / 'arm'}",
        "--baseline",
        "base",
        "--attempt-id",
        attempt,
        "--json",
    ]
    if generated_at is not None:
        args += ["--generated-at", generated_at]
    if audit_out is not None:
        args += ["--audit-out", str(audit_out)]
    if not publish:
        args.append("--no-publish")
    return args


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


def timed_generations(profile_root: Path, profile: str) -> dict[str, Any]:
    """完整进程 5 次生成计时（不发布；父进程 monotonic 墙钟）。"""
    records: list[dict[str, Any]] = []
    for index in range(perf_gate.GENERATION_REPEATS):
        attempt = f"gen_time_{profile}_{index}"
        start = time.perf_counter()
        result = _run(_case_args(profile_root, attempt, publish=False, audit_out=None))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        payload = _last_json(result["stdout"]) if result["exit_code"] == 0 else {}
        records.append(
            {
                "index": index,
                "elapsed_ms": elapsed_ms,
                "exit_code": result["exit_code"],
                "verdict": payload.get("verdict"),
                "outputs": payload.get("outputs"),
                "manifest_sha256": payload.get("manifest_sha256"),
                "file_bytes": payload.get("file_bytes"),
                "stderr_tail": result["stderr_tail"],
            }
        )
    series = perf_gate.judge_generation_series([item["elapsed_ms"] for item in records])
    succeeded = all(item["exit_code"] == 0 and item["verdict"] == "pass" for item in records)
    # 八件字节身份：七份 outputs + manifest 自封存字节（B-06）
    identities = {
        json.dumps(
            {"outputs": item["outputs"], "manifest": item["manifest_sha256"]}, sort_keys=True
        )
        for item in records
    }
    byte_identical = len(identities) == 1
    return {
        "profile": profile,
        "records": records,
        "gate": series,
        "gate_mode": "no_publish",
        "audit_included": False,
        "all_succeeded": succeeded,
        "byte_identical": byte_identical,
        "passed": bool(series["passed"] and succeeded and byte_identical),
    }


def byte_identity(profile_root: Path, work_root: Path, profile: str) -> dict[str, Any]:
    """同 ``generated_at`` 双发布到两个独立根，逐字节比对 8 件产物（B-06）。"""
    from src.lazybull.v2.reporting import generate

    digests: list[dict[str, str]] = []
    for suffix in ("a", "b"):
        request = generate.ExperimentRequest(
            reports_root=work_root / f"byte_{profile}_{suffix}",
            evidence_class="research",
            generated_at=WORKFLOW_GENERATED_AT,
            attempt_id=f"gen_byte_{profile}_{suffix}",
            input_evidence_classes=("research",),
            arms=(
                f"base=runs:{profile_root / 'runs' / 'base'}",
                f"arm=runs:{profile_root / 'runs' / 'arm'}",
            ),
            baseline="base",
            publish_run=True,
        )
        result = generate.generate_experiment(request)
        digests.append(
            {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(result["files"].items())
            }
        )
    return {
        "profile": profile,
        "file_count": len(digests[0]),
        "equal": digests[0] == digests[1] and len(digests[0]) == 8,
        "digests": digests[0],
        "passed": digests[0] == digests[1] and len(digests[0]) == 8,
    }


def publish_with_audit(
    profile_root: Path, profile: str, validation_id: str
) -> dict[str, Any]:
    """真实发布一次并落精确写审计（审计目录为 validation 兄弟目录）。"""
    audit_out = (
        ROOT / contracts.REPORTS_ROOT / "validations" / "research" / f"{validation_id}_audit_{profile}"
    )
    if audit_out.exists():
        raise AcceptanceSetupError(f"审计目录已存在，拒绝覆盖：{audit_out}")
    started = time.perf_counter()
    result = _run(
        _case_args(
            profile_root,
            f"gen_publish_{profile}",
            publish=True,
            audit_out=audit_out,
            generated_at=None,
        )
    )
    audited_ms = (time.perf_counter() - started) * 1000.0
    payload = _last_json(result["stdout"]) if result["exit_code"] == 0 else {}
    audit_file = audit_out / "readonly_audit.json"
    audit: dict[str, Any] = {}
    if audit_file.is_file():
        raw = json.loads(audit_file.read_text(encoding="utf-8"))
        scope = dict(raw.get("scan_scope") or {})
        audit = {
            "verdict": raw.get("verdict"),
            "report_status": raw.get("report_status"),
            "uncovered": list(scope.get("uncovered") or []),
            "reparse_checked": scope.get("reparse_checked"),
            "reparse_points": list(scope.get("reparse_points") or []),
            "missing_expected": list(raw.get("missing_expected") or []),
            "unexpected_inside": list(raw.get("unexpected_inside") or []),
            "outside_whitelist": list(raw.get("outside_whitelist") or []),
            "foreign_root_paths": list(raw.get("foreign_root_paths") or []),
            "overwrote_existing": list(raw.get("overwrote_existing") or []),
            "expected_paths": list(raw.get("expected_paths") or []),
        }
    audit_ok = (
        audit.get("verdict") == "pass"
        and audit.get("uncovered") == []
        and audit.get("reparse_checked") is True
        and not any(
            audit.get(key)
            for key in (
                "missing_expected",
                "unexpected_inside",
                "outside_whitelist",
                "foreign_root_paths",
                "overwrote_existing",
            )
        )
    )
    return {
        "profile": profile,
        "exit_code": result["exit_code"],
        "payload": payload,
        "audit_path": str(audit_out.relative_to(ROOT)) if audit_out.exists() else None,
        "audit": audit,
        "audit_ok": audit_ok,
        # 带审计的完整进程墙钟（只登记、不入 10s 门；门禁口径为 --no-publish，B-07）
        "audited_process_ms": audited_ms,
        "stderr_tail": result["stderr_tail"],
        "passed": bool(result["exit_code"] == 0 and payload.get("verdict") == "pass" and audit_ok),
    }


def _expected_rows_from_projection(report_path: Path) -> dict[str, int]:
    """从已发布 projection 的**源侧计数**给出五份 CSV 期望行数（A-07）。"""
    projection = json.loads((report_path / "projection.json").read_text(encoding="utf-8"))
    arms = list(projection.get("arms") or [])
    stability = dict(projection.get("stability") or {})
    expected = dict(projection.get("csv_expected_rows") or {})
    return {
        "arm_metrics.csv": len(arms),
        # 完整 CSV 行集 = 源 summary ∩ csv_splits（默认并集；由生成侧按源侧计数登记）
        "fold_metrics.csv": int(
            expected.get("fold_metrics.csv")
            if expected.get("fold_metrics.csv") is not None
            else sum(int(arm.get("fold_row_count") or 0) for arm in arms)
        ),
        "chain_nav.csv": sum(int((arm.get("chain") or {}).get("rows_total") or 0) for arm in arms),
        "execution_attribution.csv": sum(
            int((arm.get("attribution") or {}).get("row_count") or 0) for arm in arms
        ),
        "stability.csv": int(stability.get("expected_row_count") or 0),
    }


def csv_conservation(report_path: Path, out_dir: Path, profile: str) -> dict[str, Any]:
    """五份 CSV 守恒：manifest schema + 文件字节 + **源侧期望行数** 三方对账。"""
    record = json.loads((report_path / "report_manifest.json").read_text(encoding="utf-8"))
    schemas = dict(record.get("csv_schemas") or {})
    expected = _expected_rows_from_projection(report_path)
    rows: dict[str, Any] = {}
    parse_errors: list[str] = []
    for name in experiment_csv.CSV_FILENAMES:
        path = report_path / name
        if not path.is_file():
            parse_errors.append(f"{name}: 文件缺失")
            continue
        text = path.read_text(encoding="utf-8")
        try:
            verified = experiment_csv.verify_csv_text(name, text)
        except (ValueError, OSError) as exc:
            parse_errors.append(f"{name}: {exc}")
            continue
        rows[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_count": verified["row_count"],
            "expected_row_count": expected.get(name),
            "columns": verified["columns"],
            "key_columns": verified["key_columns"],
            "manifest": schemas.get(name),
            "matches_manifest": verified == schemas.get(name),
            "matches_outputs": hashlib.sha256(path.read_bytes()).hexdigest()
            == (record.get("outputs") or {}).get(name),
        }
    checks = [
        {
            "name": "five_files_present",
            "passed": set(rows) == set(experiment_csv.CSV_FILENAMES),
        },
        {
            "name": "manifest_matches",
            "passed": bool(rows) and all(item["matches_manifest"] for item in rows.values()),
        },
        {
            "name": "outputs_match",
            "passed": bool(rows) and all(item["matches_outputs"] for item in rows.values()),
        },
        {
            "name": "row_count_matches_source",
            "passed": bool(rows)
            and all(
                item["expected_row_count"] is not None
                and item["row_count"] == item["expected_row_count"]
                for item in rows.values()
            ),
        },
        {"name": "parse_ok", "passed": not parse_errors},
    ]
    payload = {
        "report_id": record.get("report_id"),
        "report_path": str(report_path.relative_to(ROOT)),
        "rows": rows,
        "expected_rows": expected,
        "parse_errors": parse_errors,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    validation.write_json_new(evidence_dir / f"csv_conservation_{profile}.json", payload)
    return payload


def recompute_published(
    report_path: Path, roots: dict[str, Path] | None = None
) -> dict[str, Any]:
    """已发布 run 的链复算；``roots`` 给出时**按源文件字节重算输入身份**（B-01）。

    只把 manifest 自己的 ``inputs`` 回灌给 ``recompute_chain`` 时输入环恒真；这里按
    ``root_id:relative`` 回到源目录逐文件重算 SHA，任何漂移先在此定位为 ``input_drift``。
    """
    record = json.loads((report_path / "report_manifest.json").read_text(encoding="utf-8"))
    projection = json.loads((report_path / "projection.json").read_text(encoding="utf-8"))
    outputs = {
        name: validation.sha256_file(report_path / name)
        for name in record.get("outputs", {})
        if (report_path / name).is_file()
    }
    inputs = list(record.get("inputs") or [])
    verified = 0
    drift: list[str] = []
    unverified_roots: set[str] = set()
    if roots:
        recomputed: list[dict[str, Any]] = []
        for row in inputs:
            path_key = str(row.get("path"))
            root_id, relative = path_key.split(":", 1)
            root = roots.get(root_id)
            if root is None:
                # 未知根：不得用 manifest 自述行“补齐”输入环（B-R2-02）
                unverified_roots.add(root_id)
                drift.append(f"{path_key}:输入根未登记")
                recomputed.append(dict(row))
                continue
            source = Path(root) / relative
            if not source.is_file():
                drift.append(f"{path_key}:源文件缺失")
                recomputed.append(dict(row))
                continue
            digest = validation.sha256_file(source)
            verified += 1
            if digest != row.get("sha256"):
                drift.append(f"{path_key}:字节变化")
            recomputed.append(
                {"path": path_key, "sha256": digest, "size": source.stat().st_size}
            )
        inputs = recomputed
    if drift:
        return {
            "report_id": record.get("report_id"),
            "report_status": "invalid",
            "degradation_reason_code": "input_drift",
            "inputs_verified_from_source": verified,
            "unverified_roots": sorted(unverified_roots),
            "drift": drift,
            "passed": False,
        }
    row = manifest.recompute_chain(
        manifest=record,
        projection=projection,
        outputs=outputs,
        inputs=inputs,
    )
    return {
        "report_id": record.get("report_id"),
        "report_status": row.get("report_status"),
        "degradation_reason_code": row.get("degradation_reason_code"),
        "detail": row.get("detail"),
        "inputs_verified_from_source": verified,
        "passed": row.get("report_status") == "available",
    }


def _arm_rows(report_path: Path) -> dict[str, dict[str, str]]:
    """读回 ``arm_metrics.csv``（成品侧复核 Δ 方向用）。"""
    text = (report_path / "arm_metrics.csv").read_text(encoding="utf-8")
    import csv as csv_module
    import io

    reader = csv_module.DictReader(io.StringIO(text))
    return {row["arm_id"]: dict(row) for row in reader}


def delta_and_stability(report_path: Path) -> dict[str, Any]:
    """成品侧复核：ΔMaxDD 方向 = arm−baseline，且稳定性正例 period_count≥2。"""
    arms = _arm_rows(report_path)
    baseline = arms.get("base", {})
    arm = arms.get("arm", {})
    problems: list[str] = []
    delta: float | None = None
    expected: float | None = None
    try:
        delta = float(arm["delta_max_drawdown"])
        expected = float(arm["subset_max_drawdown"]) - float(baseline["subset_max_drawdown"])
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"ΔMaxDD 读取失败：{type(exc).__name__}: {exc}")
    if delta is None or expected is None or abs(delta - expected) > 1e-12:
        problems.append(f"ΔMaxDD={delta} 不等于 arm-base={expected}")
    if delta is not None and abs(delta) <= 1e-12:
        # 语料必须让两臂回撤不同，否则方向复核没有判别力（A-13）
        problems.append("ΔMaxDD 为 0：语料未构造可判别回撤差（方向复核无效）")
    if delta is not None and expected is not None and (delta > 0) != (expected > 0):
        problems.append(f"ΔMaxDD 符号与 arm-base 不一致：{delta} vs {expected}")
    stability_text = (report_path / "stability.csv").read_text(encoding="utf-8")
    import csv as csv_module
    import io

    stability_rows = list(csv_module.DictReader(io.StringIO(stability_text)))
    legal = [
        row
        for row in stability_rows
        if row["report_status"] == "available" and int(row["period_count"] or 0) >= 2
    ]
    if not legal:
        problems.append("稳定性正例缺失（无 period_count≥2 的 available 行）")
    if any(not (row["stability_score"] or "") for row in legal):
        problems.append("稳定性正例缺少 stability_score")
    return {
        "delta_max_drawdown": delta,
        "expected_delta": expected,
        "stability_rows": len(stability_rows),
        "legal_stability_rows": len(legal),
        "problems": problems,
        "passed": not problems,
    }


def _no_decision_fields(projection_path: Path) -> dict[str, Any]:
    """生成链不产裁决：投影中不得出现 verdict/decision/adopt/gate_pass 字段。"""
    text = projection_path.read_text(encoding="utf-8")
    hits = [key for key in FORBIDDEN_PROJECTION_KEYS if f'"{key}"' in text]
    return {"forbidden_keys": hits, "passed": not hits}


def _measure_open(report_path: Path, validation_id: str, profile: str) -> dict[str, Any]:
    """5 冷 + 5 热打开测量（兄弟 validation 目录；失败保留现场）。"""
    out = (
        ROOT
        / contracts.REPORTS_ROOT
        / "validations"
        / "research"
        / f"{validation_id}_open_{profile}"
    )
    try:
        payload = measure_report.measure(report_path / "report.html", out)
    except (measure_report.MeasureSetupError, OSError, RuntimeError) as exc:
        return {"profile": profile, "passed": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {
        "profile": profile,
        "path": str((out / "performance.json").relative_to(ROOT)),
        "verdict": payload.get("verdict"),
        "max_ms": (payload.get("gates") or {}).get("open", {}).get("max_ms"),
        "passed": payload.get("verdict") == "pass",
    }


def _profile_evidence(
    profile: str, profile_root: Path, target: Path, validation_id: str, work_root: Path
) -> dict[str, Any]:
    """单档证据：计时、发布审计、CSV 守恒、链复算、成品侧复核、打开测量。"""
    timing = timed_generations(profile_root, profile)
    published = publish_with_audit(profile_root, profile, validation_id)
    evidence: dict[str, Any] = {
        "profile": profile,
        "generation": timing,
        "publish": published,
        "byte_identity": byte_identity(profile_root, work_root, profile),
        "audited_process_ms": published.get("audited_process_ms"),
        "generation_gate_mode": "no_publish（发布/审计耗时另登记，不入 10s 门）",
        "input_roots_audited": 0,
        "input_roots_note": (
            "臂根位于仓库内 temp/（审计排除域）：写保护由运行期只读守卫承担；"
            "本轮审计不含输入根（B-09 已登记，不冒充已覆盖）"
        ),
    }
    if not published["passed"]:
        evidence["passed"] = False
        return evidence
    report_path = ROOT / str(published["payload"]["report_path"])
    size = (report_path / "report.html").stat().st_size
    evidence["size"] = {"bytes": size, "gate": perf_gate.judge_size(size)}
    evidence["csv"] = csv_conservation(report_path, target, profile)
    roots = {
        "arm_0": profile_root / "runs" / "base",
        "arm_1": profile_root / "runs" / "arm",
    }
    evidence["chain"] = recompute_published(report_path, roots)
    record = json.loads((report_path / "report_manifest.json").read_text(encoding="utf-8"))
    inputs = list(record.get("inputs") or [])
    unverified_roots = sorted(
        {str(row.get("path", "")).split(":", 1)[0] for row in inputs} - set(roots)
    )
    evidence["chain"]["manifest_input_count"] = len(inputs)
    evidence["chain"]["input_roots_unverified"] = unverified_roots
    if evidence["chain"].get("inputs_verified_from_source") != len(inputs):
        evidence["chain"]["passed"] = False
        evidence["chain"]["failure"] = "输入覆盖度不足（存在未回源校验的输入根）"
    evidence["delta_stability"] = delta_and_stability(report_path)
    evidence["no_decision"] = _no_decision_fields(report_path / "projection.json")
    evidence["open"] = _measure_open(report_path, validation_id, profile)
    evidence["report_path"] = str(report_path.relative_to(ROOT))
    evidence["passed"] = all(
        [
            timing["passed"],
            published["passed"],
            evidence["byte_identity"]["passed"],
            evidence["size"]["gate"]["passed"],
            evidence["csv"]["passed"],
            evidence["chain"]["passed"],
            evidence["delta_stability"]["passed"],
            evidence["no_decision"]["passed"],
            evidence["open"]["passed"],
        ]
    )
    return evidence


def _collect(validation_id: str) -> dict[str, Any]:
    target = _assert_new_validation_out(
        ROOT / contracts.REPORTS_ROOT / "validations" / "research" / validation_id
    )
    validation.ensure_new_dir(target)
    work_root = ROOT / "temp" / f"p6_t3_acceptance_{validation_id}"
    work_root.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    checks["tests"] = run_tests()
    checks["complexity"] = run_complexity()
    checks["corpus_determinism"] = corpus_determinism(work_root / "determinism")
    checks["corpora"] = build_corpora(work_root)
    profiles: dict[str, Any] = {}
    for profile in ("standard", "stress"):
        profiles[profile] = _profile_evidence(
            profile,
            work_root / f"corpus_{profile}" / profile,
            target,
            validation_id,
            work_root / f"byte_{profile}",
        )
    checks["profiles"] = profiles
    environment = env_probe.capture_environment(None)
    environment_gate = perf_gate.validate_environment(environment, require_browser=False)
    versions_gate = perf_gate.verify_acceptance_versions(
        {"jinja2": environment.get("jinja2_version")}, names=("jinja2",)
    )
    checks["environment"] = {
        "captured": environment,
        "validate_environment": environment_gate,
        "acceptance_versions": versions_gate,
        "passed": bool(environment_gate["passed"] and versions_gate["passed"]),
    }
    verdict = "pass" if all(_check_passed(item) for item in checks.values()) else "fail"
    payload = {
        "schema": "p6_t3_acceptance/v1",
        "validation_id": validation_id,
        "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "workflow_generated_at": WORKFLOW_GENERATED_AT,
        "verdict": verdict,
        "checks": checks,
        # 语料保留在 temp/ 供复算复核（B-01）；任务收尾按临时文件规范统一清理
        "work_root": str(work_root.relative_to(ROOT)),
    }
    validation.write_json_new(target / "t3_acceptance.json", _sanitize(payload))
    return payload


def _sanitize(value: Any) -> Any:
    """JSON 安全化：非有限浮点 → ``None``（``allow_nan=False`` 不得因度量值抛错，B-03）。"""
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _check_passed(value: Any) -> bool:
    """检查项是否通过：布尔直取；dict 优先 ``passed``，否则递归；标量视为“无判据”→ False。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        if "passed" in value:
            return bool(value["passed"])
        return all(_check_passed(item) for item in value.values())
    if isinstance(value, list):
        return all(_check_passed(item) for item in value)
    return False


def run(validation_id: str | None = None) -> dict[str, Any]:
    """执行完整 T3 验收并返回汇总载荷。

    兜底捕获 ``Exception``（保留 ``KeyboardInterrupt/SystemExit`` 透传）：任何未登记异常都必须
    落 ``t3_acceptance.json``（``verdict=fail`` + 异常类名），不得出现“只有半份证据、没有判定”
    的现场（run2 空载荷即为反例，B-03）。
    """
    identifier = validation_id or _default_validation_id()
    try:
        return _collect(identifier)
    except Exception as exc:  # noqa: BLE001 - 验收兜底必须覆盖全部异常类型
        target = ROOT / contracts.REPORTS_ROOT / "validations" / "research" / identifier
        payload = {
            "schema": "p6_t3_acceptance/v1",
            "validation_id": identifier,
            "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "verdict": "fail",
            "stage": type(exc).__name__,
            "detail": str(exc),
        }
        target.mkdir(parents=True, exist_ok=True)
        try:
            validation.write_json_new(target / "t3_acceptance.json", _sanitize(payload))
        except (OSError, validation.ValidationError) as write_exc:
            # 已存在（重复 id）时不覆盖既有现场；把“为什么没有载荷”写回返回值，可查
            payload["payload_write_error"] = f"{type(write_exc).__name__}: {write_exc}"
        return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P6 T3 验收驱动（P6-M3）")
    parser.add_argument("--validation-id", default=None)
    args = parser.parse_args(argv)
    payload = run(args.validation_id)
    print(json.dumps(payload, ensure_ascii=False))
    out = ROOT / contracts.REPORTS_ROOT / "validations" / "research" / str(
        payload.get("validation_id")
    )
    # 摘要写 stdout：验收驱动在 PowerShell 下不应因 stderr 被当作失败信号
    print(f"verdict={payload.get('verdict')} out={out}")
    return 0 if payload.get("verdict") == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())

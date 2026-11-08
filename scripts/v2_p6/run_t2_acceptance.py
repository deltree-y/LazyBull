# -*- coding: utf-8 -*-
"""P6 T2 验收驱动（P6-M2 可复跑证据包）。

一次运行产出 ``data/reports/p6/validations/research/<validation_id>/t2_acceptance.json``：

1. T2 七组 pytest（六区三态 / 证据与三端 / 注入与 CSP / 自包含 / 文档纪律 / 性能 / 确定性 /
   发布与审计负例）+ T1 七组 + T0.5 四组 + 两个棘轮（证明 T2 未回归前序阶段）；
2. 复杂度 hook 双口径（``src/lazybull`` 与 ``src/lazybull scripts``）；
3. **P6-T2-01** 恶意字段 DOM 快照（``evidence/dom_snapshot.json`` + 渲染件）；
4. **P6-T2-02** research 三端身份卡（``evidence/three_ends.json``，实盘端固定不可用）；
5. **P6-T2-03** 证据域隔离矩阵（生产/研究/校准三类跨类组合）——CLI 级**非零退出且零产物**
   （``evidence/evidence_isolation.json``）；
6. 标准/压力两档：5 次子进程生成计时（含 headroom 比）+ 真实发布 + 精确写审计
   （``audit/readonly_audit_*.json``）+ **已发布 run 的链复算**（``recompute_chain``）
   + 5 冷 5 热打开测量（兄弟目录 ``<id>_open_<profile>/performance.json``）；
7. 汇总 verdict（任一检查失败即 fail，保留原始子证据路径）；中途异常也落汇总（自解释现场）。

写域：本驱动只写本次新 validation 目录与其兄弟 ``_audit_*``/``_open_*`` 目录；
不写生产根，不写 ``logs/``/``temp/``（审计窗口内任何 logs 写入都会使审计失败）。
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

from scripts.v2_p6 import build_t2_corpus, env_probe, measure_report  # noqa: E402
from src.lazybull.v2.reporting import (  # noqa: E402
    contracts,
    html_check,
    manifest,
    perf_gate,
    validation,
)
from tests.p6_t1_env import build_paper_root, build_projection  # noqa: E402
from tests.p6_t2_env import (  # noqa: E402
    MALICIOUS_TEXTS,
    malicious_projection,
    render_projection,
)

__all__ = ["AcceptanceSetupError", "main", "run"]

T2_TEST_FILES: tuple[str, ...] = (
    "tests/test_v2_p6_t2_render_states.py",
    "tests/test_v2_p6_t2_evidence.py",
    "tests/test_v2_p6_t2_security.py",
    "tests/test_v2_p6_t2_selfcontained.py",
    "tests/test_v2_p6_t2_publish.py",
    "tests/test_v2_p6_t2_perf.py",
    "tests/test_v2_p6_t2_determinism.py",
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
GOVERNANCE_TEST_FILES: tuple[str, ...] = (
    "tests/test_v2_p6_t05_corpus.py",
    "tests/test_v2_p6_t05_deps.py",
    "tests/test_v2_p6_t05_isolation.py",
    "tests/test_v2_p6_t05_perf_gate.py",
    "tests/test_check_inuse_closure.py",
    "tests/test_check_complexity.py",
)
#: 最少通过数（低于即视为“测试未真的跑”；T2 ≈ 29 + T1 95 + T0.5 13 + 棘轮 ≈ 155）
EXPECTED_MIN_PASSED = 140
BASELINE = Path("scripts/check/complexity_baseline.json")
REPORT_SCRIPT = Path("scripts/p6_report.py")
EXIT_CODE_CONVENTION = {"pass": 0, "fail": 3, "setup_error": 2}
#: 计时/审计复跑共用的显式 generated_at（新墙钟即新 run，此处为协议固定值）
WORKFLOW_GENERATED_AT = "20261106T070000Z"
#: 证据域隔离矩阵（CLI 级：三类跨类组合 + 合法对照组）
ISOLATION_CASES: tuple[tuple[str, str], ...] = (
    ("production", "production,calibration_only"),
    ("research", "production,calibration_only"),
    ("calibration_only", "calibration_only,research"),
)


class AcceptanceSetupError(RuntimeError):
    """验收驱动前置条件不满足（输出目录形态非法/已存在）。"""


def _default_validation_id() -> str:
    return "p6_t2_acceptance_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


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
        "stdout": completed.stdout,
        "stdout_tail": [line for line in completed.stdout.splitlines() if line.strip()][-5:],
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
    """运行 T2 七组 + T1 七组 + 治理面定向集（禁字节码与 pytest 缓存）。"""
    files = list(T2_TEST_FILES) + list(T1_TEST_FILES) + list(GOVERNANCE_TEST_FILES)
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


def _extract_int(line: str, label: str) -> int | None:
    match = re.search(rf"{re.escape(label)}\s*[:：]?\s*(\d+)", str(line))
    return int(match.group(1)) if match else None


def run_complexity() -> dict[str, Any]:
    """复杂度 hook 双口径（均须 exit=0、新增硬超限 0、mccabe 真跑过）。"""
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
        line = next((item for item in reversed(result["stdout_tail"]) if "检查文件" in item), "")
        soft = _extract_int(line, "软超限")
        new_hard = _extract_int(line, "新增硬超限")
        suites[name] = {
            "exit_code": result["exit_code"],
            "summary_line": line,
            "new_hard": new_hard,
            "soft": soft,
            "mccabe_ran": bool(soft),
            "passed": result["exit_code"] == 0 and new_hard == 0 and bool(soft),
        }
    return {"suites": suites, "passed": all(item["passed"] for item in suites.values())}


def _git(arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return ""
    return completed.stdout or ""


def worktree_fingerprint() -> dict[str, Any]:
    """证据绑定代码状态：HEAD + 完整工作树摘要 + 驱动字节 SHA。"""
    status = _git(["status", "--porcelain"])
    driver = Path(__file__)
    return {
        "head": _git(["rev-parse", "HEAD"]).strip() or "<unavailable>",
        "status_lines": len([line for line in status.splitlines() if line.strip()]),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dirty": bool(status.strip()),
        "driver_path": driver.relative_to(ROOT).as_posix(),
        "driver_sha256": validation.sha256_file(driver),
        "python": sys.version.split()[0],
    }


def _render_case(tmp_path: Path, *, evidence_class: str) -> str:
    """合成纸面根 → 投影 → 渲染（证据用，不读真实数据）。"""
    built = build_paper_root(tmp_path / "paper")
    projection, _snapshot, _audit, _info = build_projection(
        built["root"],
        tmp_path / "staging",
        attempt_id="attempt_evidence",
        evidence_class=evidence_class,
    )
    return render_projection(projection)


def dom_evidence(tmp_path: Path, out_dir: Path) -> dict[str, Any]:
    """P6-T2-01：恶意字段 DOM 快照（注入串只作文本节点，属性闭集不外溢）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    html = render_projection(malicious_projection())
    scan = html_check.scan_html(html)
    nodes = html_check.text_snapshot(html)
    payload = {
        "schema": "p6_t2_dom_evidence/v1",
        "evidence_id": "P6-T2-01",
        "scan": scan,
        "injected_strings": list(MALICIOUS_TEXTS),
        "text_nodes_total": len(nodes),
        "text_nodes_sample": nodes[:60],
        "escaped_markers": {
            "u2028": "&#8232;" in html,
            "u2029": "&#8233;" in html,
            "lt_script": "&lt;script&gt;" in html,
            "scheme_neutralized": "https://" not in html,
        },
        "verdict": "pass" if scan["ok"] else "fail",
    }
    (out_dir / "malicious_report.html").write_bytes(html.encode("utf-8"))
    validation.write_json_new(out_dir / "dom_snapshot.json", payload)
    return payload


def three_ends_evidence(tmp_path: Path, out_dir: Path) -> dict[str, Any]:
    """P6-T2-02：research 三端卡存在、非 research 面缺省、实盘端固定不可用。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    research_html = _render_case(tmp_path / "research", evidence_class="research")
    production_html = _render_case(tmp_path / "production", evidence_class="production")
    texts = [node["text"] for node in html_check.text_snapshot(research_html)]
    live_rows = [text for text in texts if "实盘端" in text or "P8" in text]
    research_card = 'id="sec-three-ends"' in research_html
    production_card = 'id="sec-three-ends"' in production_html
    live_unavailable = any("P8 实盘端未建" in text for text in texts)
    payload = {
        "schema": "p6_t2_three_ends_evidence/v1",
        "evidence_id": "P6-T2-02",
        "research_has_three_ends": research_card,
        "production_has_three_ends": production_card,
        "live_end_rows": live_rows[:6],
        "live_end_not_available": live_unavailable,
        "production_mentions_live": ("实盘端" in production_html or "P8" in production_html),
        "verdict": "pass"
        if (
            research_card
            and not production_card
            and live_unavailable
            and "实盘端" not in production_html
        )
        else "fail",
    }
    validation.write_json_new(out_dir / "three_ends.json", payload)
    return payload


def _case_args(case: dict[str, Any]) -> list[str]:
    """把语料路径转成相对 repo 的命令行参数（避免绝对路径进入证据）。"""
    return [
        "--paper-root",
        str(Path(case["paper"]).relative_to(ROOT)),
        "--quality-root",
        str(Path(case["quality_root"]).relative_to(ROOT)),
    ]


def isolation_evidence(case: dict[str, Any], work_root: Path, out_dir: Path) -> dict[str, Any]:
    """P6-T2-03：三类跨类组合必须**非零退出且零产物**（不只看退出码）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, (evidence_class, inputs) in enumerate(ISOLATION_CASES):
        reports_root = work_root / f"iso_{index}"
        result = _run(
            [
                sys.executable,
                str(REPORT_SCRIPT),
                "generate",
                *_case_args(case),
                "--reports-root",
                str(reports_root.relative_to(ROOT)),
                "--evidence-class",
                evidence_class,
                "--input-evidence-classes",
                inputs,
                "--attempt-id",
                f"t2_isolation_{index}",
                "--json",
            ]
        )
        artifacts = (
            sorted(path.relative_to(reports_root).as_posix()
                   for path in reports_root.rglob("*"))
            if reports_root.exists()
            else []
        )
        results.append(
            {
                "evidence_class": evidence_class,
                "input_evidence_classes": inputs.split(","),
                "exit_code": result["exit_code"],
                "artifacts": artifacts,
                "stdout_tail": result["stdout_tail"],
                "stderr_tail": result["stderr_tail"],
                "passed": result["exit_code"] != 0 and not artifacts,
            }
        )
    payload = {
        "schema": "p6_t2_evidence_isolation/v1",
        "evidence_id": "P6-T2-03",
        "cases": results,
        "verdict": "pass" if all(item["passed"] for item in results) else "fail",
    }
    validation.write_json_new(out_dir / "evidence_isolation.json", payload)
    return payload


def _last_json(text: str) -> dict[str, Any]:
    """取子进程 stdout 的 JSON 对象（单行优先，退化为整体解析；缺失返回空 dict）。"""
    for line in reversed([item for item in text.splitlines() if item.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _report_file_digests(report_path: Any) -> tuple[dict[str, str], int]:
    """已发布 run 的逐文件 SHA 与 report.html 字节数。"""
    if not isinstance(report_path, str):
        return {}, 0
    target = Path(report_path)
    if not target.is_dir():
        return {}, 0
    digests = {
        path.name: validation.sha256_file(path)
        for path in sorted(target.iterdir())
        if path.is_file()
    }
    html = target / "report.html"
    return digests, html.stat().st_size if html.is_file() else 0


def _headroom(value: float | int | None, threshold: int) -> float | None:
    """实测/阈值（越小越安全；用于登记“灵敏度”，不参与判定）。"""
    if not isinstance(value, (int, float)) or not threshold:
        return None
    return round(float(value) / float(threshold), 6)


def _timed_generation(case: dict[str, Any], work_root: Path, profile: str) -> dict[str, Any]:
    """5 次子进程完整链路生成（同 generated_at），返回计时与字节一致性。

    生成门用**父进程墙钟**（``subprocess.run`` 前后 ``perf_counter``）：这才是设计 §7.1/§7.2
    的“完整进程”口径，包含解释器启动、pandas/numpy/pyarrow 导入与 CLI 装配；子进程自报的
    ``elapsed_ms`` 只覆盖 ``generate_dashboard`` 进入后的区间，仅作**诊断项**（T2-R3-07）。
    """
    work_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index in range(contracts.THRESHOLDS["generation_repeats"]):
        reports_root = work_root / f"run_{index}"
        started = time.perf_counter()
        result = _run(
            [
                sys.executable,
                str(REPORT_SCRIPT),
                "generate",
                *_case_args(case),
                "--reports-root",
                str(reports_root.relative_to(ROOT)),
                "--evidence-class",
                "research",
                "--generated-at",
                WORKFLOW_GENERATED_AT,
                "--attempt-id",
                f"t2_{profile}_{index}",
                "--json",
            ]
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        parsed = _last_json(result["stdout"])
        digests, size = _report_file_digests(parsed.get("report_path"))
        records.append(
            {
                "index": index,
                "exit_code": result["exit_code"],
                "wall_ms": round(wall_ms, 1),
                "elapsed_ms": parsed.get("elapsed_ms"),
                "report_id": parsed.get("report_id"),
                "report_path": parsed.get("report_path"),
                "report_bytes": size,
                "file_sha256": digests,
                "stderr_tail": result["stderr_tail"],
            }
        )
    wall = [record["wall_ms"] for record in records if isinstance(record["wall_ms"], (int, float))]
    inprocess = [
        record["elapsed_ms"]
        for record in records
        if isinstance(record["elapsed_ms"], (int, float))
    ]
    gate = perf_gate.judge_generation_series(wall)
    inprocess_gate = perf_gate.judge_generation_series(inprocess)
    stable = all(record["file_sha256"] == records[0]["file_sha256"] for record in records)
    size_gate = perf_gate.judge_size(records[0]["report_bytes"])
    return {
        "profile": profile,
        "records": records,
        "gate": gate,
        "in_process_gate": inprocess_gate,
        "size_gate": size_gate,
        "byte_identical": stable,
        "headroom": {
            "generation": _headroom(gate["max_ms"], perf_gate.MAX_GENERATION_MS),
            "size": _headroom(records[0]["report_bytes"], perf_gate.MAX_REPORT_BYTES),
            "note": (
                "生成门为**完整进程**墙钟（父进程 subprocess.run 前后 perf_counter）；"
                "in_process_gate 仅诊断；打开门为页面自报 loadEventEnd"
            ),
        },
        "passed": (
            gate["passed"]
            and stable
            and size_gate["passed"]
            and all(record["exit_code"] == 0 for record in records)
        ),
    }


def _chain_check(report_path: str | None) -> dict[str, Any]:
    """对已发布 run 复算 manifest 链（projection/outputs 与登记逐环比对）。"""
    if not report_path:
        return {"passed": False, "detail": "无已发布 run"}
    target = Path(report_path)
    try:
        projection = json.loads((target / "projection.json").read_text(encoding="utf-8"))
        record = json.loads((target / "report_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "detail": f"{type(exc).__name__}: {exc}"}
    # 只复算 manifest 已登记的产出（manifest 自身不入 outputs，避免自指）
    declared = record.get("outputs") if isinstance(record.get("outputs"), dict) else {}
    missing = [name for name in declared if not (target / name).is_file()]
    if missing:
        return {"passed": False, "detail": f"manifest 登记的产出缺失：{missing}"}
    outputs = {name: validation.sha256_file(target / name) for name in declared}
    status = manifest.recompute_chain(
        manifest=record, projection=projection, outputs=outputs, inputs=record["inputs"]
    )
    return {
        "passed": status["report_status"] == "available",
        "report_status": status["report_status"],
        "reason_code": status.get("degradation_reason_code"),
        "checks": len(status.get("checks") or []),
        "detail": status.get("detail"),
    }


def _publish_and_audit(case: dict[str, Any], validation_id: str, profile: str) -> dict[str, Any]:
    """真实发布 + 精确写审计（审计落兄弟 validation 目录，随后复制进主证据包）。"""
    audit_dir = (
        ROOT / contracts.REPORTS_ROOT / "validations/research" / f"{validation_id}_audit_{profile}"
    )
    result = _run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "generate",
            *_case_args(case),
            "--evidence-class",
            "research",
            "--audit-out",
            str(audit_dir.relative_to(ROOT)),
            "--json",
        ]
    )
    payload = _last_json(result["stdout"])
    return {
        "profile": profile,
        "command": result["command"],
        "exit_code": result["exit_code"],
        "audit_dir": str(audit_dir.relative_to(ROOT)),
        "audit_json": str((audit_dir / "readonly_audit.json").relative_to(ROOT)),
        "report_path": payload.get("report_path"),
        "payload_verdict": payload.get("verdict"),
        "payload_scan_scope": payload.get("scan_scope"),
        "stderr_tail": result["stderr_tail"],
        "passed": result["exit_code"] == 0 and payload.get("verdict") == "pass",
        "chain": _chain_check(payload.get("report_path")),
    }


def _copy_audits(published: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """把兄弟目录的审计件复制进主证据包（只复制，不改判据；不覆盖既有文件）。

    同时登记**源与副本的 SHA-256**：让“复制后被改写”可被发现（摘要自身不可防篡改，
    属诚实边界——T2-R2-N4）。
    """
    audits: dict[str, Any] = {}
    target_dir = out_dir / "audit"
    target_dir.mkdir(parents=True, exist_ok=True)
    for profile, record in published.items():
        source = ROOT / record["audit_json"]
        if not source.is_file():
            audits[profile] = {"copied": None, "verdict": "missing"}
            continue
        target = target_dir / f"readonly_audit_{profile}.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        validation.write_json_new(target, payload)
        audits[profile] = {
            "copied": str(target.relative_to(ROOT)),
            "source": str(source.relative_to(ROOT)),
            "source_sha256": validation.sha256_file(source),
            "copy_sha256": validation.sha256_file(target),
            "verdict": payload.get("verdict", "missing"),
            "scan_scope": payload.get("scan_scope"),
        }
    return audits


def _measure_opens(
    published: dict[str, dict[str, Any]], identifier: str
) -> dict[str, Any]:
    """5 冷 + 5 热打开测量（真实 Chromium；结果写兄弟 validation 目录）。"""
    opens: dict[str, Any] = {}
    for profile, record in published.items():
        report_path = record.get("report_path")
        if not report_path:
            opens[profile] = {"passed": False, "detail": "无已发布 HTML"}
            continue
        measure_out = (
            ROOT / contracts.REPORTS_ROOT / "validations/research" / f"{identifier}_open_{profile}"
        )
        try:
            performance = measure_report.measure(Path(report_path) / "report.html", measure_out)
        except (measure_report.MeasureSetupError, OSError, RuntimeError) as exc:
            opens[profile] = {"passed": False, "detail": f"{type(exc).__name__}: {exc}"}
            continue
        opens[profile] = {
            "passed": performance["verdict"] == "pass",
            "performance_json": str((measure_out / "performance.json").relative_to(ROOT)),
            "report_bytes": performance["html"]["bytes"],
            "size_gate": performance["gates"]["size"],
            "open_max_ms": performance["gates"]["open"]["max_ms"],
            "cold_count": performance["gates"]["open"]["cold_count"],
            "hot_count": performance["gates"]["open"]["hot_count"],
            "requests_passed": performance["gates"]["requests"]["passed"],
            "environment_passed": performance["gates"]["environment"]["passed"],
            "versions_passed": performance["gates"]["acceptance_versions"]["passed"],
            "headroom": {
                "open": _headroom(
                    performance["gates"]["open"]["max_ms"], perf_gate.MAX_OPEN_MS
                ),
                "size": _headroom(
                    performance["html"]["bytes"], perf_gate.MAX_REPORT_BYTES
                ),
                "note": "打开门为页面自报 loadEventEnd（不含浏览器进程启动）",
            },
        }
    return opens


def _collect(
    out_dir: Path, identifier: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """执行全部检查并就地写入 ``payload``（异常由 ``run`` 负责登记）。"""
    payload["stage"] = "corpus"
    cases = {
        profile: build_t2_corpus.build_case(out_dir / "corpus" / profile, profile)
        for profile in ("standard", "stress")
    }
    payload["stage"] = "pytest"
    tests = run_tests()
    payload["stage"] = "complexity"
    complexity = run_complexity()
    payload["stage"] = "evidence"
    dom = dom_evidence(out_dir / "work" / "evidence_render", out_dir / "evidence")
    three_ends = three_ends_evidence(out_dir / "work" / "three_ends", out_dir / "evidence")
    isolation = isolation_evidence(
        cases["standard"], out_dir / "work" / "isolation", out_dir / "evidence"
    )
    payload["stage"] = "generation"
    generation = {
        profile: _timed_generation(cases[profile], out_dir / "work" / profile, profile)
        for profile in ("standard", "stress")
    }
    payload["stage"] = "publish"
    published = {
        profile: _publish_and_audit(cases[profile], identifier, profile)
        for profile in ("standard", "stress")
    }
    audits = _copy_audits(published, out_dir)
    payload["stage"] = "open"
    opens = _measure_opens(published, identifier)
    payload["stage"] = "aggregate"

    checks = {
        "pytest": tests["passed"],
        "complexity": complexity["passed"],
        "dom_snapshot": dom["verdict"] == "pass",
        "three_ends": three_ends["verdict"] == "pass",
        "evidence_isolation": isolation["verdict"] == "pass",
        "generation": all(item["passed"] for item in generation.values()),
        "published": all(item["passed"] for item in published.values()),
        "audit": all(item["verdict"] == "pass" for item in audits.values()),
        "audit_scan_scope": all(
            (item["scan_scope"] or {}).get("uncovered") == [] for item in audits.values()
        ),
        # 审计件“源/副本 SHA 相等”入判据（不只登记）：复制后改写必须使验收 fail
        "audits_intact": all(
            bool(item.get("source_sha256"))
            and item.get("source_sha256") == item.get("copy_sha256")
            for item in audits.values()
        ),
        "chain": all(item["chain"]["passed"] for item in published.values()),
        "open": all(item["passed"] for item in opens.values()),
    }
    payload.update(
        {
            "checks": checks,
            "verdict": "pass" if all(checks.values()) else "fail",
            "pytest": tests,
            "complexity": complexity,
            "evidence": {
                "dom_snapshot": str(
                    (out_dir / "evidence" / "dom_snapshot.json").relative_to(ROOT)
                ),
                "three_ends": str((out_dir / "evidence" / "three_ends.json").relative_to(ROOT)),
                "evidence_isolation": str(
                    (out_dir / "evidence" / "evidence_isolation.json").relative_to(ROOT)
                ),
            },
            "corpus": {
                profile: {
                    "scale": case["scale"],
                    "quality_abnormal_rows_written": case["quality_abnormal_rows_written"],
                }
                for profile, case in cases.items()
            },
            "generation": generation,
            "published": published,
            "audits": audits,
            "open": opens,
            "environment": env_probe.capture_environment(None),
            "exit_code_convention": EXIT_CODE_CONVENTION,
        }
    )
    return payload


def run(validation_id: str | None = None) -> dict[str, Any]:
    """执行全部检查并落证据包；中途异常也写汇总（verdict=fail + stage + 异常类名）。"""
    identifier = validation_id or _default_validation_id()
    out_dir = _assert_new_validation_out(
        ROOT / contracts.REPORTS_ROOT / "validations/research" / identifier
    )
    validation.ensure_new_dir(out_dir)
    started = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "schema": "p6_t2_acceptance/v2",
        "validation_id": identifier,
        "generated_at": validation.utc_stamp(started),
        "stage": "start",
        "verdict": "fail",
        "checks": {},
    }
    try:
        _collect(out_dir, identifier, payload)
    except BaseException as exc:  # 现场必须自解释：记录阶段与异常，再落盘
        payload["stage"] = f"exception:{payload.get('stage')}"
        payload["exception"] = f"{type(exc).__name__}: {exc}"
        payload["verdict"] = "fail"
    payload["worktree"] = worktree_fingerprint()
    validation.write_json_new(out_dir / "t2_acceptance.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口；退出码 = 0 pass / 3 fail / 2 setup_error。"""
    parser = argparse.ArgumentParser(description="P6 T2 验收驱动（T2 证据包）")
    parser.add_argument("--validation-id", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = run(args.validation_id)
    except AcceptanceSetupError as exc:
        print(f"设置错误：{exc}", file=sys.stderr)
        return EXIT_CODE_CONVENTION["setup_error"]
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"verdict={payload['verdict']} stage={payload.get('stage')} "
            f"checks={payload.get('checks')}"
        )
    return EXIT_CODE_CONVENTION["pass" if payload["verdict"] == "pass" else "fail"]


if __name__ == "__main__":
    raise SystemExit(main())

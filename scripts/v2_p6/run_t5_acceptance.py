# -*- coding: utf-8 -*-
"""P6 T5 验收驱动（P6-M5：联验 + 独立真实 golden/smoke + 双平台）。

一次运行产出 ``validations/research/<validation_id>/t5_acceptance.json``，并就地写出三面
生成原始记录；打开测量与真实 golden 的原始 JSON 落在同名兄弟目录（``<id>_open_*``、
``<id>_golden``、``<id>_audit_*``），均属本次运行的新 validation 产物。

检查项（计划 §3-T5 的 7 条验收 / 14 场景）：

1. ``pytest``：T5 + T4 + T3 + T2 + T1 + T0.5 + 治理面定向集（禁字节码/pytest 缓存）；
2. ``complexity``：复杂度 hook 双口径（``src/lazybull`` 与 ``src/lazybull scripts``）；
3. ``real_golden``：**非 pytest** 调用 ``check_real_golden.py`` 核验真实 P5b 三族
   （seq10/7/21）的路径集合、逐文件 SHA、seal、链与关键值；真实根缺失/不符即失败；
4. ``real_smoke``：三条**只读**真实 smoke（P4 生产驾驶舱 / 真实 ``data/runs`` 趋势 /
   真实 P5b campaign 趋势），每条带产品精确写审计 + 驱动侧独立整树判定 +
   输入根 before/after 全树路径+SHA 比对 + 内容期望断言；
5. ``zero_write``：汇总“白名单外零差异、白名单内恰为本次产物”，并登记扫描域/排除树
   边界与运行期守卫（禁网络/禁子进程）证据；
6. ``three_faces``：三面 × 标准/压力 的 5 次完整进程生成 + 5 冷 5 热打开 + 体积门；
7. ``reproduce``：三面同输入同 ``generated_at`` 的**全链重建**逐字节一致；
8. ``dual_platform``：Windows 侧 canonical 哈希 + Linux 侧探测/执行结论（不可用时显式
   ``not_achieved`` 并给出可执行准备包，**不得**用“跳过”代替）；
9. ``historical``：由 smoke 前后整树快照证明既有文件零改写/零删除、新增仅在允许集合；
10. ``documentation``：版本号与文档同步（CHANGELOG/README/CLAUDE/计划/闭环报告）。

写域（设计 §2.5）：只写 ``temp/p6_t5_acceptance_<id>/**`` 与本次 validation 目录及其
兄弟 ``_open_*``/``_golden``/``_audit_*`` 目录；不写任何生产数据根。
"""

# 复杂度豁免：阶段专用验收驱动（scripts/v2_p6，明确验收专用、不入生产链路，见 CLAUDE.md §7.8）。
# 拆分计划：本文件只保留“编排 + 判定 + 证据装配”；`pytest`/`complexity`(前序驱动)、语料
# (build_t2/t3/t4_corpus)、写域审计(readonly_audit/publish) 均已外置。若继续增长，下一步把
# “真实 smoke”（smoke_p4/runs/p5b 及其期望常量）拆到 `scripts/v2_p6/t5_real_smoke.py`。

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
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

from scripts.v2_p6 import (  # noqa: E402
    build_t2_corpus,
    build_t3_corpus,
    build_t4_corpus,
    env_probe,
    measure_report,
    run_t2_acceptance,
    run_t3_acceptance,
    run_t4_acceptance,
)
from src.lazybull.v2.core.accounting.engine_state import loads_blob  # noqa: E402
from src.lazybull.v2.reporting import (  # noqa: E402
    contracts,
    dashboard_view,
    generate,
    manifest as manifest_module,
    perf_gate,
    publish,
    readonly_audit,
    render,
    validation,
)

__all__ = ["AcceptanceSetupError", "main", "run"]

T5_TEST_FILES: tuple[str, ...] = (
    "tests/test_v2_p6_t5_real_golden.py",
    "tests/test_v2_p6_t5_acceptance.py",
    "tests/test_v2_p6_t5_selftest_guard.py",
)
#: T5 测试文件**磁盘真值**（用于“清单覆盖”判据：新增文件不得漏收，评审 R3-T5-R1-03）
T5_TEST_GLOB = "test_v2_p6_t5_*.py"


def _missing_t5_test_files() -> list[str]:
    """磁盘上存在但未纳入 ``T5_TEST_FILES`` 的 T5 测试文件。"""
    on_disk = {
        f"tests/{path.name}" for path in (ROOT / "tests").glob(T5_TEST_GLOB) if path.is_file()
    }
    return sorted(on_disk - set(T5_TEST_FILES))


FACE_TEST_FILES: tuple[str, ...] = (
    tuple(run_t4_acceptance.T4_TEST_FILES)
    + tuple(run_t4_acceptance.T3_TEST_FILES)
    + tuple(run_t4_acceptance.T1_TEST_FILES)
    + tuple(run_t4_acceptance.T2_TEST_FILES)
    + tuple(run_t4_acceptance.GOVERNANCE_TEST_FILES)
)
#: 最少通过数（T5 定向 ≈ 318 + T4 53 + T5 新增，留出下界）
EXPECTED_MIN_PASSED = 330
BASELINE = Path("scripts/check/complexity_baseline.json")
REPORT_SCRIPT = Path("scripts/p6_report.py")
GOLDEN_SCRIPT = Path("scripts/v2_p6/check_real_golden.py")
#: 三面固定 generated_at（**按面沿用各自验收协议的冻结值**：复算必须与基准轮同值）
FACE_GENERATED_AT: dict[str, str] = {
    "dashboard": str(run_t2_acceptance.WORKFLOW_GENERATED_AT),
    "experiment": str(run_t3_acceptance.WORKFLOW_GENERATED_AT),
    "trend": str(run_t4_acceptance.WORKFLOW_GENERATED_AT),
}
#: 真实 smoke 的兜底 generated_at（正常路径用 ``_utc_stamp()`` 的墙钟值）
WORKFLOW_GENERATED_AT = "20261107T100000Z"


def _utc_stamp() -> str:
    """真实 smoke 的墙钟 ``generated_at``（同输入同墙钟 ⇒ 同 report_id）。

    真实 smoke 必须用**墙钟**：报告根是生产根，固定 stamp 会让第二次验收与上一轮的
    run 目录同名（CLI 正确 fail-closed 拒绝覆盖）——证据轮次靠 validation id 区分，
    报告 run 靠墙钟区分；两者都在载荷里登记。
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


#: 三面档位
PROFILES: tuple[str, ...] = ("standard", "stress")
KINDS: tuple[str, ...] = ("dashboard", "experiment", "trend")
#: 真实根（生产只读；本驱动不写其中任何路径）
REAL_PAPER_ROOT = Path("data/paper")
REAL_QUALITY_ROOT = Path("data/reports/quality")
REAL_RUNS_ROOT = Path("data/runs")
REAL_BUNDLE_ROOT = Path("data/paper_shadow/p5b_testbed")
#: 真实 P5b smoke 的 campaign（**钉死**，与 golden 冻结件同一 campaign；新增 campaign 不得静默漂移目标）
REAL_BUNDLE_CAMPAIGN = "p5b_t5_drawdown_20261104c"
REAL_GOLDEN_FREEZE = Path("docs/contracts/p6_t5_real_golden_freeze_20261107.json")
#: 用户裁决的完成判据豁免（机器可读；与 CLAUDE.md §5-T5 ⑭、计划 §24.7 同文）
USER_WAIVERS: tuple[dict[str, str], ...] = (
    {
        "id": "L1",
        "decided_at": "2026-11-08",
        "decision": "用户裁决：真实 Linux 端到端可不达成，作为阶段豁免",
        "scope": "不要求真机 Linux 执行与跨平台字节对照；准备包与自检证据保留，可随时补做",
    },
    {
        "id": "L2",
        "decided_at": "2026-11-08",
        "decision": "用户裁决：层 2（包内 projection→HTML 逐字节重建）降为诊断项",
        "scope": "复算判据唯一取层 3（同输入同 generated_at 全链重建逐字节一致）",
    },
)
#: 已知限制登记（机器可读；与计划 §24.4、闭合报告 §5 同编号）
KNOWN_LIMITATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "L1",
        "item": "真实 Linux 全链与跨平台字节对照",
        "status": "waived_by_user",
        "detail": "无满足 Python>=3.12 + 同依赖的 Linux 环境（WSL 为 3.10.12 且缺 pandas），"
        "未获装依赖/联网授权；**2026-11-08 用户裁决豁免**，准备包与自检证据保留",
    },
    {
        "id": "L2",
        "item": "设计 §8.3 层 2（包内 projection→HTML 逐字节重建）",
        "status": "waived_by_user",
        "detail": "footer 以 dict repr 渲染 generator_files，canonical JSON 往返丢失插入序"
        "（实测差异 1 行全在该 footer）；**2026-11-08 用户裁决降为诊断项**，复算判据唯一取层 3",
    },
    {
        "id": "L3",
        "item": "audit 域文件/字节量化",
        "status": "not_measured",
        "detail": "产品审计只给 scan_scope/changed，未计数；驱动侧登记 changed 全文与树摘要",
    },
    {
        "id": "L4",
        "item": "UNC/`file://host` 检测面",
        "status": "not_covered",
        "detail": "_payload_paths 未覆盖到 UNC/file URI 形态；本轮载荷反斜杠 0 命中",
    },
    {
        "id": "L5",
        "item": "多源 manifest 输入复核",
        "status": "not_covered",
        "detail": "_verify_inputs_on_disk 仍多源 fail-closed，未扩展到多源合并",
    },
    {
        "id": "L6",
        "item": "1,200 点抽样端到端",
        "status": "not_triggered",
        "detail": "页面可见窗口 ≤500 行不触发抽样；由单测与 2,000 点用例承担",
    },
    {
        "id": "L7",
        "item": "NFC/NFD 名称端到端字节对照",
        "status": "not_triggered",
        "detail": "仅单测覆盖；真实根无 NFD 名称样本，未做端到端字节对照",
    },
    {
        "id": "L8",
        "item": "`prev/` 副本随包或 previous 位置登记增强",
        "status": "not_implemented",
        "detail": "T4 已登记，T5 未做",
    },
    {
        "id": "L9",
        "item": "P4 smoke 期望为钉住态",
        "status": "by_design",
        "detail": "data/paper 新增业务日/成交后本 smoke 必失败；重基线需新冻结与新 validation id",
    },
    {
        "id": "L10",
        "item": "跨平台复算包的 trend 面对路径敏感",
        "status": "by_design",
        "detail": "root_id 由来源根的仓库相对路径派生，trend 每行 run_key/source 含 root_id ⇒ 跨机"
        "对照必须在**同一仓库相对路径**重建语料（属设计约束而非缺口）；准备包已按记录路径重建并在"
        "本机自检三面 MATCH；真机执行属 L1（已豁免）",
    },
)
#: 真实 P4 期望（计划 §3-T5 第 3 条：2026-11-05 已验证修复状态，非由报告反推）
EXPECTED_P4_NAV_DAYS: tuple[str, ...] = (
    "20260506",
    "20260519",
    "20260522",
    "20260527",
    "20260601",
    "20260603",
)
EXPECTED_P4_LAST_TOTAL = 619016.1802662
EXPECTED_P4_INITIAL_CAPITAL = 650000.0
EXPECTED_P4_ACCOUNT_REASON = "run_record_pre_switch_shape"
EXPECTED_P4_MODEL_REASON = "no_persisted_actual_model"
#: 文档同步检查（版本号唯一来源 + 必须同步的文件）
PLAN_DOC = "docs/plans/v2/p6_report_plan_20261104.md"
CLOSURE_DOC = "docs/reports/p6_t5_closure_20261107.md"
DOC_FILES: tuple[str, ...] = (
    "CHANGELOG.md",
    "README.md",
    "CLAUDE.md",
    PLAN_DOC,
    CLOSURE_DOC,
)


class AcceptanceSetupError(RuntimeError):
    """验收驱动前置条件不满足（输出目录形态非法/已存在）。"""


def _default_validation_id() -> str:
    return "p6_t5_acceptance_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


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


def _sibling(validation_id: str, suffix: str, evidence_class: str = "research") -> Path:
    """本次运行的同名兄弟目录（必须不存在）。"""
    target = (
        ROOT / contracts.REPORTS_ROOT / "validations" / evidence_class / f"{validation_id}{suffix}"
    )
    if target.exists():
        raise AcceptanceSetupError(f"兄弟目录已存在，拒绝覆盖：{target}")
    return target


def _run(command: Sequence[str], *, timeout: int = 3600, cwd: Path | None = None) -> dict[str, Any]:
    """执行子进程并记录实际命令/退出码/首尾输出（不吞错、不重定向到 logs）。"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        list(command),
        cwd=str(cwd or ROOT),
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


def _parse_pytest_counts(lines: Sequence[str]) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for line in lines:
        for key in counts:
            match = re.search(rf"(\d+)\s+{key}", line)
            if match:
                counts[key] = int(match.group(1))
    return counts


def run_tests() -> dict[str, Any]:
    """运行 T5 + 前序各面 + 治理面定向集（禁字节码与 pytest 缓存）。"""
    missing = _missing_t5_test_files()
    files = list(T5_TEST_FILES) + list(FACE_TEST_FILES)
    result = _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files])
    tail = [line for line in result["stdout"].splitlines() if line.strip()][-4:]
    counts = _parse_pytest_counts(tail)
    return {
        "command": _posix_command(_relativize(result["command"])),
        "exit_code": result["exit_code"],
        "stdout_tail": tail,
        "stderr_tail": result["stderr_tail"],
        "counts": counts,
        "expected_min_passed": EXPECTED_MIN_PASSED,
        "t5_test_files": list(T5_TEST_FILES),
        # 评审 R3-T5-R1-03：磁盘上存在但清单未收的 T5 测试文件（必须为空）
        "t5_files_not_collected": missing,
        "passed": (
            result["exit_code"] == 0
            and counts["failed"] == 0
            and counts["skipped"] == 0
            and counts["passed"] >= EXPECTED_MIN_PASSED
            and not missing
        ),
    }


def run_complexity() -> dict[str, Any]:
    """复杂度 hook 双口径（均须 exit=0、新增硬超限 0、基线未改动）。"""
    baseline_sha_before = hashlib.sha256((ROOT / BASELINE).read_bytes()).hexdigest()
    suites: dict[str, Any] = {}
    for name, paths in (
        ("src_only", ["src/lazybull"]),
        ("with_scripts", ["src/lazybull", "scripts"]),
    ):
        # ``--paths`` 是 ``nargs='+'``：多次出现只有最后一次生效（会把双口径退化成单口径）
        command = [
            sys.executable,
            "scripts/check/check_complexity.py",
            "--baseline",
            str(BASELINE),
            "--paths",
            *paths,
        ]
        result = _run(command)
        hard_lines = [
            line for line in result["stdout"].splitlines() if line.strip().startswith("[hard]")
        ]
        new_hard = [line for line in hard_lines if "存量基线" not in line and "已豁免" not in line]
        summary = [line for line in result["stdout"].splitlines() if "新增硬超限" in line][-1:]
        declared = None
        if summary:
            match = re.search(r"新增硬超限\s*(\d+)\s*条", summary[0])
            declared = int(match.group(1)) if match else None
        suites[name] = {
            "command": _posix_command(_relativize(command)),
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
    baseline_sha_after = hashlib.sha256((ROOT / BASELINE).read_bytes()).hexdigest()
    unchanged = baseline_sha_before == baseline_sha_after
    return {
        "suites": suites,
        "baseline": {
            "path": BASELINE.as_posix(),
            "sha256_before": baseline_sha_before,
            "sha256_after": baseline_sha_after,
            "unchanged": unchanged,
        },
        "passed": all(item["passed"] for item in suites.values()) and unchanged,
    }


def _tree_digest(roots: Mapping[str, Path]) -> dict[str, Any]:
    """多根输入树的路径+SHA+字节全量摘要（before/after 比对用，独立于产品审计）。

    **fail-closed**：输入根缺失或为空即 ``AcceptanceSetupError``——否则 ``entries=[]`` 会让
    ``unchanged=True`` 真空成立（评审 B2-1）。
    """
    entries: list[dict[str, Any]] = []
    for root_id, root in sorted(roots.items()):
        base = Path(root)
        if not base.is_dir():
            raise AcceptanceSetupError(f"输入根缺失或非目录：{root_id}={base}")
        per_root = 0
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            entries.append(
                {
                    "root": root_id,
                    "path": path.relative_to(base).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size": path.stat().st_size,
                }
            )
            per_root += 1
        if per_root == 0:
            raise AcceptanceSetupError(f"输入根为空（0 文件）：{root_id}={base}")
    digest = hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "files": len(entries),
        "bytes": sum(int(item["size"]) for item in entries),
        "digest": digest,
        "entries": entries,
    }


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
    """命令记录归一为 POSIX 分隔符（执行仍用原列表，记录可跨平台复制）。"""
    return [str(item).replace("\\", "/") for item in command]


def _rel_text(path: Path | str) -> str:
    """仓库内路径 → 相对 POSIX（命令参数用）；仓库外保持绝对（显式边界）。"""
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _relativize(value: Any) -> Any:
    """证据载荷路径归一：绝对路径（含 ``kind:path``）→ 仓库相对 POSIX。"""
    if isinstance(value, str):
        if value == sys.executable:
            return "python"
        if ":" in value and not value.startswith(("http", "file")):
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


def _read_json(path: Path | str) -> Any:
    """读 JSON（失败返回 ``None``；调用方负责判据）。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _check(name: str, passed: bool, detail: str = "", **extra: Any) -> dict[str, Any]:
    """单条判据记录（``passed`` 缺失即视为失败，不做默认通过）。"""
    return {"name": name, "passed": bool(passed), "detail": detail, **extra}


def _all_passed(checks: Sequence[Mapping[str, Any]]) -> bool:
    return bool(checks) and all(bool(item.get("passed")) for item in checks)


#: 跳过类调用名（AST 判定：只看**真实调用**，docstring 里写“禁用 skipif”不算）
SKIP_CALL_NAMES: frozenset[str] = frozenset(
    {"skip", "skipif", "xfail", "importorskip", "skip_module"}
)


def _source_skips(relative: Path) -> list[str]:
    """核验器源码里的跳过标记（AST；空列表 = 干净）。

    “命令里没写 ``--skipif``”是同义反复（命令是本脚本自己拼的），判定改为静态扫描：
    出现 ``import pytest``／``from pytest import ...`` 或任何 ``*.skip/skipif/xfail/
    importorskip`` 调用即算不干净；docstring 里的说明文字不构成命中。
    """
    import ast

    path = ROOT / relative
    if not path.is_file():
        return [f"缺文件:{relative}"]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        return [f"解析失败:{exc}"]
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest":
                    findings.append("import pytest")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") == "pytest":
                findings.append("from pytest import ...")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in SKIP_CALL_NAMES:
                findings.append(f"call:{name}")
    return sorted(set(findings))


def _has_no_skip_flags(command: Sequence[str]) -> bool:
    """命令行不含跳过开关（记录用；判定以源码静态扫描为准）。"""
    joined = " ".join(str(item) for item in command)
    return "skipif" not in joined and "xfail" not in joined


def _within(relative: str, prefix: str) -> bool:
    """路径是否落在前缀内：**精确等值**或前缀后紧跟分隔符（避免 ``run4`` 放行 ``run40*``）。"""
    base = prefix.rstrip("/")
    return relative == base or relative.startswith(base + "/")


def _read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    """读 CSV 为行字典列表（UTF-8、无 BOM、严格列序由契约侧保证）。"""
    import csv

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


# ---------------------------------------------------------------------------
# 真实 smoke：审计 + 内容断言
# ---------------------------------------------------------------------------


def _driver_patterns(audit_relative: str) -> list[str]:
    """驱动侧写域判定模式：generate 白名单 + 本次 validation 目录（审计件落点）。

    **不是**放宽产品审计：``generate`` 的允许写集仍是 T1 冻结的那两条；这里额外登记的
    只是本命令的审计文件落点（设计 §2.5 明确它不属于 generate 写集，由调用方写入新的
    validation 目录）。判定仍是“白名单外零差异 ∧ 白名单内恰为 expected_paths”。
    """
    return list(contracts.WRITE_WHITELIST["generate"]) + [f"{audit_relative}/**"]


def _driver_audit(
    *,
    before: Any,
    after: Any,
    expected_paths: Sequence[str],
    allowed_patterns: Sequence[str],
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    """驱动侧独立写域判定（直接调 T1 ``audit_command``，判据与阈值全不变）。"""
    uncovered = publish.uncovered_paths(ROOT)
    checked = all(
        isinstance(getattr(tree, "reparse", None), (frozenset, set, tuple, list))
        for tree in (before, after)
    )
    reparse = sorted(
        set(getattr(before, "reparse", ()) or ()) | set(getattr(after, "reparse", ()) or ())
    )
    violations = [
        {"action": "scan_scope", "path": item, "kind": "unregistered_scan_scope"}
        for item in uncovered
    ]
    if not checked:
        violations.append(
            {"action": "reparse_scan", "path": "<missing>", "kind": "reparse_not_checked"}
        )
    report = readonly_audit.audit_command(
        command="generate",
        before=before,
        after=after,
        expected_paths=list(expected_paths),
        allowed_patterns=list(allowed_patterns),
        bindings=dict(bindings),
        reparse_paths=reparse,
        extra_violations=violations,
        output_root="worktree",
    )
    report["scan_scope"] = {
        "includes": list(publish.AUDIT_SCAN_INCLUDES),
        "excludes": list(publish.AUDIT_SCAN_EXCLUDES),
        "uncovered": uncovered,
        "reparse_checked": checked,
        "reparse_points": reparse,
        "note": "驱动侧独立判定（同一 T1 audit_command；白名单 = generate + 本次 validation 目录）",
    }
    return report


def _audited_generate(
    *,
    label: str,
    kind: str,
    evidence_class: str,
    attempt: str,
    extra_args: Sequence[str],
    audit_out: Path,
    input_roots: Mapping[str, Path],
    scan_extra_roots: Mapping[str, Path],
) -> dict[str, Any]:
    """跑一次真实只读 generate（带产品审计）并做驱动侧独立写域判定。

    证据形态是「**两个快照 + 一个独立实现**」（不是三套独立判据，评审 B1-2/B1-3）：
    ① 产品侧 ``readonly_audit.json``（CLI 内 T1 ``audit_command`` 判定 + 运行期守卫）；
    ② 驱动侧 ``_driver_audit``（**直接调** T1 ``readonly_audit.audit_command``，白名单仅多登记本次
       validation 目录；与①同判据、快照时点不同）；
    ③ 驱动侧输入根 before/after 全树 ``path+SHA+size`` 比对（``_tree_digest``，**独立实现**，
       覆盖产品审计的排除树）。
    """
    if audit_out.exists():
        raise AcceptanceSetupError(f"审计目录已存在，拒绝覆盖：{audit_out}")
    audit_relative = audit_out.relative_to(ROOT).as_posix()
    command = [
        sys.executable,
        str(REPORT_SCRIPT),
        "generate",
        "--kind",
        kind,
        "--evidence-class",
        evidence_class,
        *[str(item) for item in extra_args],
        "--attempt-id",
        attempt,
        "--audit-out",
        audit_relative,
        "--json",
    ]
    before = publish.scan_audit_roots(repo_root=ROOT, extra_roots=dict(scan_extra_roots))
    input_before = _tree_digest(input_roots)
    started = time.perf_counter()
    result = _run(command)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    input_after = _tree_digest(input_roots)
    after = publish.scan_audit_roots(repo_root=ROOT, extra_roots=dict(scan_extra_roots))
    payload = _last_json(result["stdout"]) if result["exit_code"] == 0 else {}
    run_dir = Path(str(payload["report_path"])) if payload.get("report_path") else None
    names = (
        sorted(path.name for path in run_dir.iterdir() if path.is_file())
        if run_dir and run_dir.is_dir()
        else []
    )
    report_id = str(payload.get("report_id") or "")
    driver_audit: dict[str, Any] = {}
    if report_id and names:
        expected = [
            f"{contracts.REPORTS_ROOT}/{evidence_class}/{kind}/{report_id}/{name}" for name in names
        ]
        expected.append(f"{audit_relative}/readonly_audit.json")
        driver_audit = _driver_audit(
            before=before,
            after=after,
            expected_paths=expected,
            allowed_patterns=_driver_patterns(audit_relative),
            bindings={
                "attempt": attempt,
                "evidence_class": evidence_class,
                "kind": kind,
                "report_id": report_id,
            },
        )
    product = _read_json(audit_out / "readonly_audit.json") or {}
    scope = dict(product.get("scan_scope") or {})
    return {
        "label": label,
        "command": _posix_command(_relativize(command)),
        "command_args": _posix_command(_relativize(command[1:])),
        "interpreter": "python",
        "exit_code": result["exit_code"],
        "elapsed_ms": round(elapsed_ms, 1),
        "verdict": payload.get("verdict"),
        "report_id": report_id or None,
        "generated_at": payload.get("generated_at"),
        "report_path": run_dir.as_posix() if run_dir else None,
        "report_files": names,
        "file_bytes": dict(payload.get("file_bytes") or {}),
        "stderr_tail": result["stderr_tail"],
        "cli_failure": (
            None
            if result["exit_code"] == 0
            else {
                "exit_code": result["exit_code"],
                "stdout_tail": [line for line in result["stdout"].splitlines() if line.strip()][
                    -3:
                ],
                "stderr_tail": result["stderr_tail"],
            }
        ),
        "audit": {
            "path": f"{audit_relative}/readonly_audit.json",
            "verdict": product.get("verdict"),
            "uncovered": list(scope.get("uncovered") or []),
            "reparse_checked": scope.get("reparse_checked"),
            "reparse_points": list(scope.get("reparse_points") or []),
            "missing_expected": list(product.get("missing_expected") or []),
            "unexpected_inside": list(product.get("unexpected_inside") or []),
            "outside_whitelist": list(product.get("outside_whitelist") or []),
            "foreign_root_paths": list(product.get("foreign_root_paths") or []),
            "overwrote_existing": list(product.get("overwrote_existing") or []),
            "forbidden_paths": list(product.get("forbidden_paths") or []),
        },
        "driver_audit": {
            "verdict": driver_audit.get("verdict"),
            "outside_whitelist": list(driver_audit.get("outside_whitelist") or []),
            "unexpected_inside": list(driver_audit.get("unexpected_inside") or []),
            "missing_expected": list(driver_audit.get("missing_expected") or []),
            "foreign_root_paths": list(driver_audit.get("foreign_root_paths") or []),
            "overwrote_existing": list(driver_audit.get("overwrote_existing") or []),
            "changed": list(driver_audit.get("changed") or []),
            "uncovered": list((driver_audit.get("scan_scope") or {}).get("uncovered") or []),
            "allowed_patterns": list(driver_audit.get("allowed_patterns") or []),
        },
        "input_evidence": {
            "roots": sorted(input_roots),
            "files": input_before["files"],
            "bytes": input_before["bytes"],
            "digest_before": input_before["digest"],
            "digest_after": input_after["digest"],
            "unchanged": input_before["entries"] == input_after["entries"],
        },
        "tree_before_digest": hashlib.sha256(
            json.dumps({key: value for key, value in before.items()}, sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest(),
        "tree_after_digest": hashlib.sha256(
            json.dumps({key: value for key, value in after.items()}, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "_before_tree": before,
        "_after_tree": after,
    }


def _audit_checks(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """把一次真实 smoke 的写域证据折成判定条目（三条证据缺一即失败）。"""
    audit = dict(record.get("audit") or {})
    driver = dict(record.get("driver_audit") or {})
    inputs = dict(record.get("input_evidence") or {})
    return [
        _check("exit_code_zero", record.get("exit_code") == 0, f"exit={record.get('exit_code')}"),
        _check("payload_verdict", record.get("verdict") == "pass", str(record.get("verdict"))),
        _check("product_audit_pass", audit.get("verdict") == "pass", str(audit.get("verdict"))),
        _check(
            "product_audit_clean",
            audit.get("verdict") == "pass"
            and not audit.get("uncovered")
            and not audit.get("unexpected_inside")
            and not audit.get("outside_whitelist")
            and not audit.get("foreign_root_paths")
            and not audit.get("overwrote_existing")
            and not audit.get("forbidden_paths")
            and audit.get("reparse_checked") is True,
            "uncovered/unexpected/outside/foreign/overwrote/forbidden 全空且已检查重解析点",
        ),
        _check("driver_audit_pass", driver.get("verdict") == "pass", str(driver.get("verdict"))),
        _check(
            "driver_audit_clean",
            driver.get("verdict") == "pass"
            and not driver.get("outside_whitelist")
            and not driver.get("unexpected_inside")
            and not driver.get("missing_expected")
            and not driver.get("foreign_root_paths")
            and not driver.get("overwrote_existing")
            and not driver.get("uncovered"),
            "白名单外零差异、expected 全中、无跨根/覆盖/盲区",
        ),
        _check("inputs_unchanged", bool(inputs.get("unchanged")), f"roots={inputs.get('roots')}"),
    ]


def _authority_nav_rows() -> dict[str, dict[str, float]]:
    """从权威 blob 独立读出 ``trade_date → {total_value, cash, position_value}``。

    **不经报告实现**：直接 ``loads_blob`` 读 ``state/engine_state.bin`` 的
    ``portfolio_values``，作为 NAV 三原值的独立对账来源。
    """
    raw = (ROOT / REAL_PAPER_ROOT / "state" / "engine_state.bin").read_bytes()
    blob = loads_blob(raw)
    fields = blob.get("fields") if isinstance(blob, Mapping) else None
    values = (fields or {}).get("portfolio_values") if isinstance(fields, Mapping) else None
    rows: dict[str, dict[str, float]] = {}
    for item in values or ():
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("date") or "")
        day = text[:10].replace("-", "") if text else ""
        if len(day) != 8 or not day.isdigit():
            continue
        rows[day] = {
            "total_value": float(item.get("portfolio_value")),
            "cash": float(item.get("capital")),
            "position_value": float(item.get("market_value")),
        }
    return rows


def smoke_p4(validation_id: str) -> dict[str, Any]:
    """真实 P4 生产驾驶舱只读 smoke（期望 = 2026-11-05 已验证修复状态）。"""
    audit_out = _sibling(validation_id, "_audit_p4", "production")
    record = _audited_generate(
        label="real_p4_dashboard_production",
        kind="dashboard",
        evidence_class="production",
        attempt=f"{validation_id}_p4",
        extra_args=[
            "--paper-root",
            REAL_PAPER_ROOT.as_posix(),
            "--quality-root",
            REAL_QUALITY_ROOT.as_posix(),
            "--generated-at",
            _utc_stamp(),
        ],
        audit_out=audit_out,
        input_roots={"paper": ROOT / REAL_PAPER_ROOT, "quality": ROOT / REAL_QUALITY_ROOT},
        scan_extra_roots={},
    )
    checks = _audit_checks(record)
    run_dir = Path(str(record["report_path"])) if record.get("report_path") else None
    projection = _read_json(run_dir / "projection.json") if run_dir else None
    checks.append(_check("projection_present", isinstance(projection, Mapping), "projection.json"))
    if isinstance(projection, Mapping):
        account = dict(projection.get("account") or {})
        instructions = dict(projection.get("instructions") or {})
        config = dict(projection.get("config") or {})
        nav = dict(projection.get("nav") or {})
        model = dict(config.get("model") or {})
        actual_model = dict(model.get("actual_model") or {})
        annualization = dict(config.get("annualization") or {})
        nav_values = dict(nav.get("nav_values") or {})
        nav_rows = list(nav.get("rows") or [])
        checks.extend(
            [
                _check(
                    "account_degraded_pre_switch",
                    account.get("report_status") == "degraded"
                    and account.get("degradation_reason_code") == EXPECTED_P4_ACCOUNT_REASON,
                    f"{account.get('report_status')}/{account.get('degradation_reason_code')}",
                ),
                _check(
                    "instructions_not_available_pre_switch",
                    instructions.get("report_status") == "not_available"
                    and instructions.get("degradation_reason_code") == EXPECTED_P4_ACCOUNT_REASON,
                    f"{instructions.get('report_status')}/"
                    f"{instructions.get('degradation_reason_code')}",
                ),
                _check(
                    "model_no_source",
                    actual_model.get("report_status") == "not_available"
                    and actual_model.get("degradation_reason_code") == EXPECTED_P4_MODEL_REASON,
                    f"{actual_model.get('report_status')}/"
                    f"{actual_model.get('degradation_reason_code')}",
                ),
                _check(
                    "nav_region_available",
                    nav.get("report_status") == "available"
                    and nav.get("annualization_allowed") is True,
                    f"{nav.get('report_status')}/annualization_allowed="
                    f"{nav.get('annualization_allowed')}",
                ),
                _check(
                    "nav_days_exact",
                    tuple(sorted(nav_values)) == EXPECTED_P4_NAV_DAYS
                    and all(
                        str(row.get("trade_date")) in EXPECTED_P4_NAV_DAYS
                        and row.get("matched") is True
                        for row in nav_rows
                    ),
                    f"days={sorted(nav_values)}",
                ),
                _check(
                    "nav_last_total_exact",
                    float(
                        nav_values.get(EXPECTED_P4_NAV_DAYS[-1], {}).get(
                            "total_value", float("nan")
                        )
                    )
                    == EXPECTED_P4_LAST_TOTAL,
                    f"total={nav_values.get(EXPECTED_P4_NAV_DAYS[-1], {}).get('total_value')}",
                ),
                _check(
                    "annualization_available",
                    annualization.get("report_status") == "available"
                    and isinstance(annualization.get("annualized_return"), (int, float))
                    and annualization.get("current_value") == EXPECTED_P4_LAST_TOTAL
                    and annualization.get("initial_capital") == EXPECTED_P4_INITIAL_CAPITAL,
                    f"annualized={annualization.get('annualized_return')} "
                    f"current={annualization.get('current_value')}",
                ),
            ]
        )
        authority = _authority_nav_rows()
        mismatches = []
        for day in EXPECTED_P4_NAV_DAYS:
            row = dict(nav_values.get(day) or {})
            source = authority.get(day)
            if source is None:
                mismatches.append(f"{day}: 权威缺该行")
                continue
            for key in ("total_value", "cash", "position_value"):
                if row.get(key) != source[key]:
                    mismatches.append(f"{day}.{key}: {row.get(key)} ≠ {source[key]}")
        checks.append(
            _check(
                "nav_three_values_match_authority",
                not mismatches,
                "; ".join(mismatches[:5]) if mismatches else "6 业务日三原值与权威逐值相等",
                compared_days=list(EXPECTED_P4_NAV_DAYS),
            )
        )
    audit_count = len(_audit_checks(record))
    record["content_checks"] = checks[audit_count:]
    record["checks"] = checks
    record["passed"] = _all_passed(checks)
    return record


#: 原生提交批次的最小文件集合（缺一即不是规范化 run 批次）
NATIVE_BATCH_FILES = ("batch_meta.json", "summary.parquet", "chain_nav.parquet")
#: paper replay 目录的标记（P4 回放旁路产物，不是原生提交批次）
PAPER_REPLAY_MARKERS = ("paper_state", "checkpoints")


def _native_batch(path: Path) -> tuple[bool, str, str]:
    """判定一个 ``data/runs`` 子目录是否**原生提交批次**；返回 ``(是否, 原因, created_at)``。

    评审 A-01 口径：不能用“目录名排序取末 N 个”冒充“最近”，也不能只按 ``summary.parquet``
    存在性判定（真实根里有含 ``paper_state``/``checkpoints`` 的 P4 paper replay 目录同样带
    summary）。本函数要求 ``batch_meta.json`` + ``summary.parquet`` + ``chain_nav.parquet`` +
    ``folds/`` 齐备、且不含回放标记，最近性以 ``batch_meta.created_at`` 为准。
    """
    missing = [name for name in NATIVE_BATCH_FILES if not (path / name).is_file()]
    if missing:
        marker = [name for name in PAPER_REPLAY_MARKERS if (path / name).exists()]
        reason = f"缺 {','.join(missing)}" + (f"；含回放标记 {','.join(marker)}" if marker else "")
        return False, reason, ""
    if not (path / "folds").is_dir():
        return False, "缺 folds/ 目录", ""
    marker = [name for name in PAPER_REPLAY_MARKERS if (path / name).exists()]
    if marker:
        return False, f"含回放标记 {','.join(marker)}（非原生提交批次）", ""
    try:
        meta = json.loads((path / "batch_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"batch_meta.json 不可读：{exc}", ""
    created = str(meta.get("created_at") or "")
    return True, "原生提交批次", created


def _scan_runs_root() -> tuple[list[tuple[str, Path]], dict[str, str]]:
    """扫 ``data/runs``：返回 ``(已按 created_at 升序接受的原生批次, 拒绝清单→原因)``。"""
    root = ROOT / REAL_RUNS_ROOT
    accepted: list[tuple[str, Path]] = []
    rejected: dict[str, str] = {}
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        native, reason, created = _native_batch(path)
        if native:
            accepted.append((created, path))
        else:
            rejected[path.name] = reason
    accepted.sort(key=lambda item: (item[0], item[1].name))
    return accepted, rejected


def _newest_native_names(limit: int) -> list[str]:
    """按 ``batch_meta.created_at`` 排序的原生批次名（末 N 个）——供判据复核。"""
    accepted, _rejected = _scan_runs_root()
    return [path.name for _created, path in accepted[-limit:]]


def _real_runs_batches(limit: int = 2) -> tuple[list[Path], dict[str, str]]:
    """真实 ``data/runs`` 下**最近的原生提交批次**（按 ``batch_meta.created_at`` 排序取末 N）。

    返回 ``(批次目录, 被排除目录 → 原因)``：真实根是**异构**的（P4 paper replay 目录带
    ``paper_state``，有的还带 ``summary.parquet``），整根当批次必须被趋势来源白名单 fail-closed 拒绝。
    """
    accepted, rejected = _scan_runs_root()
    return [path for _, path in accepted[-limit:]], rejected


def smoke_runs(validation_id: str) -> dict[str, Any]:
    """真实 ``data/runs`` 只读 smoke（研究趋势面：真实规范化批次 + 异构根 fail-closed 负例）。"""
    batches, others = _real_runs_batches()
    if not batches:
        raise AcceptanceSetupError(f"真实 runs 批次缺失（{REAL_RUNS_ROOT} 下无原生提交批次）")
    # 负例：整根当批次必须非零退出（异构根混入非批次目录 ⇒ 白名单 fail-closed），零产物
    negative_root = ROOT / "temp" / f"p6_t5_acceptance_{validation_id}" / "runs_negative_root"
    if negative_root.exists():
        raise AcceptanceSetupError(f"负例输出根已存在，拒绝覆盖：{negative_root}")
    negative = _run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "generate",
            "--kind",
            "trend",
            "--evidence-class",
            "research",
            "--source",
            f"runs:{REAL_RUNS_ROOT.as_posix()}",
            "--reports-root",
            _rel_text(negative_root),
            "--attempt-id",
            "t5_runs_negative",
            "--generated-at",
            _utc_stamp(),
            "--no-publish",
            "--json",
        ]
    )
    negative_payload = _last_json(negative["stdout"])
    negative_artifacts = (
        sorted(path.name for path in negative_root.rglob("*") if path.is_file())
        if negative_root.exists()
        else []
    )
    audit_out = _sibling(validation_id, "_audit_runs")
    record = _audited_generate(
        label="real_runs_trend_research",
        kind="trend",
        evidence_class="research",
        attempt=f"{validation_id}_runs",
        extra_args=[
            *[
                item
                for path in batches
                for item in ("--source", f"runs:{path.relative_to(ROOT).as_posix()}")
            ],
            "--generated-at",
            _utc_stamp(),
        ],
        audit_out=audit_out,
        input_roots={"runs": ROOT / REAL_RUNS_ROOT},
        scan_extra_roots={},
    )
    checks = _audit_checks(record)
    run_dir = Path(str(record["report_path"])) if record.get("report_path") else None
    index_rows = _read_csv_rows(run_dir / "run_index.csv") if run_dir else []
    trend_rows = _read_csv_rows(run_dir / "trend.csv") if run_dir else []
    checks.extend(
        [
            _check(
                "heterogeneous_root_rejected",
                negative["exit_code"] != 0
                and str(negative_payload.get("stage")) == "trend_source"
                and not negative_artifacts,
                f"exit={negative['exit_code']} stage={negative_payload.get('stage')} "
                f"artifacts={len(negative_artifacts)}",
            ),
            _check(
                "index_one_row_per_batch",
                len(index_rows) == len(batches),
                f"rows={len(index_rows)} batches={len(batches)}",
            ),
            _check(
                "trend_five_metrics_per_run",
                len(trend_rows) == 5 * len(index_rows),
                f"rows={len(trend_rows)}",
            ),
            _check(
                "index_mode_full",
                all(str(row.get("index_mode")) == "full" for row in index_rows),
                "全部 run 行 index_mode=full",
            ),
            _check(
                "identity_sha256_present",
                all(len(str(row.get("source_identity_sha256") or "")) == 64 for row in index_rows),
                "每行 source_identity_sha256 为 64 hex",
            ),
            _check(
                "manifest_absent_registered",
                all(
                    str(row.get("source_manifest_sha256") or "") in ("", "None")
                    for row in index_rows
                ),
                "真实 runs 批次无 manifest ⇒ source_manifest_sha256 空（登记 non-manifest 身份）",
            ),
            _check(
                "trend_keys_unique",
                len({(row.get("run_key"), row.get("metric_name")) for row in trend_rows})
                == len(trend_rows),
                "trend.csv (run_key, metric_name) 唯一",
            ),
            _check(
                "batches_are_native",
                all(_native_batch(path)[0] for path in batches),
                f"batches={[path.name for path in batches]}",
            ),
            _check(
                "batches_newest_by_meta",
                [path.name for path in batches] == _newest_native_names(len(batches)),
                "按 batch_meta.created_at 排序取末 N（顺序与 created_at 一致）",
            ),
        ]
    )
    record["checks"] = checks
    record["passed"] = _all_passed(checks)
    record["batches"] = [path.relative_to(ROOT).as_posix() for path in batches]
    record["batches_created_at"] = {path.name: _native_batch(path)[2] for path in batches}
    record["rejected_dirs"] = dict(sorted(others.items()))
    record["negative_root_rejected"] = {
        "command": _posix_command(_relativize(negative["command"])),
        "exit_code": negative["exit_code"],
        "stage": negative_payload.get("stage"),
        "detail": negative_payload.get("detail"),
        "artifacts": negative_artifacts,
    }
    return record


def smoke_p5b(validation_id: str) -> dict[str, Any]:
    """真实 P5b campaign 只读 smoke（calibration_only 趋势面，单 bundle）。"""
    chosen = ROOT / REAL_BUNDLE_ROOT / REAL_BUNDLE_CAMPAIGN
    bundle_dir = chosen / "reports"
    if not (bundle_dir / "drawdown_manifest.json").is_file():
        raise AcceptanceSetupError(
            f"真实 P5b campaign 缺失（{REAL_BUNDLE_ROOT / REAL_BUNDLE_CAMPAIGN} 无回撤族 bundle）"
        )
    manifest_bytes = (bundle_dir / "drawdown_manifest.json").read_bytes()
    expected_identity = hashlib.sha256(manifest_bytes).hexdigest()
    native_report = _read_json(bundle_dir / "drawdown_report.json") or {}
    sample = dict(native_report.get("sample_sources") or {})
    window = dict(sample.get("b0_window") or {})
    expected_mdd = window.get("mdd_percent")
    audit_out = _sibling(validation_id, "_audit_p5b", "calibration_only")
    record = _audited_generate(
        label="real_p5b_campaign_trend_calibration",
        kind="trend",
        evidence_class="calibration_only",
        attempt=f"{validation_id}_p5b",
        extra_args=[
            "--bundle",
            bundle_dir.relative_to(ROOT).as_posix(),
            "--generated-at",
            _utc_stamp(),
        ],
        audit_out=audit_out,
        input_roots={"campaign": chosen},
        scan_extra_roots={"campaign": chosen},
    )
    checks = _audit_checks(record)
    run_dir = Path(str(record["report_path"])) if record.get("report_path") else None
    index_rows = _read_csv_rows(run_dir / "run_index.csv") if run_dir else []
    trend_rows = _read_csv_rows(run_dir / "trend.csv") if run_dir else []
    mdd_rows = [row for row in trend_rows if row.get("metric_name") == "max_drawdown"]
    identity_ok = bool(index_rows) and all(
        str(row.get("source_identity_sha256")) == expected_identity for row in index_rows
    )
    mdd_value = None
    if mdd_rows:
        try:
            mdd_value = float(mdd_rows[0].get("value"))
        except (TypeError, ValueError):
            mdd_value = None
    checks.extend(
        [
            _check("single_bundle_row", len(index_rows) == 1, f"rows={len(index_rows)}"),
            _check(
                "identity_is_bundle_manifest_sha",
                identity_ok,
                f"expected={expected_identity[:12]}",
            ),
            _check(
                "log_mdd_matches_native",
                isinstance(expected_mdd, (int, float))
                and mdd_value is not None
                and float(mdd_value) == float(expected_mdd)
                and str(mdd_rows[0].get("report_status")) == "available",
                f"trend={mdd_value} native={expected_mdd}",
            ),
        ]
    )
    record["checks"] = checks
    record["passed"] = _all_passed(checks)
    record["campaign"] = chosen.relative_to(ROOT).as_posix()
    return record


def _history_of(record: Mapping[str, Any], allowed_prefixes: Sequence[str]) -> dict[str, Any]:
    """由一次 smoke 的前后整树快照证明“既有文件零改写/零删除、新增仅在允许集合”。"""
    before = dict(record.get("_before_tree") or {})
    after = dict(record.get("_after_tree") or {})
    modified = sorted(
        key
        for key in before
        if key in after and before[key].get("sha256") != after[key].get("sha256")
    )
    removed = sorted(key for key in before if key not in after)
    added = sorted(key for key in after if key not in before)
    unexpected = sorted(
        key
        for key in added
        if not any(_within(key.split(":", 1)[-1], prefix) for prefix in allowed_prefixes)
    )
    return {
        "label": record.get("label"),
        "pre_existing_files": len(before),
        "modified": modified[:10],
        "removed": removed[:10],
        "added": len(added),
        "added_outside_allowed": unexpected[:10],
        "allowed_prefixes": list(allowed_prefixes),
        "passed": not modified and not removed and not unexpected,
    }


#: 报告包内禁止 import 的模块（``urllib.parse`` 是纯字符串解析，允许）
FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        "socket",
        "socketio",
        "requests",
        "httpx",
        "aiohttp",
        "http.client",
        "urllib.request",
        "urllib.error",
        "subprocess",
    }
)
#: 报告包内禁止的调用名（``os.system``/``os.popen``/``urlopen`` 一类起进程/联网入口）
FORBIDDEN_CALLS: frozenset[str] = frozenset({"system", "popen", "urlopen", "Popen"})


def _static_network_scan() -> dict[str, Any]:
    """AST 静态扫描 `src/lazybull/v2/reporting/**`：不联网、不起子进程（守卫窗口外的替代证据）。

    只认**真实 import 与调用**（docstring 里的 ``subprocess.Popen`` 文本与
    ``urllib.parse.urlsplit`` 不算），避免“文本 grep 假命中”把结论做废。
    """
    import ast

    package = ROOT / "src" / "lazybull" / "v2" / "reporting"
    hits: list[dict[str, str]] = []
    files = sorted(path for path in package.rglob("*.py") if path.is_file())
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            hits.append({"file": relative, "kind": "parse_error", "detail": str(exc)})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in FORBIDDEN_MODULES:
                        hits.append({"file": relative, "kind": "import", "detail": alias.name})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in FORBIDDEN_MODULES:
                    hits.append({"file": relative, "kind": "import", "detail": module})
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in FORBIDDEN_CALLS:
                    hits.append({"file": relative, "kind": "call", "detail": name})
    return {
        "root": "src/lazybull/v2/reporting",
        "files": len(files),
        "forbidden_modules": sorted(FORBIDDEN_MODULES),
        "forbidden_calls": sorted(FORBIDDEN_CALLS),
        "hits": hits,
        "passed": bool(files) and not hits,
    }


def zero_write(smokes: Mapping[str, Any]) -> dict[str, Any]:
    """三条真实 smoke 的写域汇总（白名单外零差异 + 输入根全等 + 运行期守卫）。"""
    rows: dict[str, Any] = {}
    for name, record in smokes.items():
        if not isinstance(record, Mapping) or "audit" not in record:
            rows[name] = {"passed": False, "detail": "smoke 未产出审计证据"}
            continue
        audit = dict(record.get("audit") or {})
        driver = dict(record.get("driver_audit") or {})
        inputs = dict(record.get("input_evidence") or {})
        rows[name] = {
            "product_audit_verdict": audit.get("verdict"),
            "product_audit_uncovered": audit.get("uncovered"),
            "product_audit_reparse_checked": audit.get("reparse_checked"),
            "driver_audit_verdict": driver.get("verdict"),
            "driver_audit_changed": driver.get("changed"),
            "inputs_unchanged": inputs.get("unchanged"),
            "input_roots": inputs.get("roots"),
            "passed": bool(
                audit.get("verdict") == "pass"
                and driver.get("verdict") == "pass"
                and inputs.get("unchanged") is True
            ),
        }
    return {
        "schema": "p6_t5_zero_write/v1",
        "smokes": rows,
        "scan_scope": {
            "includes": list(publish.AUDIT_SCAN_INCLUDES),
            "excludes": list(publish.AUDIT_SCAN_EXCLUDES),
            "note": (
                "产品审计按设计 §2.5 为**有界子树**哈希：扫描域内全树零差异，排除树"
                "（data/models、data/raw、data/features、data/paper_shadow、temp 等）"
                "由运行期只读守卫承担；真实输入根（data/paper、data/runs、campaign 目录）"
                "另由驱动侧 before/after 全树 path+SHA+size 比对，跨根写入必判 foreign_root_paths"
            ),
        },
        "runtime_guard": {
            "deny_network": True,
            "deny_subprocess": True,
            "source": "scripts/p6_report.py::_guard（ReadonlyGuard，CPython 审计钩子执法）",
            # 守卫只在带 --audit-out 的命令上生效：不得让读者以为整轮 15 次生成都被守卫
            "guarded_commands": [
                "三条真实 smoke（p4/runs/p5b）的 generate --audit-out",
                "真实 golden 核验器（check_real_golden.py 自装 ReadonlyGuard）",
                "runs smoke 的异构根负例（--no-publish）",
            ],
            "unguarded_commands": [
                "三面 10 次计时生成与三面复算（--reports-root 指向 temp，无 --audit-out）",
                "linux_package_selftest（内部起 Python 子进程跑 replay，输出亦在 temp/）",
                "定向 pytest、复杂度 hook（只读检查器）",
            ],
            "unguarded_evidence": (
                "无守卫窗口的诚实边界：①`static_network_scan` 只覆盖 `src/lazybull/v2/reporting/**`"
                "（44 文件 0 命中，不联网/不起子进程）；②`scripts/p6_report.py` 在守卫安装**之前**会经 "
                "`git_record()` 执行 2 条**只读** git 子进程（rev-parse/status，`.git/**` 属审计排除树），"
                "因此“整轮不起子进程”不成立，只能说“报告生成逻辑不联网、守卫窗口内禁子进程”；"
                "③上述无守卫命令的输出根均在 `temp/`（一次性工作树，不属任何产品根、也不在 "
                "`scan_audit_roots` 的 includes 内，故**不受** global_scope 覆盖）；产品根零写由三条"
                "真实 smoke 的审计窗口 + 输入根 before/after 全树比对证明"
            ),
            "note": "守卫拦截网络/子进程/白名单外写入；命令 exit=0 即未触发任何拦截",
        },
        "static_network_scan": _static_network_scan(),
        "no_ensure_or_repair": (
            "只读命令不调用 ensure/runner/state_service/recovery；守卫窗口内子进程被禁用，"
            "P4 根零写由输入根 before/after 全树比对证明"
        ),
        "passed": bool(rows) and all(item.get("passed") for item in rows.values()),
    }


# ---------------------------------------------------------------------------
# 真实 golden（非 pytest）
# ---------------------------------------------------------------------------


def real_golden(validation_id: str) -> dict[str, Any]:
    """调用 ``check_real_golden.py`` 核验真实 P5b 三族的路径/SHA/seal/链/值。"""
    out = _sibling(validation_id, "_golden")
    command = [
        sys.executable,
        str(GOLDEN_SCRIPT),
        "--root",
        REAL_BUNDLE_ROOT.as_posix(),
        "--out",
        out.relative_to(ROOT).as_posix(),
        "--freeze",
        REAL_GOLDEN_FREEZE.as_posix(),
    ]
    input_before = _tree_digest({"campaigns": ROOT / REAL_BUNDLE_ROOT})
    result = _run(command)
    input_after = _tree_digest({"campaigns": ROOT / REAL_BUNDLE_ROOT})
    payload = _read_json(out / "real_golden.json") or {}
    families = dict(payload.get("families") or {})
    checks = [
        _check("exit_code_zero", result["exit_code"] == 0, f"exit={result['exit_code']}"),
        _check("payload_verdict", payload.get("verdict") == "pass", str(payload.get("verdict"))),
        _check(
            "all_families_pass",
            bool(families) and all(row.get("verdict") == "pass" for row in families.values()),
            f"families={sorted(families)}",
        ),
        _check(
            "family_count_three",
            len(families) == 3,
            f"count={len(families)}（seq10/7/21 三族）",
        ),
        _check(
            "inputs_unchanged",
            input_before["digest"] == input_after["digest"],
            f"files={input_before['files']} bytes={input_before['bytes']}",
        ),
        _check(
            "checker_not_pytest",
            not _source_skips(GOLDEN_SCRIPT),
            "核验器源码无 pytest/skipif/xfail（非“命令里没写”这种同义反复）",
        ),
    ]
    return {
        "label": "real_golden_seq10_seq7_seq21",
        "command": _posix_command(_relativize(command)),
        "command_has_no_skip_flags": _has_no_skip_flags(command),
        "exit_code": result["exit_code"],
        "stderr_tail": result["stderr_tail"],
        "out": out.relative_to(ROOT).as_posix(),
        "freeze": REAL_GOLDEN_FREEZE.as_posix(),
        "freeze_sha256": (
            validation.sha256_file(ROOT / REAL_GOLDEN_FREEZE)
            if (ROOT / REAL_GOLDEN_FREEZE).is_file()
            else None
        ),
        "verdict": payload.get("verdict"),
        "families": {
            name: {
                "verdict": row.get("verdict"),
                "checks": row.get("checks"),
                "campaign_dir": row.get("campaign_dir"),
                "frozen_file_count": row.get("frozen_file_count"),
                "failures": row.get("failures"),
            }
            for name, row in families.items()
        },
        "input_evidence": {
            "digest_before": input_before["digest"],
            "digest_after": input_after["digest"],
            "files": input_before["files"],
            "unchanged": input_before["digest"] == input_after["digest"],
        },
        "checks": checks,
        "passed": _all_passed(checks),
    }


# ---------------------------------------------------------------------------
# 三面（dashboard / experiment / trend）× 标准/压力：生成 + 打开 + 体积
# ---------------------------------------------------------------------------


def _face_publish_args(kind: str, profile: str, paths: Mapping[str, Any]) -> list[str]:
    """三面的 generate 参数（reports-root / attempt 由调用方补，保证复算命令同构）。"""
    if kind == "dashboard":
        return [
            "--kind",
            "dashboard",
            "--evidence-class",
            "research",
            "--paper-root",
            _rel_text(paths["paper"]),
            "--quality-root",
            _rel_text(paths["quality"]),
            "--generated-at",
            FACE_GENERATED_AT.get(kind, WORKFLOW_GENERATED_AT),
        ]
    if kind == "experiment":
        return [
            "--kind",
            "experiment",
            "--evidence-class",
            "research",
            "--arm",
            f"base=runs:{_rel_text(paths['base'])}",
            "--arm",
            f"arm=runs:{_rel_text(paths['arm'])}",
            "--baseline",
            "base",
            "--generated-at",
            FACE_GENERATED_AT.get(kind, WORKFLOW_GENERATED_AT),
        ]
    return [
        "--kind",
        "trend",
        "--evidence-class",
        "research",
        "--source",
        f"runs:{_rel_text(paths['new_run'])}",
        "--previous",
        _rel_text(paths["previous"]),
        "--generated-at",
        FACE_GENERATED_AT.get(kind, WORKFLOW_GENERATED_AT),
    ]


def _face_publish(
    kind: str, paths: Mapping[str, Any], *, reports_root: Path, attempt: str
) -> dict[str, Any]:
    """按面发布一次（temp 根；不带审计——三面性能不承担真实 smoke 的写域证据）。"""
    command = [
        sys.executable,
        str(REPORT_SCRIPT),
        "generate",
        *_face_publish_args(kind, "", paths),
        "--reports-root",
        _rel_text(reports_root),
        "--attempt-id",
        attempt,
        "--json",
    ]
    started = time.perf_counter()
    result = _run(command)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    payload = _last_json(result["stdout"]) if result["exit_code"] == 0 else {}
    run_dir = Path(str(payload["report_path"])) if payload.get("report_path") else None
    files = (
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(run_dir.iterdir())
            if path.is_file()
        }
        if run_dir and run_dir.is_dir()
        else {}
    )
    return {
        "command": _posix_command(_relativize(command)),
        "exit_code": result["exit_code"],
        "elapsed_ms": round(elapsed_ms, 1),
        "verdict": payload.get("verdict"),
        "report_id": payload.get("report_id"),
        "run_dir": run_dir.as_posix() if run_dir else None,
        "files": files,
        "file_bytes": dict(payload.get("file_bytes") or {}),
        "stderr_tail": result["stderr_tail"],
        "passed": result["exit_code"] == 0 and payload.get("verdict") == "pass" and bool(files),
    }


def _open_gate(run_dir: Path | None, validation_id: str, suffix: str) -> dict[str, Any]:
    """5 冷 + 5 热打开测量（真实 Chromium；原始 performance.json 落兄弟 validation 目录）。"""
    if run_dir is None or not (Path(run_dir) / "report.html").is_file():
        return {"passed": False, "detail": "无已发布 HTML"}
    out = _sibling(validation_id, f"_open_{suffix}")
    try:
        performance = measure_report.measure(Path(run_dir) / "report.html", out, repeat=5)
    except (measure_report.MeasureSetupError, OSError, RuntimeError) as exc:
        return {"passed": False, "detail": f"{type(exc).__name__}: {exc}"}
    gates = dict(performance.get("gates") or {})
    opened = dict(gates.get("open") or {})
    return {
        "performance_json": (out / "performance.json").relative_to(ROOT).as_posix(),
        "performance_sha256": validation.sha256_file(out / "performance.json"),
        "verdict": performance.get("verdict"),
        "report_bytes": (performance.get("html") or {}).get("bytes"),
        "size_gate": gates.get("size"),
        "open_gate": opened,
        "open_max_ms": opened.get("max_ms"),
        "cold_count": opened.get("cold_count"),
        "hot_count": opened.get("hot_count"),
        "threshold_ms": perf_gate.MAX_OPEN_MS,
        "requests_passed": (gates.get("requests") or {}).get("passed"),
        "environment_passed": (gates.get("environment") or {}).get("passed"),
        "versions_passed": (gates.get("acceptance_versions") or {}).get("passed"),
        "passed": performance.get("verdict") == "pass",
    }


def _generation_evidence(kind: str, profile: str, timed: Mapping[str, Any]) -> dict[str, Any]:
    """把某面某档的生成计时折成统一证据（原始记录另落 JSON）。

    ``all_succeeded`` **不取默认值**（评审 B3-2）：逐条记录显式要求 ``exit_code == 0``，
    T2 计时结构不带该键的旧形态一律判失败——避免“缺键即成功”。
    """
    gate = dict(timed.get("gate") or {})
    records = [dict(item) for item in (timed.get("records") or [])]
    if "all_succeeded" in timed:
        all_succeeded = bool(timed.get("all_succeeded"))
    else:
        all_succeeded = bool(records) and all(item.get("exit_code") == 0 for item in records)
    return {
        "profile": profile,
        "gate": gate,
        "max_ms": gate.get("max_ms"),
        "threshold_ms": perf_gate.MAX_GENERATION_MS,
        "byte_identical": bool(timed.get("byte_identical")),
        "all_succeeded": all_succeeded,
        "records": records,
        "gate_mode": timed.get("gate_mode"),
        "passed": bool(gate.get("passed") and timed.get("byte_identical") and all_succeeded),
    }


def _face_evidence(
    kind: str, profile: str, validation_id: str, out_dir: Path, work_root: Path
) -> dict[str, Any]:
    """单面单档：语料 → 5 次生成计时 → 发布一次 → 打开门 → 体积门 → 复算基准。"""
    face_work = work_root / f"{kind}_{profile}"
    face_work.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] | None = None
    if kind == "dashboard":
        case = build_t2_corpus.build_case(face_work / "corpus", profile)
        paths: dict[str, Any] = {
            "paper": Path(case["paper"]),
            "quality": Path(case["quality_root"]),
        }
        timed = run_t2_acceptance._timed_generation(case, face_work / "gen", profile)
        published_dir = Path(str(timed["records"][0]["report_path"])) if timed["records"] else None
        publish_record = {
            "report_id": (timed["records"][0].get("report_id") if timed["records"] else None),
            "run_dir": published_dir.as_posix() if published_dir else None,
            "file_bytes": dict(timed["records"][0].get("file_bytes") or {}),
            "files": dict(timed["records"][0].get("file_sha256") or {}),
            "note": "计时轮自身已发布（--reports-root 指向 temp），打开门复用该 HTML",
        }
    elif kind == "experiment":
        corpus_root = face_work / "corpus"
        build_t3_corpus.build(corpus_root, profile=profile)
        # 语料按档位分目录：``<corpus>/<profile>/runs/{base,arm}``（T3 构建器约定）
        profile_root = corpus_root / profile
        paths = {
            "base": profile_root / "runs" / "base",
            "arm": profile_root / "runs" / "arm",
        }
        timed = run_t3_acceptance.timed_generations(profile_root, profile)
        publish_record = _face_publish(
            kind,
            paths,
            reports_root=face_work / "reports",
            attempt=f"t5_pub_{kind}_{profile}",
        )
        published_dir = Path(publish_record["run_dir"]) if publish_record["run_dir"] else None
    else:
        roots = run_t4_acceptance._profile_roots(face_work, profile)
        previous = run_t4_acceptance.publish_previous(roots, profile)
        if not previous.get("passed"):
            return {"profile": profile, "passed": False, "previous": previous}
        build_t4_corpus.build(roots["corpus"], profile)
        timed = run_t4_acceptance.timed_incremental(roots, Path(str(previous["run_dir"])), profile)
        new_run = run_t4_acceptance._new_run_dir(roots["corpus"], profile)
        paths = {"new_run": new_run, "previous": Path(str(previous["run_dir"]))}
        landed = dict(timed.get("landed") or {})
        published_dir = Path(str(landed["dir"])) if landed.get("dir") else None
        publish_record = {
            "report_id": (timed["records"][0].get("report_id") if timed["records"] else None),
            "run_dir": landed.get("dir"),
            "matches_declared": landed.get("matches_declared"),
            "note": "增量落地轮（--previous + 单新 run，reports-root=temp）",
        }
    opened = _open_gate(published_dir, validation_id, f"{kind}_{profile}")
    generation = _generation_evidence(kind, profile, timed)
    files = (
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(published_dir.iterdir())
            if path.is_file()
        }
        if published_dir and published_dir.is_dir()
        else {}
    )
    size_bytes = (
        (published_dir / "report.html").stat().st_size
        if published_dir and (published_dir / "report.html").is_file()
        else None
    )
    size_gate = (
        perf_gate.judge_size(size_bytes) if isinstance(size_bytes, int) else {"passed": False}
    )
    raw_path = out_dir / "generation" / f"{kind}_{profile}.json"
    validation.write_json_new(
        raw_path,
        {
            "schema": "p6_t5_generation_raw/v1",
            "kind": kind,
            "profile": profile,
            "generated_at": FACE_GENERATED_AT.get(kind, WORKFLOW_GENERATED_AT),
            "records": generation["records"],
            "gate": generation["gate"],
        },
    )
    records = list(generation.get("records") or [])
    timed_passed = timed.get("passed")
    generation_ok = (
        bool(generation.get("passed"))
        and bool(generation.get("all_succeeded"))
        and timed_passed is not False  # 下层若显式给出 passed，必须为真（不得丢弃）
        and len(records) == int(contracts.THRESHOLDS["generation_repeats"])
    )
    checks = [
        _check(
            "generation_gate",
            bool(generation["gate"].get("passed")),
            f"max={generation['max_ms']}ms",
        ),
        _check(
            "generation_byte_identical",
            generation["byte_identical"],
            "同输入同 generated_at 逐次字节一致",
        ),
        # 评审 R3-T5-04：失败执行的短耗时不得被当成合格性能——逐次成功 + 轮数精确 + 上层传递
        _check(
            "generation_all_succeeded",
            bool(generation.get("all_succeeded")),
            f"exit_codes={[item.get('exit_code') for item in records]}",
        ),
        _check(
            "generation_repeat_protocol",
            len(records) == int(contracts.THRESHOLDS["generation_repeats"]),
            f"records={len(records)} expected={contracts.THRESHOLDS['generation_repeats']}",
        ),
        _check("generation_ok", generation_ok, f"timed_passed={timed_passed}"),
        _check("size_gate", bool(size_gate.get("passed")), f"bytes={size_bytes}"),
        _check("open_gate", bool(opened.get("passed")), f"max={opened.get('open_max_ms')}ms"),
        _check("published_present", published_dir is not None and bool(files), str(published_dir)),
    ]
    return {
        "profile": profile,
        "kind": kind,
        "report_id": publish_record.get("report_id"),
        "work_root": face_work.as_posix(),
        "paths": {name: _rel_text(value) for name, value in paths.items()},
        "previous": previous,
        "generation": generation,
        "generation_raw": raw_path.relative_to(ROOT).as_posix(),
        "open": opened,
        "size_bytes": size_bytes,
        "size_gate": size_gate,
        "published": publish_record,
        "files": files,
        "checks": checks,
        "reproduce_args": _face_publish_args(kind, profile, paths),
        "reports_root": (face_work / "reports").as_posix(),
        "passed": _all_passed(checks),
    }


def three_faces(validation_id: str, out_dir: Path, work_root: Path) -> dict[str, Any]:
    """三面 × 两档的完整证据（任一面任一档失败即整项失败）。"""
    faces: dict[str, Any] = {}
    for kind in KINDS:
        faces[kind] = {}
        for profile in PROFILES:
            faces[kind][profile] = _guarded(
                f"{kind}_{profile}",
                lambda k=kind, p=profile: _face_evidence(k, p, validation_id, out_dir, work_root),
            )
    rows = [item for kind in faces.values() for item in kind.values()]
    return {
        "faces": faces,
        "row_count": len(rows),
        "passed": len(rows) == len(KINDS) * len(PROFILES)
        and all(bool(item.get("passed")) for item in rows),
    }


# ---------------------------------------------------------------------------
# 复算（全链重建 + dashboard 包内重建）
# ---------------------------------------------------------------------------


def _file_hashes(directory: Path | None) -> dict[str, str]:
    """目录内单层文件 SHA-256（证据侧独立回读，不信子进程自报）。"""
    if directory is None or not directory.is_dir():
        return {}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _canonical_of(files: Mapping[str, str]) -> str:
    """文件哈希集合的 canonical 摘要（跨平台比较用；不含绝对路径）。"""
    return hashlib.sha256(
        json.dumps(dict(files), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _package_internal_rebuild(run_dir: Path) -> dict[str, Any]:
    """包内重建**诊断项**（设计 §8.3 层 2，dashboard）：projection + manifest 生成器 → HTML。

    诚实边界（T5 实测）：footer 以 ``dict`` repr 渲染 ``generator_files``，其**插入序**在
    canonical JSON（``sort_keys``）往返后不可恢复，故包内重建字节不等。本项只作诊断登记，
    **不参与** ``verdict``；复算判据是层 3（同输入同 ``generated_at`` 全链重建逐字节一致）。
    """
    projection = _read_json(run_dir / "projection.json")
    record = _read_json(run_dir / "report_manifest.json")
    html_path = run_dir / "report.html"
    if (
        not isinstance(projection, Mapping)
        or not isinstance(record, Mapping)
        or not html_path.is_file()
    ):
        return {
            "passed": False,
            "detail": "缺 projection.json / report_manifest.json / report.html",
        }
    try:
        view = dashboard_view.build_dashboard_view(
            dict(projection),
            report_id=str(record.get("report_id")),
            generated_at=str(record.get("generated_at")),
            input_evidence_classes=tuple(record.get("input_evidence_classes") or ()),
            generator=dict(record.get("generator") or {}),
        )
        rebuilt = render.render_dashboard_bytes(view)
    except (KeyError, TypeError, ValueError) as exc:
        return {"passed": False, "detail": f"{type(exc).__name__}: {exc}"}
    actual = html_path.read_bytes()
    published_lines = actual.decode("utf-8").splitlines()
    rebuilt_lines = rebuilt.decode("utf-8").splitlines()
    differing = [
        (left, right) for left, right in zip(published_lines, rebuilt_lines) if left != right
    ]
    differing_all_generator = bool(differing) and all(
        "generator" in side for pair in differing for side in pair
    )
    return {
        "rebuilt_sha256": hashlib.sha256(rebuilt).hexdigest(),
        "published_sha256": hashlib.sha256(actual).hexdigest(),
        "bytes_equal": rebuilt == actual,
        "line_count_equal": len(published_lines) == len(rebuilt_lines),
        "differing_lines": len(differing),
        "differing_lines_all_generator_footer": differing_all_generator,
        "manifest_seal": manifest_module.verify_manifest(dict(record)).get("report_status"),
    }


def reproduce_faces(
    validation_id: str, faces: Mapping[str, Any], work_root: Path
) -> dict[str, Any]:
    """三面（standard 档）同输入同 ``generated_at`` 的全链重建逐字节一致。"""
    rows: dict[str, Any] = {}
    for kind in KINDS:
        source = dict((faces.get(kind) or {}).get("standard") or {})
        if not source.get("passed"):
            rows[kind] = {"kind": kind, "passed": False, "detail": "基准档未通过，无法复算"}
            continue
        target = work_root / f"reproduce_{kind}"
        command = [
            sys.executable,
            str(REPORT_SCRIPT),
            "generate",
            *[str(item) for item in source.get("reproduce_args") or []],
            "--reports-root",
            _rel_text(target),
            "--attempt-id",
            f"t5_repro_{kind}",
            "--json",
        ]
        started = time.perf_counter()
        result = _run(command)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload = _last_json(result["stdout"]) if result["exit_code"] == 0 else {}
        rebuilt_dir = Path(str(payload["report_path"])) if payload.get("report_path") else None
        rebuilt = _file_hashes(rebuilt_dir)
        original = dict(source.get("files") or {})
        extra: dict[str, Any] = {}
        if kind == "dashboard" and rebuilt_dir is not None:
            extra["package_internal_probe"] = _package_internal_rebuild(rebuilt_dir)
        checks = [
            _check("exit_code_zero", result["exit_code"] == 0, f"exit={result['exit_code']}"),
            _check(
                "payload_verdict", payload.get("verdict") == "pass", str(payload.get("verdict"))
            ),
            _check(
                "report_id_identical",
                payload.get("report_id") is not None
                and payload.get("report_id") == source.get("report_id"),
                f"report_id={payload.get('report_id')}",
            ),
            _check(
                "files_byte_identical",
                bool(original) and rebuilt == original,
                f"files={sorted(rebuilt)}",
            ),
            _check(
                "file_count_matches",
                len(rebuilt) == len(original) and set(rebuilt) == set(original),
                f"rebuilt={len(rebuilt)} original={len(original)}",
            ),
        ]
        # 层 2（包内重建）只作诊断：footprint 见 ``_package_internal_rebuild`` docstring；
        # 复算判定只认层 3（全链重建）——不得把诊断项伪装成通过项
        probe = dict(extra.get("package_internal_probe") or {})
        if probe:
            probe["normalization_note"] = (
                "footer 的 generator_files 以 dict repr 渲染；canonical JSON 往返丢失插入序，"
                "故包内重建字节不等（诊断登记，不参与 verdict）"
            )
        rows[kind] = {
            "kind": kind,
            "command": _posix_command(_relativize(command)),
            "exit_code": result["exit_code"],
            "elapsed_ms": round(elapsed_ms, 1),
            "report_id": payload.get("report_id"),
            "rebuilt_dir": rebuilt_dir.as_posix() if rebuilt_dir else None,
            "files": rebuilt,
            "canonical_sha256": _canonical_of(rebuilt) if rebuilt else None,
            "original_canonical_sha256": _canonical_of(original) if original else None,
            "checks": checks,
            "stderr_tail": result["stderr_tail"],
            "package_internal_probe": probe or None,
            "passed": _all_passed(checks),
        }
    return {
        "generated_at_by_kind": dict(FACE_GENERATED_AT),
        "rows": rows,
        "passed": bool(rows) and all(bool(item.get("passed")) for item in rows.values()),
    }


# ---------------------------------------------------------------------------
# 双平台
# ---------------------------------------------------------------------------

_LINUX_PROBE_SCRIPT = (
    'echo "PY=$(python3 -V 2>&1)"; '
    'python3 -c "import pandas,numpy,jinja2;print(\\"DEPS=ok\\")" '
    '2>&1 | tail -1 | sed "s/^/DEPSLINE=/"; '
    'command -v pip3 >/dev/null && echo "PIP=yes" || echo "PIP=no"; '
    'ls /usr/bin/python3.* 2>/dev/null | tr "\\n" " " | sed "s/^/PYS=/"; '
    'uname -r | sed "s/^/KERNEL=/"'
)


#: 目标解释器自证脚本（评审 R3-T5-03）：平台身份 + Python 版本 + 关键依赖版本
_LINUX_IDENTITY_CODE = (
    "import json,sys,importlib.util;"
    "print(json.dumps({'platform':sys.platform,'python':sys.version.split()[0],"
    "'modules':{n:(getattr(importlib.import_module(n),'__version__',None) "
    "if importlib.util.find_spec(n) else None) for n in ('jinja2','pandas','numpy')}}))"
)


def _linux_identity(python: str, cwd: str) -> dict[str, Any]:
    """让**目标解释器**自证身份（平台/版本/依赖），不采信“参数传入即 Linux”。"""
    result = _run([python, "-c", _LINUX_IDENTITY_CODE], cwd=Path(cwd))
    payload = _last_json(result["stdout"]) or {}
    return {
        "exit_code": result["exit_code"],
        "platform": payload.get("platform"),
        "python": payload.get("python"),
        "modules": dict(payload.get("modules") or {}),
        "stderr_tail": result["stderr_tail"],
    }


def _parse_probe(text: str) -> dict[str, str]:
    """把探测脚本输出解析成键值（未出现的键不伪造）。"""
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.strip().partition("=")
        if key and value:
            values[key] = value.strip()
    if "DEPSLINE" in values:
        values["deps_line"] = values["DEPSLINE"]
    return values


def _linux_probe() -> dict[str, Any]:
    """探测本机可用的 Linux 执行环境（WSL）；只登记事实，不安装任何依赖。"""
    info: dict[str, Any] = {"host_platform": platform.system(), "host_release": platform.release()}
    if platform.system() == "Linux":
        result = _run(["bash", "-lc", _LINUX_PROBE_SCRIPT], timeout=120)
        parsed = _parse_probe(result["stdout"])
        return {
            **info,
            "available": True,
            "executed_here": True,
            "distro": "native",
            "probe": parsed,
            "probe_exit": result["exit_code"],
        }
    executable = shutil.which("wsl")
    info["wsl_executable"] = executable
    if not executable:
        return {**info, "available": False, "reason": "本机无 wsl 可执行文件", "probe": {}}
    listing = _run([executable, "-l", "-q"], timeout=120)
    distros = [
        line.strip().replace("\x00", "")
        for line in listing["stdout"].splitlines()
        if line.strip().replace("\x00", "").strip()
    ]
    info["distros"] = distros
    if not distros:
        return {**info, "available": False, "reason": "WSL 无已安装发行版", "probe": {}}
    distro = distros[0]
    result = _run([executable, "-d", distro, "--", "bash", "-lc", _LINUX_PROBE_SCRIPT], timeout=180)
    parsed = _parse_probe(result["stdout"])
    deps_ok = "ok" in str(parsed.get("deps_line", "")).lower()
    python_text = str(parsed.get("PY", ""))
    version_ok = bool(re.search(r"3\.(1[2-9]|[2-9]\d)", python_text))
    info.update(
        {
            "distro": distro,
            "probe": parsed,
            "probe_exit": result["exit_code"],
            "python_312_plus": version_ok,
            "dependencies_ok": deps_ok,
            "pip_available": str(parsed.get("PIP", "")).lower() == "yes",
        }
    )
    info["available"] = bool(version_ok and deps_ok)
    if not info["available"]:
        missing = []
        if not version_ok:
            missing.append(f"Python≥3.12（实测 {python_text or '未知'}）")
        if not deps_ok:
            missing.append("运行时依赖 pandas/numpy/jinja2")
        if str(parsed.get("PIP", "")).lower() != "yes":
            missing.append("pip（无法在不联网/不装包授权下构建 3.12 环境）")
        info["reason"] = "Linux 环境不可执行：" + "、".join(missing)
    return info


def _linux_commands(faces: Mapping[str, Any], validation_id: str) -> list[str]:
    """Linux 侧复算准备包（**单命令自包含**）。

    语料与前序快照都在 ``temp/`` 且不入包 ⇒ 跨机必须重建；``run_t5_linux_replay.py``
    负责“重建 T2/T3/T4 语料 + trend 上一快照 → 按同一协议复算三面 → 与 Windows 载荷
    逐文件 SHA 与 canonical 摘要比对”，因此准备包只有一条可执行命令（另加环境前置）。
    ``faces`` 只用于在载荷里留一条“Windows 侧协议来源”的追溯（复算参数由脚本按同一
    实现重建，不做字符串改写）。
    """
    del faces  # 协议来源：run_t5_acceptance._face_publish_args（同一实现，避免字符串改写漂移）
    return [
        "cd <repo_root>",
        "python -V  # 需 ≥3.12；依赖版本须与 Windows 侧一致（jinja2 3.1.6 / playwright 1.64.1）",
        "python scripts/v2_p6/run_t5_linux_replay.py "
        f"--windows data/reports/p6/validations/research/{validation_id}/t5_acceptance.json "
        "--json  # 期望 verdict=pass（三面 MATCH）；脚本按载荷记录的相对路径重建语料，"
        "目标工作树已存在即拒绝",
        "python scripts/v2_p6/selftest_t5_linux_replay.py --json  # 本机自检同一条闭环（可选）",
    ]


def linux_package_selftest(work_root: Path) -> dict[str, Any]:
    """跨平台复算包的**本机自检**（独立前缀闭环：参照 → 删树 → 按记录路径重建 → 三面比对）。

    真实 Linux 未达成时，这是“准备包不是纸面命令”的机器证据；代价约 60~90s。
    """
    work = work_root / "linux_selftest"
    command = [
        sys.executable,
        "-X",
        "utf8",
        "scripts/v2_p6/selftest_t5_linux_replay.py",
        "--work",
        _rel_text(work),
        "--json",
    ]
    result = _run(command)
    payload = _last_json(result["stdout"]) if result["exit_code"] == 0 else {}
    return {
        "command": _posix_command(_relativize(command)),
        "exit_code": result["exit_code"],
        "verdict": payload.get("verdict"),
        "failed_kinds": list(payload.get("failed_kinds") or []),
        "kinds": dict(payload.get("kinds") or {}),
        "reference_ids": dict(payload.get("reference_ids") or {}),
        "stderr_tail": result["stderr_tail"] or list(payload.get("stderr_tail") or []),
        "passed": bool(result["exit_code"] == 0 and payload.get("verdict") == "pass"),
    }


def dual_platform(
    validation_id: str,
    faces: Mapping[str, Any],
    reproduce: Mapping[str, Any],
    *,
    work_root: Path | None = None,
    linux_python: str | None = None,
    linux_cwd: str | None = None,
    windows_evidence: Path | None = None,
) -> dict[str, Any]:
    """Windows 侧 canonical 哈希 + Linux 侧执行/未达成登记（不得以“跳过”代替）。"""
    windows = {
        kind: {
            "canonical_sha256": dict((reproduce.get("rows") or {}).get(kind) or {}).get(
                "canonical_sha256"
            ),
            "files": dict((reproduce.get("rows") or {}).get(kind) or {}).get("files") or {},
        }
        for kind in KINDS
    }
    windows_ok = all(item.get("canonical_sha256") for item in windows.values())
    probe = _linux_probe()
    linux: dict[str, Any]
    if platform.system() == "Linux" and windows_evidence is not None:
        payload = _read_json(windows_evidence) or {}
        block = dict((payload.get("checks") or {}).get("dual_platform") or {})
        reference = dict(block.get("windows") or {})
        comparisons = {}
        for kind in KINDS:
            expected = dict((reference.get(kind) or {}).get("files") or {})
            actual = dict(windows.get(kind, {}).get("files") or {})
            comparisons[kind] = {
                "equal": bool(expected) and expected == actual,
                "expected_canonical": (reference.get(kind) or {}).get("canonical_sha256"),
                "actual_canonical": windows.get(kind, {}).get("canonical_sha256"),
            }
        linux = {
            "status": "executed",
            "probe": probe,
            "comparison": comparisons,
            "passed": all(item["equal"] for item in comparisons.values()),
        }
    elif linux_python and linux_cwd:
        identity = _linux_identity(linux_python, linux_cwd)
        platform_name = str(identity.get("platform") or "")
        modules = dict(identity.get("modules") or {})
        python_text = str(identity.get("python") or "")
        version_ok = bool(re.search(r"3\.(1[2-9]|[2-9]\d)", python_text))
        deps_ok = all(modules.get(name) for name in ("jinja2", "pandas", "numpy"))
        if not platform_name.startswith("linux"):
            return {
                "windows": windows,
                "linux": {
                    "status": "rejected_not_linux",
                    "python": linux_python,
                    "cwd": "<linux_repo_root>",
                    "identity": identity,
                    "reason": (
                        f"目标解释器自证平台为 {platform_name or '未知'}（非 Linux），"
                        "拒绝登记为跨平台已执行"
                    ),
                    "passed": False,
                },
                "linux_package_selftest": {},
                "checks": [
                    _check("windows_canonical_present", windows_ok, f"kinds={sorted(windows)}"),
                    _check(
                        "windows_reproduce_identical",
                        bool(reproduce.get("passed")),
                        "三面全链复算逐字节一致（Windows 侧）",
                    ),
                    _check("linux_identity_verified", False, f"platform={platform_name}"),
                ],
                "achieved": False,
                "unmet": ["linux_dual_platform"],
                "passed": False,
            }
        if not (version_ok and deps_ok):
            return {
                "windows": windows,
                "linux": {
                    "status": "rejected_identity",
                    "python": linux_python,
                    "identity": identity,
                    "reason": (
                        f"目标解释器 Python={python_text or '未知'}（需≥3.12）、"
                        f"依赖={modules}（需 jinja2/pandas/numpy 全部就位）"
                    ),
                    "passed": False,
                },
                "linux_package_selftest": {},
                "checks": [
                    _check("windows_canonical_present", windows_ok, f"kinds={sorted(windows)}"),
                    _check(
                        "windows_reproduce_identical",
                        bool(reproduce.get("passed")),
                        "三面全链复算逐字节一致（Windows 侧）",
                    ),
                    _check("linux_identity_verified", False, f"python={python_text}"),
                ],
                "achieved": False,
                "unmet": ["linux_dual_platform"],
                "passed": False,
            }
        rows: dict[str, Any] = {}
        for kind in KINDS:
            source = dict((faces.get(kind) or {}).get("standard") or {})
            command = [
                linux_python,
                "scripts/p6_report.py",
                "generate",
                *[str(item) for item in source.get("reproduce_args") or []],
                "--reports-root",
                f"temp/p6_t5_linux/{kind}",
                "--attempt-id",
                f"linux_{kind}",
                "--json",
            ]
            result = _run(command, cwd=Path(linux_cwd))
            payload = _last_json(result["stdout"]) if result["exit_code"] == 0 else {}
            rebuilt = (
                _file_hashes(Path(str(payload["report_path"])))
                if payload.get("report_path")
                else {}
            )
            expected = dict((windows.get(kind) or {}).get("files") or {})
            rows[kind] = {
                "exit_code": result["exit_code"],
                "files": rebuilt,
                "equal": bool(expected) and rebuilt == expected,
                "stderr_tail": result["stderr_tail"],
            }
        linux = {
            "status": "executed",
            "python": linux_python,
            "cwd": "<linux_repo_root>",
            "rows": rows,
            "passed": all(item["equal"] for item in rows.values()),
        }
    else:
        linux = {
            "status": "not_achieved",
            "probe": probe,
            "reason": (
                "真实验收机无满足条件的 Linux 运行环境（见 probe：Python 版本/依赖/pip），"
                "且本任务未获装依赖或联网授权 ⇒ 不得以“跳过”代替；准备包见 linux_commands"
            ),
            "linux_commands": _linux_commands(faces, validation_id),
            "prepared": True,
            "passed": None,
        }
    checks = [
        _check("windows_canonical_present", windows_ok, f"kinds={sorted(windows)}"),
        _check(
            "windows_reproduce_identical",
            bool((reproduce.get("passed"))),
            "三面全链复算逐字节一致（Windows 侧）",
        ),
        _check(
            "linux_status_recorded",
            linux.get("status") in ("executed", "not_achieved"),
            str(linux.get("status")),
        ),
    ]
    if linux.get("status") == "executed":
        checks.append(_check("linux_equal_to_windows", bool(linux.get("passed")), "跨平台哈希全等"))
    selftest: dict[str, Any] = {}
    if linux.get("status") != "executed" and work_root is not None:
        selftest = linux_package_selftest(work_root)
        checks.append(
            _check(
                "linux_package_selftest",
                bool(selftest.get("passed")),
                f"本机自检 verdict={selftest.get('verdict')} "
                f"failed={selftest.get('failed_kinds')}",
            )
        )
    achieved = bool(linux.get("status") == "executed" and linux.get("passed"))
    return {
        "windows": windows,
        "linux": linux,
        "linux_package_selftest": selftest,
        # ``passed`` = 本项检查**自身**无缺陷（状态如实登记、自检可执行）；
        # ``achieved`` = 真实跨平台门是否**达成**（评审 R3-T5-02/03：二者不得混用）
        "achieved": achieved,
        "checks": checks,
        "unmet": [] if achieved else ["linux_dual_platform"],
        "passed": _all_passed(checks),
    }


# ---------------------------------------------------------------------------
# 历史不覆盖（由 smoke 前后整树快照推导）与文档同步
# ---------------------------------------------------------------------------


def _report_prefixes(record: Mapping[str, Any]) -> list[str]:
    """一次 smoke 允许新增的**本次产物**前缀：新报告 run 目录 + 本次审计目录。"""
    prefixes: list[str] = []
    path = record.get("report_path")
    if path:
        try:
            prefixes.append(Path(str(path)).relative_to(ROOT).as_posix())
        except ValueError:
            prefixes.append(str(path))
    audit = (record.get("audit") or {}).get("path")
    if audit:
        prefixes.append(Path(str(audit)).parent.as_posix())
    return prefixes


def _produced_prefixes(checks: Mapping[str, Any], validation_id: str) -> list[str]:
    """整轮允许新增的**精确目录**集合（不用目录通配，避免把兄弟目录一起放行）。

    来源：①本次三类 validation 目录；②真实 golden 输出目录；③每条 smoke 的新报告 run 与审计目录；
    ④三面打开测量的 ``_open_*`` 目录（``performance.json`` 所在目录）。
    """
    prefixes: list[str] = [
        f"{contracts.REPORTS_ROOT}/validations/research/{validation_id}",
        f"{contracts.REPORTS_ROOT}/validations/production/{validation_id}",
        f"{contracts.REPORTS_ROOT}/validations/calibration_only/{validation_id}",
    ]
    golden = dict(checks.get("real_golden") or {})
    if golden.get("out"):
        prefixes.append(str(golden["out"]))
    smokes = dict((checks.get("real_smoke") or {}).get("smokes") or {})
    for record in smokes.values():
        prefixes.extend(_report_prefixes(record))
    faces = dict((checks.get("three_faces") or {}).get("faces") or {})
    for rows in faces.values():
        for row in (rows or {}).values():
            if not isinstance(row, Mapping):
                continue
            perf = (row.get("open") or {}).get("performance_json")
            if perf:
                prefixes.append(Path(str(perf)).parent.as_posix())
    return sorted({prefix.rstrip("/") for prefix in prefixes if prefix})


def historical(smokes: Mapping[str, Any], prefixes: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """历史/增量不覆盖：既有文件零改写零删除，新增文件仅落在本次允许集合。"""
    rows: dict[str, Any] = {}
    for name, record in smokes.items():
        if not isinstance(record, Mapping) or "_before_tree" not in record:
            rows[name] = {"passed": False, "detail": "无前后快照"}
            continue
        allowed = [str(item) for item in prefixes.get(name, ())]
        allowed.extend(_report_prefixes(record))
        rows[name] = _history_of(record, allowed)
    return {
        "schema": "p6_t5_history/v1",
        "smokes": rows,
        "note": (
            "快照为产品审计的扫描域（data/reports 全树 + data/paper + data/runs + src/scripts/"
            "tests/docs/configs 等），故既有历史 validation 与既有报告 run 的逐字节不变都被覆盖"
        ),
        "passed": bool(rows) and all(item.get("passed") for item in rows.values()),
    }


def documentation(validation_id: str = "") -> dict[str, Any]:
    """版本号与文档同步检查（CHANGELOG/README/CLAUDE/计划/闭环报告 + 证据轮次一致）。"""
    version = generate._project_version()
    files: dict[str, Any] = {}
    for relative in DOC_FILES:
        path = ROOT / relative
        files[relative] = {
            "exists": path.is_file(),
            # 取证时值：文档正文在取证后定稿（仅补充本轮实测数字），判定只认
            # “存在 + 版本号 + 章节标记”，不得据此声称文档字节被冻结
            "sha256_at_run_time": validation.sha256_file(path) if path.is_file() else None,
        }

    def _read(relative: str) -> str:
        path = ROOT / relative
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    changelog = _read("CHANGELOG.md")
    plan = _read(PLAN_DOC)
    readme = _read("README.md")
    claude = _read("CLAUDE.md")
    checks = [
        _check(
            "files_present",
            bool(files) and all(bool(row.get("exists")) for row in files.values()),
            f"{sum(1 for row in files.values() if row.get('exists'))}/{len(files)}",
        ),
        _check("version_read", bool(version) and version != "<unavailable>", version),
        _check(
            "changelog_has_version",
            version in changelog,
            f"CHANGELOG 含 {version}",
        ),
        _check(
            "plan_has_t5_record",
            "## 24." in plan and "T5" in plan,
            "计划含 §24 T5 执行记录",
        ),
        _check("readme_mentions_p6", "P6" in readme, "README 含 P6 使用文档"),
        _check("claude_has_p6_contract", "P6" in claude and "T5" in claude, "CLAUDE 含 P6/T5 契约"),
        _check(
            "closure_report_present",
            (ROOT / CLOSURE_DOC).is_file(),
            CLOSURE_DOC,
        ),
        _check(
            "plan_and_closure_name_this_run",
            bool(validation_id) and validation_id in plan and validation_id in _read(CLOSURE_DOC),
            f"文档须写明本次证据轮次 {validation_id or '<未传>'}（防“文档引用别的轮次”）",
        ),
    ]
    return {
        "project_version": version,
        "validation_id": validation_id or None,
        "files": files,
        "note": (
            "文档 sha 为**取证时**值；正文在取证后按本轮实测数字定稿（差量见闭合报告），"
            "判定口径为‘文件存在 + 版本号同步 + 章节标记齐全’"
        ),
        "checks": checks,
        "passed": _all_passed(checks),
    }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def _top_buckets(keys: Sequence[str], depth: int = 4) -> dict[str, int]:
    """把 ``tree:path`` 键按路径前 ``depth`` 段聚合计数（可机读的工作域画像）。"""
    buckets: dict[str, int] = {}
    for key in keys:
        parts = key.split(":", 1)[-1].split("/")
        bucket = "/".join(parts[:depth]) if len(parts) > depth else "/".join(parts)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return dict(sorted(buckets.items()))


def _global_scope(before: Any, after: Any, allowed_prefixes: Sequence[str]) -> dict[str, Any]:
    """整轮运行的工作树写域判定（扫描域内；排除树见 scan_scope 登记）。"""
    modified = sorted(
        key
        for key in before
        if key in after and before[key].get("sha256") != after[key].get("sha256")
    )
    removed = sorted(key for key in before if key not in after)
    added = sorted(key for key in after if key not in before)
    unexpected = sorted(
        key
        for key in added
        if not any(_within(key.split(":", 1)[-1], prefix) for prefix in allowed_prefixes)
    )
    return {
        "pre_existing_files": len(before),
        "modified": modified[:10],
        "removed": removed[:10],
        "added": len(added),
        # 新增按“目录前四段”聚合（评审 A-04）：让第三方可独立复核“全在允许前缀”，
        # 而不必信任本函数的判定
        "added_by_top": _top_buckets(added),
        "added_outside_allowed": unexpected[:10],
        "allowed_prefixes": list(allowed_prefixes),
        "passed": not modified and not removed and not unexpected,
    }


def _collect(
    validation_id: str,
    out_dir: Path,
    *,
    linux_python: str | None,
    linux_cwd: str | None,
    windows_evidence: Path | None,
) -> dict[str, Any]:
    """顺序执行全部检查（任何一步失败都留在载荷里，不中断后续取证）。"""
    work_root = ROOT / "temp" / f"p6_t5_acceptance_{validation_id}"
    work_root.mkdir(parents=True, exist_ok=True)
    allowed_prefixes = (
        f"{contracts.REPORTS_ROOT}/validations/research/{validation_id}",
        f"{contracts.REPORTS_ROOT}/validations/production/{validation_id}",
        f"{contracts.REPORTS_ROOT}/validations/calibration_only/{validation_id}",
    )
    global_before = publish.scan_audit_roots(repo_root=ROOT)
    checks: dict[str, Any] = {}
    checks["pytest"] = _guarded("pytest", run_tests)
    checks["complexity"] = _guarded("complexity", run_complexity)
    checks["real_golden"] = _guarded("real_golden", lambda: real_golden(validation_id))
    smokes: dict[str, Any] = {}
    for name, action in (
        ("p4", lambda: smoke_p4(validation_id)),
        ("runs", lambda: smoke_runs(validation_id)),
        ("p5b", lambda: smoke_p5b(validation_id)),
    ):
        smokes[name] = _guarded(f"smoke_{name}", action)
    checks["real_smoke"] = {
        "smokes": {
            name: {key: value for key, value in record.items() if not key.startswith("_")}
            for name, record in smokes.items()
        },
        "passed": bool(smokes) and all(bool(item.get("passed")) for item in smokes.values()),
    }
    checks["zero_write"] = zero_write(smokes)
    checks["historical"] = historical(
        smokes,
        {name: allowed_prefixes for name in smokes},
    )
    faces = _guarded("three_faces", lambda: three_faces(validation_id, out_dir, work_root))
    checks["three_faces"] = faces
    checks["reproduce"] = _guarded(
        "reproduce",
        lambda: reproduce_faces(validation_id, dict(faces.get("faces") or {}), work_root),
    )
    checks["dual_platform"] = _guarded(
        "dual_platform",
        lambda: dual_platform(
            validation_id,
            dict(faces.get("faces") or {}),
            dict(checks["reproduce"] or {}),
            work_root=work_root,
            linux_python=linux_python,
            linux_cwd=linux_cwd,
            windows_evidence=windows_evidence,
        ),
    )
    checks["documentation"] = _guarded("documentation", lambda: documentation(validation_id))
    global_after = publish.scan_audit_roots(repo_root=ROOT)
    checks["global_scope"] = _global_scope(
        global_before, global_after, _produced_prefixes(checks, validation_id)
    )
    return checks


_PASS_REQUIRED: tuple[str, ...] = (
    "pytest",
    "complexity",
    "real_golden",
    "real_smoke",
    "zero_write",
    "historical",
    "three_faces",
    "reproduce",
    "dual_platform",
    "documentation",
    "global_scope",
)


def _final_verdict(failed: Sequence[str], unmet: Sequence[str]) -> str:
    """总体验收判定（评审 R3-T5-02）：未达成必需项**不得**输出 ``pass``。

    ``fail`` = 有检查项不通过；``incomplete`` = 检查项全通过但存在未达成必需项；
    ``pass`` = 二者皆无。退出码：0 / 1 / 3（见 ``main``）。
    """
    if failed:
        return "fail"
    return "incomplete" if unmet else "pass"


def run(
    validation_id: str | None = None,
    *,
    linux_python: str | None = None,
    linux_cwd: str | None = None,
    windows_evidence: Path | None = None,
) -> dict[str, Any]:
    """执行一次完整 T5 验收并写 ``t5_acceptance.json``（拒绝覆盖既有目录）。"""
    name = validation_id or _default_validation_id()
    out = _assert_new_validation_out(
        ROOT / contracts.REPORTS_ROOT / "validations" / "research" / name
    )
    out.mkdir(parents=True)
    started = time.perf_counter()
    checks = _collect(
        name,
        out,
        linux_python=linux_python,
        linux_cwd=linux_cwd,
        windows_evidence=windows_evidence,
    )
    failed = sorted(
        key for key in _PASS_REQUIRED if not bool((checks.get(key) or {}).get("passed"))
    )
    # 未达成项 = ``known_limitations`` 中状态为 not_achieved 的编号（与计划 §24.4 同编号）
    limitations = [dict(item) for item in KNOWN_LIMITATIONS]
    linux_executed = (
        dict((checks.get("dual_platform") or {}).get("linux") or {}).get("status") == "executed"
    )
    unmet = sorted(
        item["id"]
        for item in limitations
        if item["status"] == "not_achieved" and not (item["id"] == "L1" and linux_executed)
    )
    # 评审 R3-T5-02：``unmet_items`` 必须参与最终 verdict——必需项未达成时不得输出 pass
    verdict = _final_verdict(failed, unmet)
    payload = _relativize(
        _sanitize(
            {
                "schema": "p6_t5_acceptance/v1",
                "validation_id": name,
                "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "project_version": generate._project_version(),
                "checks": checks,
                "failed_checks": failed,
                "unmet_items": unmet,
                "waivers": [dict(item) for item in USER_WAIVERS],
                "completion": {
                    "checks_passed": not failed,
                    "required_unmet": unmet,
                    "complete": not failed and not unmet,
                    "note": (
                        "``verdict=pass`` 仅在**全部检查项通过且无未达成必需项**时给出；"
                        "有未达成项（`status=not_achieved`）时为 ``incomplete``，不得当作通过"
                        "（评审 R3-T5-02）。`status=waived_by_user` 的条目由用户裁决豁免、"
                        "`by_design` 为设计约束，二者均不计入 `unmet_items`，但仍在 "
                        "`known_limitations` 中如实登记"
                    ),
                },
                "known_limitations": limitations,
                "workspace_note": (
                    "工作树含 T2/T3/T4 未提交内容（本轮 T5 只新增 scripts/v2_p6、tests、docs；"
                    "`git status -- src/` 的改动均来自前序阶段，非本任务）"
                ),
                "environment": {
                    "captured": env_probe.capture_environment(),
                    "note": "环境/依赖版本仅登记，不进 canonical 报告（不参与 verdict）",
                },
                "thresholds": dict(contracts.THRESHOLDS),
                "command_cwd": "<repo_root>",
                "path_style": "仓库相对 POSIX（命令记录已归一正斜杠）",
                "verdict": verdict,
            }
        )
    )
    validation.write_json_new(out / "t5_acceptance.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口（``--validation-id`` 缺省按 UTC 时间生成新 id）。"""
    parser = argparse.ArgumentParser(description="P6 T5 验收驱动（P6-M5）")
    parser.add_argument("--validation-id", default=None)
    parser.add_argument(
        "--linux-python", default=None, help="Linux 侧解释器（提供则在其上执行三面复算）"
    )
    parser.add_argument("--linux-cwd", default=None, help="Linux 侧仓库根（绝对路径）")
    parser.add_argument(
        "--windows-evidence", default=None, help="Windows 侧 t5_acceptance.json（Linux 运行时比对）"
    )
    args = parser.parse_args(argv)
    try:
        payload = run(
            args.validation_id,
            linux_python=args.linux_python,
            linux_cwd=args.linux_cwd,
            windows_evidence=Path(args.windows_evidence) if args.windows_evidence else None,
        )
    except AcceptanceSetupError as exc:
        print(f"[P6-T5] 设置错误：{exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "validation_id": payload["validation_id"],
                "failed_checks": payload["failed_checks"],
                "unmet_items": payload["unmet_items"],
            },
            ensure_ascii=False,
        )
    )
    if payload["verdict"] == "pass":
        return 0
    return 1 if payload["verdict"] == "fail" else 3


if __name__ == "__main__":
    raise SystemExit(main())

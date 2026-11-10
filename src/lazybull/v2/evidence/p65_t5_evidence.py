"""P6.5 T5 证据装配：G5~G7（恢复矩阵 / 同代信封本体 / 回归与复杂度 / 自含 manifest）。

契约依据：
- 设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md`` §10.1（G5 恢复、
  G6 回归、G7 证据身份）与 §10.1 证据根 ``data/frozen_reference/v2_p65/<run_id>/``；
- 任务 §3.3：证据根至少含 ``recovery_matrix.json``、``state_envelopes/``、
  ``pytest_targeted.txt``、``pytest_full.txt``、``complexity.json``、``manifest.json``。

口径复用（**不建第二套实现**）：
- 规范化 JSON / 目录树哈希 / SHA-256 与排除表来源哈希全部复用 T4 的
  ``evidence/p65_t4_evidence.py``（只读导入，该文件零改动）；
- 单段 ASCII ``run_id`` 校验复用 ``hosts/virtual_arm_runs.require_ascii_id``；
- 排除表内容与理由从 ``evidence/arm_comparator.py`` 现读并记录哈希（证明 T5 未放宽）。

边界（如实登记，不得读成更强结论）：
- 只产出**合成对账边界内**的候选门证据：不宣称 G5~G7 已通过阶段验收、不宣称 P6.5
  出口、B 区转正、生产多袖能力或前向/日历验证；
- 失败路径也落盘（FAIL 结论与差异清单保留现场），但**破缺态不写入新信封**；
- 已存在的证据目录一律拒绝（无覆盖）；不写 ``data/runs/``；不读真实配置/行情/账户。
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.core.accounting.reconciliation import (
    CASH_TOLERANCE,
    HOLDINGS_TOLERANCE_SHARES,
    NAV_TOLERANCE,
)
from src.lazybull.v2.core.accounting.state_envelope import (
    ENVELOPE_SCHEMA_VERSION,
    OWNER_NAMES,
)
from src.lazybull.v2.evidence.arm_comparator import (
    EXCLUDED_META_FIELDS,
    EXCLUSION_REASONS,
)

#: 复用 T4 的唯一实现（规范化 JSON / 树哈希 / 摘要 / 证据根）：只读导入，不改 T4 文件
from src.lazybull.v2.evidence.p65_t4_evidence import (
    EVIDENCE_ROOT as _T4_EVIDENCE_ROOT,
)
from src.lazybull.v2.evidence.p65_t4_evidence import (
    _sha256_bytes,
    _tree_hashes,
    _write_json,
)
from src.lazybull.v2.hosts.self_split_arms import (
    COMMISSION_RATE,
    MIN_COMMISSION,
    SLIPPAGE,
    STAMP_TAX,
)
from src.lazybull.v2.hosts.recovery_cases import (
    RecoveryMatrix,
    RecoveryScenario,
    run_recovery_matrix,
)
from src.lazybull.v2.hosts.virtual_arm_runs import require_ascii_id

#: 本模块唯一阶段标识
STAGE = "P6.5"
#: 证据 schema 版本（本模块自含，不等于 runs 契约版本）
T5_EVIDENCE_SCHEMA = 1
#: 证据根（与 T4 同一根，按 run_id 分目录）
EVIDENCE_ROOT = _T4_EVIDENCE_ROOT
#: 排除表来源模块（记录内容哈希，证明未放宽）
COMPARATOR_PATH = Path("src/lazybull/v2/evidence/arm_comparator.py")
#: T4 证据模块（复用来源，记录哈希以便复核）
T4_EVIDENCE_PATH = Path("src/lazybull/v2/evidence/p65_t4_evidence.py")
#: runs writer（T5 不写 runs，仅登记身份）
RUNS_WRITER_PATH = Path("src/lazybull/v2/hosts/backtest/runs_writer.py")
#: T5 交付面（**8 新增交付模块 + 5 被修改模块**）：
#: ``run_id`` 短内容指纹与 manifest ``code_surface`` 共用这一份清单（单一来源）。
#: 被修改模块 = `physical_lot_view`（成本/顺序/事件/初始身份）、`order_chain`（成本传递）、
#: `virtual_native_engine`（恢复代链级指纹门）、`common/cost`（四项显式费率时不读默认配置）、
#: `v2/common/cost`（同源同义复制件）；任何一处改动都必须改变 run_id。
CODE_SURFACE: Tuple[str, ...] = (
    "src/lazybull/common/cost.py",
    "src/lazybull/v2/common/cost.py",
    "src/lazybull/v2/core/accounting/state_envelope.py",
    "src/lazybull/v2/core/accounting/state_owners.py",
    "src/lazybull/v2/core/accounting/state_consistency.py",
    "src/lazybull/v2/core/accounting/envelope_store.py",
    "src/lazybull/v2/core/accounting/physical_lot_view.py",
    "src/lazybull/v2/hosts/recovery_runtime.py",
    "src/lazybull/v2/hosts/recovery_state_digest.py",
    "src/lazybull/v2/hosts/recovery_cases.py",
    "src/lazybull/v2/hosts/order_chain.py",
    "src/lazybull/v2/hosts/virtual_native_engine.py",
    "src/lazybull/v2/evidence/p65_t5_evidence.py",
)

#: 逐位比较口径（写入 manifest，禁止事后放宽）
COMPARE_RULE = (
    "float64 位模式（float.hex / struct 位串）+ 整数股 + 批次与键序；禁止 approx/容差兜底"
)
#: 跨恢复边界不能逐字节相等的登记（如实说明，不冒充"信封相同"）
ENVELOPE_BYTE_BOUNDARY = (
    "跨恢复边界的信封字节不等：信封含运行 ID 与 P4 pickle blob，而 pickle 协议 5 用 "
    "memo 表示重复对象引用（同一 Timestamp 在参照链内被多条目共享、跨恢复边界后是"
    "等价但不同的实例）。因此 G5 的逐位一致以**状态投影 digest**（值/位模式/键序）"
    "为判据，不以信封字节相等为判据"
)


class P65T5EvidenceError(ValueError):
    """证据装配输入非法（零副作用拒绝）。"""


class T5EvidenceVerdictError(P65T5EvidenceError):
    """证据已完整落盘但 G5 候选门判定为 FAIL（驱动映射退出码 3）。"""

    def __init__(self, message: str, *, root: Path, verdict: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.verdict = dict(verdict)
        super().__init__(message)


def _error(message: str) -> Exception:
    return P65T5EvidenceError(f"阶段={STAGE} T5 证据 {message}")


@dataclass(frozen=True, slots=True)
class T5EvidenceBundle:
    """T5 证据产物（根目录、逐文件 SHA-256、判定与是否完整）。"""

    run_id: str
    root: Path
    files: Tuple[Tuple[str, str], ...]
    verdict: Mapping[str, Any]
    complete: bool

    def sha_of(self, relative: str) -> str:
        """单文件 SHA-256（缺失即拒绝）。"""
        for path, digest in self.files:
            if path == relative:
                return digest
        raise _error(f"{relative} 不在证据产物集中")


def default_run_id(*, date: Optional[str] = None, salt: str = "") -> str:
    """ASCII 运行标识：阶段 + 日期 + 短内容指纹（不含主机名/墙钟噪声）。"""
    stamp = date or datetime.now().strftime("%Y%m%d")
    if not stamp.isdigit() or len(stamp) != 8:
        raise _error(f"日期必须为 YYYYMMDD，实际 {stamp!r}")
    digest = _sha256_bytes(f"{STAGE}-T5-{stamp}-{salt}".encode("utf-8"))[:8]
    return f"v2p65_t5_{stamp}_{digest}"


def _read_bytes_or_fail(path: Path, label: str, *, allow_empty: bool = False) -> bytes:
    """读取回归/静态门日志（缺失即拒绝；空内容仅对"无输出即成功"的门放行）。"""
    candidate = Path(path)
    if not candidate.is_file():
        raise _error(f"{label} 日志不存在：{candidate}；证据不得登记未发生的运行")
    payload = candidate.read_bytes()
    if not payload.strip() and not allow_empty:
        raise _error(f"{label} 日志为空：{candidate}")
    return payload


def reproduce_command(
    *,
    root: Any = "",
    run_id: str,
    pytest_targeted: Any,
    complexity: Any,
    complexity_src: Any = "",
    pytest_full: Any = "",
    flake8: Any = "",
) -> str:
    """按**实际参数**拼接可逐字复跑的命令（G7「文件哈希与命令」）。

    只登记真正决定证据内容的参数；缺省项不写（避免登记空值造成"复跑即报错"）。
    """
    parts = ["python scripts/v2_p65/run_t5_acceptance.py"]
    if str(root):
        parts.append(f"--root {root}")
    parts.append(f"--run-id {run_id}")
    if str(pytest_targeted):
        parts.append(f"--pytest-targeted {pytest_targeted}")
    if str(complexity):
        parts.append(f"--complexity {complexity}")
    if str(complexity_src):
        parts.append(f"--complexity-src {complexity_src}")
    if str(pytest_full):
        parts.append(f"--pytest-full {pytest_full}")
    if str(flake8):
        parts.append(f"--flake8 {flake8}")
    return " ".join(parts)


def content_salt(files: Sequence[str] = CODE_SURFACE, *, root: Path = Path(".")) -> str:
    """交付面短内容指纹（run_id 的"短内容指纹"段；文件缺失即拒绝）。

    口径：把交付面文件的**相对路径 + 字节**按清单顺序喂进同一个 SHA-256，取前 8 位
    十六进制。任何交付文件改动（含被修改的 `physical_lot_view.py` / `order_chain.py`）
    都会改变指纹；文件缺失直接报错，禁止用占位串静默退化（否则"同 run_id 即同代码"
    的推定失效）。
    """
    digest = hashlib.sha256()
    for relative in files:
        path = Path(root) / relative
        if not path.is_file():
            raise _error(f"内容指纹登记的文件缺失：{relative}（禁止静默退化）")
        digest.update(str(relative).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:8]


def code_surface_hashes() -> Dict[str, str]:
    """T5 交付面逐文件 SHA-256（缺失即拒绝：代码身份不得只存在于评审文书里）。"""
    hashes: Dict[str, str] = {}
    for relative in CODE_SURFACE:
        path = Path(relative)
        if not path.is_file():
            raise _error(f"代码面文件缺失：{relative}（禁止登记不存在的代码身份）")
        hashes[relative] = _sha256_bytes(path.read_bytes())
    return hashes


def _pending_full_declaration(*, reason: str) -> bytes:
    """全量 pytest 待跑声明（复审态证据；**不得**冒充已运行）。"""
    payload = {
        "status": "PENDING",
        "what": "pytest 全量回归",
        "reason": reason,
        "rule": "复审通过前禁止运行全量 pytest（CLAUDE.md §7.6，2026-11-09 裁决）",
        "note": "本文件不是运行日志；收口时必须用真实全量日志重新生成证据包",
    }
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _envelope_index(
    *, source_root: Path, target_dir: Path, name_prefix: str
) -> Dict[str, Any]:
    """把链上信封本体复制进证据包（两代及以上）+ 逐文件摘要索引。"""
    entries: List[Dict[str, Any]] = []
    if not source_root.is_dir():
        raise _error(f"信封链根不存在：{source_root}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_root.glob("envelope-*.json")):
        payload = path.read_bytes()
        relative = f"{name_prefix}/{path.name}"
        (target_dir / path.name).write_bytes(payload)
        document = json.loads(payload.decode("ascii"))
        entries.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "commit_point": int(document["commit_point"]),
                "trade_date": document["trade_date"],
                "run_id": document["run_id"],
                "previous_envelope_sha256": document["previous"]["envelope_sha256"],
                "owner_sha256": {
                    name: document["owners"][name]["sha256"] for name in OWNER_NAMES
                },
            }
        )
    if len(entries) < 2:
        raise _error(f"信封链至少需要两代（实际 {len(entries)} 个）：G5 需两代本体同包")
    if [entry["commit_point"] for entry in entries] != list(range(1, len(entries) + 1)):
        raise _error(f"信封提交点必须从 1 起连续：{[e['commit_point'] for e in entries]}")
    for index, entry in enumerate(entries[1:], start=1):
        if entry["previous_envelope_sha256"] != entries[index - 1]["sha256"]:
            raise _error(
                f"信封链摘要不衔接：提交点 {entry['commit_point']} 记录 "
                f"{entry['previous_envelope_sha256'][:12]}…，上一代实为 "
                f"{entries[index - 1]['sha256'][:12]}…"
            )
    index_document = {
        "schema_version": T5_EVIDENCE_SCHEMA,
        "run_id": entries[0]["run_id"],
        "envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
        "note": (
            "两代及以上信封本体与逐文件 SHA-256；可直接重算校验，且每代 "
            "previous_envelope_sha256 与上一代 sha256 逐项相等"
        ),
        "entries": entries,
    }
    _write_json(target_dir / "index.json", index_document)
    return index_document


def _exclusion_snapshot() -> Dict[str, Any]:
    """排除表登记（T5 未使用排除表；记录哈希以证明未放宽）。"""
    comparator = Path(COMPARATOR_PATH)
    t4_module = Path(T4_EVIDENCE_PATH)
    return {
        "fields": list(EXCLUDED_META_FIELDS),
        "reasons": dict(EXCLUSION_REASONS),
        "source_module": str(comparator).replace("\\", "/"),
        "source_sha256": _sha256_bytes(comparator.read_bytes()) if comparator.is_file() else "",
        "t4_evidence_module": str(t4_module).replace("\\", "/"),
        "t4_evidence_sha256": (
            _sha256_bytes(t4_module.read_bytes()) if t4_module.is_file() else ""
        ),
        "used_by_t5": False,
        "note": (
            "T5 只比较状态投影（自定义 21 字段 + 五项所有权），不使用 runs 帧排除表；"
            "此处登记来源与内容哈希，以便复核排除表未被放宽/扩大"
        ),
    }


def _tolerances() -> Dict[str, Any]:
    """容差登记（恒等式门 vs 逐位门严格分开）。"""
    return {
        "i1_cash_absolute": CASH_TOLERANCE,
        "i2_shares": HOLDINGS_TOLERANCE_SHARES,
        "i3_nav_absolute": NAV_TOLERANCE,
        "recovery_state_projection": 0,
        "envelope_owner_sha256": 0,
        "note": "I1/I3 为恒等式容差门；恢复比较与信封哈希为 0 容差（位模式/摘要相等）",
    }


def _rates() -> Dict[str, float]:
    """费率登记（合成费率，与 T4 同源）。"""
    return {
        "commission_rate": COMMISSION_RATE,
        "min_commission": MIN_COMMISSION,
        "stamp_tax": STAMP_TAX,
        "slippage": SLIPPAGE,
    }


def _scenario_list(matrix: RecoveryMatrix) -> List[Dict[str, Any]]:
    """恢复场景清单（名称 + 输入摘要 + 提交点 + 结论）。"""
    return [
        {
            "case": case.name,
            "description": case.description,
            "verdict": case.verdict,
            "commit_points": list(case.commit_points),
            "restart_points": [point for point, _watermark in case.restored_watermarks],
            "difference_count": len(case.differences),
        }
        for case in matrix.cases
    ]


def collect_t5_evidence(
    *,
    root: Path,
    run_id: str,
    code_state: Mapping[str, Any],
    command: str,
    pytest_targeted: Path,
    complexity_log: Path,
    complexity_src_log: Optional[Path] = None,
    pytest_full: Optional[Path] = None,
    flake8_log: Optional[Path] = None,
    pending_full_reason: str = "",
    scenario: Optional[RecoveryScenario] = None,
    scenario_notes: Optional[Mapping[str, Any]] = None,
) -> T5EvidenceBundle:
    """在 ``root/<run_id>/`` 下产出全部证据文件并写自含 manifest。

    实现口径：
    - **失败即零残留**：装配失败时删除本次创建的 ``root/<run_id>/``；
    - **失败路径也留证据**：G5 差异不在此抛错，而是写进 ``recovery_matrix.json`` 并把
      manifest verdict 记 FAIL，最后抛 ``T5EvidenceVerdictError``（驱动映射退出码 3）；
    - ``pytest_full`` 缺省 ⇒ 写入"待跑声明"并把 manifest 标记 ``PENDING``（复审态证据，
      收口时必须用真实日志重新生成，**不得**冒充已运行）。

    Args:
        root: 证据根（默认 ``data/frozen_reference/v2_p65``）
        run_id: ASCII 运行标识（目录名）
        code_state: ``{git_commit, git_dirty}``
        command: 复跑命令
        pytest_targeted: 定向回归日志
        complexity_log: 复杂度双口径日志（两口径合并）
        pytest_full: 全量回归日志（缺省视为复审态）
        flake8_log: flake8 日志（可缺省）
        pending_full_reason: 全量待跑原因（缺省用统一规则文案）
        scenario: 恢复算例（缺省默认算例）
        scenario_notes: 附加登记
    """
    try:
        run_id = require_ascii_id(run_id, "run_id")
    except ValueError as error:
        raise _error(f"run_id 非法：{error}") from error
    root = Path(root)
    target = root / run_id
    if target.resolve().parent != root.resolve():
        raise _error(
            f"run_id 解析后越出证据根（run_id={run_id!r} 目标={target.resolve()}）；"
            "拒绝在证据根之外创建任何目录"
        )
    if target.exists():
        raise _error(f"证据目录已存在 {target}；禁止覆盖既有证据（请使用新 run_id）")
    root.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True)
    try:
        manifest = _assemble(
            target,
            run_id=run_id,
            code_state=code_state,
            command=command,
            pytest_targeted=pytest_targeted,
            complexity_log=complexity_log,
            complexity_src_log=complexity_src_log,
            pytest_full=pytest_full,
            flake8_log=flake8_log,
            pending_full_reason=pending_full_reason,
            scenario=scenario,
            scenario_notes=scenario_notes,
        )
    except BaseException as error:
        # 区分"执行前的输入拒绝"与"已经开始运行之后的完整性/恒等式失败"（T5-Codex-C4）：
        # 后者必须保留破缺前已完成的信封与只读现场，并落下可核验的失败记录；
        # 前者（还没有任何证据文件）按零副作用清理。
        if not _retain_failure(target, error=error):
            shutil.rmtree(target, ignore_errors=True)
        raise
    verdict = dict(manifest["verdict"])
    files = _tree_hashes(target)
    bundle = T5EvidenceBundle(
        run_id=run_id,
        root=target,
        files=files,
        verdict=verdict,
        complete=bool(manifest["evidence_complete"]),
    )
    if verdict.get("recovery_matrix") != "PASS":
        raise T5EvidenceVerdictError(
            f"阶段={STAGE} T5 证据已落盘但 G5 候选门未通过：{verdict}；"
            f"差异清单见 {target / 'recovery_matrix.json'}",
            root=target,
            verdict=verdict,
        )
    return bundle


#: 全新解释器中执行的"禁止真实配置读取"探针（ASCII 源码，避免任何平台编码差异）。
#: 拦截 ``Config.load_config``：合成宿主的完整构造链一旦触碰默认配置即失败。
_CONFIG_PROBE_SOURCE = """
import shutil
import tempfile
from pathlib import Path
from src.lazybull.common.config import Config

def _blocked(self, path):
    raise RuntimeError("PROBE_CONFIG_READ:" + str(path))

Config.load_config = _blocked

from src.lazybull.v2.hosts.recovery_runtime import RecoverableRuntime, RecoveryScenario

# T5-Codex-R1-02 / R2-02：每次探针使用**仓库 temp/ 下的独立随机目录**，
# 只清理自己创建的那一个——并发运行互不删除、互不占用对方的信封链，
# 同时遵守"一切临时产物统一落仓库 temp/"的约定（不豁免到系统临时区）。
base = Path.cwd() / "temp"
base.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix="p65_t5_config_probe_", dir=base))
try:
    runtime = RecoverableRuntime(
        scenario=RecoveryScenario(), root=root / "chain", run_id="config-probe"
    )
    runtime.run_and_commit(0)
finally:
    shutil.rmtree(root, ignore_errors=True)
print("PROBE_OK")
"""


def _run_config_probe() -> Dict[str, Any]:
    """跑一次全新进程的配置读取探针，返回可核验结果（T5-Codex-C5）。"""
    completed = subprocess.run(  # noqa: S603 - 固定解释器 + 常量源码，无外部输入
        [sys.executable, "-c", _CONFIG_PROBE_SOURCE],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        timeout=300,
        check=False,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    return {
        "ok": completed.returncode == 0 and "PROBE_OK" in stdout,
        "returncode": completed.returncode,
        "command": f"{sys.executable} -c <config-probe>",
        "stdout_tail": stdout[-400:],
        "stderr_tail": stderr[-400:],
        "rule": (
            "全新解释器 + 拦截 Config.load_config：合成宿主完整构造链（build_host + 一日运行）"
            "必须不读默认配置；失败即证否"
        ),
    }


def _preflight_regression_inputs(
    *,
    pytest_targeted: Path,
    complexity_log: Path,
    complexity_src_log: Optional[Path],
    pytest_full: Optional[Path],
    flake8_log: Optional[Path],
) -> None:
    """执行前把回归输入全部读一遍（缺失即拒绝；**不产生任何文件**）。"""
    _read_bytes_or_fail(pytest_targeted, "定向回归")
    _read_bytes_or_fail(complexity_log, "复杂度（src+scripts）")
    if complexity_src_log is not None:
        _read_bytes_or_fail(complexity_src_log, "复杂度（src）")
    if pytest_full is not None:
        _read_bytes_or_fail(pytest_full, "全量回归")
    if flake8_log is not None:
        # flake8 无输出即通过（exit 0）：空文件是合法成功态（与落盘时同一判据）
        _read_bytes_or_fail(flake8_log, "flake8", allow_empty=True)


def _evidence_produced(target: Path) -> bool:
    """是否已经产生了**恢复证据**（信封链或矩阵；配置探针不算证据）。"""
    chains = target / "envelope_chains"
    if chains.is_dir() and any(path.is_file() for path in chains.rglob("*")):
        return True
    return (target / "recovery_matrix.json").is_file()


def _retain_failure(target: Path, *, error: BaseException) -> bool:
    """保留破缺前证据并写 ``failure.json``（返回 True = 已留证，调用方不得删除目录）。

    T5-Codex-C4：原实现在任何异常上 ``rmtree``，把已经落盘的有效信封与现场一并删除；
    现按"是否已产生证据"区分：已产生 ⇒ 留证 + 失败记录，未产生 ⇒ 零副作用清理。
    """
    if not _evidence_produced(target):
        return False
    retained = [
        [path.relative_to(target).as_posix(), _sha256_bytes(path.read_bytes())]
        for path in sorted(target.rglob("*"))
        if path.is_file()
    ]
    document = {
        "schema_version": T5_EVIDENCE_SCHEMA,
        "stage": STAGE,
        "task": "T5",
        "kind": "failure",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "error_type": type(error).__name__,
        "message": str(error)[:2000],
        "input_rejection": isinstance(error, (P65T5EvidenceError, ValueError)),
        "retained_files": retained,
        "rule": (
            "运行开始后的完整性/恒等式失败必须保留破缺前信封与只读现场；"
            "破缺态不得写成新的有效信封；本记录不是证据包 manifest，"
            "不构成 G5/G6/G7 通过"
        ),
    }
    _write_json(target / "failure.json", document)
    return True


def _assemble(
    target: Path,
    *,
    run_id: str,
    code_state: Mapping[str, Any],
    command: str,
    pytest_targeted: Path,
    complexity_log: Path,
    complexity_src_log: Optional[Path],
    pytest_full: Optional[Path],
    flake8_log: Optional[Path],
    pending_full_reason: str,
    scenario: Optional[RecoveryScenario],
    scenario_notes: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """在 ``target`` 下装配全部证据并返回 manifest（不负责目录创建/失败清理）。

    顺序是**先校验输入、再跑矩阵**：输入缺失属执行前拒绝，必须零副作用（目录都不该
    留下）；运行开始后的完整性/恒等式失败才留证（见 ``_retain_failure``）。
    """
    _preflight_regression_inputs(
        pytest_targeted=pytest_targeted,
        complexity_log=complexity_log,
        complexity_src_log=complexity_src_log,
        pytest_full=pytest_full,
        flake8_log=flake8_log,
    )
    config_probe = _run_config_probe()
    _write_json(target / "config_probe.json", config_probe)
    if not config_probe["ok"]:
        raise _error(
            "合成宿主在全新解释器中触碰了默认配置（探针失败）；"
            f"stdout={config_probe['stdout_tail']!r} stderr={config_probe['stderr_tail']!r}"
        )
    matrix = run_recovery_matrix(
        root=target / "envelope_chains", run_id=f"{run_id}-chain", scenario=scenario
    )
    _write_json(target / "recovery_matrix.json", matrix.document())
    envelope_index = _envelope_index(
        source_root=target / "envelope_chains" / "reference",
        target_dir=target / "state_envelopes",
        name_prefix="state_envelopes",
    )
    regression = _write_regression_files(
        target,
        pytest_targeted=pytest_targeted,
        complexity_log=complexity_log,
        complexity_src_log=complexity_src_log,
        pytest_full=pytest_full,
        flake8_log=flake8_log,
        pending_full_reason=pending_full_reason,
    )
    full_status = regression["pytest_full_status"]
    writer_path = Path(RUNS_WRITER_PATH)
    manifest = {
        "schema_version": T5_EVIDENCE_SCHEMA,
        "stage": STAGE,
        "task": "T5",
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "code_state": dict(code_state),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": _pandas_version(),
        },
        "command": command,
        "envelope_contract": {
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "owner_names": list(OWNER_NAMES),
            "commit_point_chain": [
                entry["commit_point"] for entry in envelope_index["entries"]
            ],
            "owner_count": len(OWNER_NAMES),
            "state_ownership": (
                "P4 基础 blob / 物理 lot 伴随视图 / 虚拟账 / 结算与估值身份 / "
                "schema 与同代标记，同一原子提交点"
            ),
        },
        "recovery": {
            "scenarios": _scenario_list(matrix),
            "scenario_count": len(matrix.cases),
            "reference_digest_sha256": matrix.reference_digest_sha256,
            "compare_rule": COMPARE_RULE,
            "envelope_byte_boundary": ENVELOPE_BYTE_BOUNDARY,
        },
        "code_surface": code_surface_hashes(),
        "inputs": {
            "scenario": matrix.scenario.summary(),
            "config": {
                "read": False,
                "reason": (
                    "T5 全部输入为显式合成值（场景/费率/容差见 scenario/rates/tolerances），"
                    "**不读取 configs/**，也不读真实行情/账户"
                ),
                "probe": {
                    "kind": "fresh_process_block_load_config",
                    "ok": bool(config_probe["ok"]),
                    "command": config_probe["command"],
                    "artifact": "config_probe.json",
                    "rule": config_probe["rule"],
                    "note": (
                        "`read=False` 由全新解释器探针证明（拦截 Config.load_config 后"
                        "仍能构造宿主并跑完一日），不是固定声明；探针失败时本包不得生成"
                    ),
                },
            },
            "rates": _rates(),
            "tolerances": _tolerances(),
            "exclusion_table": _exclusion_snapshot(),
            "recovery_parameters": {
                "restart_points": {
                    case.name: [point for point, _wm in case.restored_watermarks]
                    for case in matrix.cases
                },
                "half_commit_crash": "提交点 2 截断文件 + 残留临时文件",
                "identity_breach": "第 2 日物理现金 +0.01 注入（I1/I3 破缺）",
            },
        },
        "runs_writer": {
            "module": str(writer_path).replace("\\", "/"),
            "sha256": _sha256_bytes(writer_path.read_bytes()) if writer_path.is_file() else "",
            "wrote_runs": False,
            "note": "T5 不写 runs（恢复证据自含于本包）；仅只读登记 writer 身份",
        },
        "regression": regression["document"],
        "scenario_notes": dict(scenario_notes or {}),
        "file_sha256": [list(item) for item in _tree_hashes(target, skip=("manifest.json",))],
        "evidence_complete": full_status == "RUN",
        "verdict": {
            "recovery_matrix": matrix.document()["verdict"],
            "case_count": len(matrix.cases),
            "failed_cases": [case.name for case in matrix.cases if not case.equal],
            "difference_count": sum(len(case.differences) for case in matrix.cases),
            "pytest_full": full_status,
            "envelope_generations": len(envelope_index["entries"]),
            "boundary": (
                "合成对账边界内的候选门结果：不构成 P6.5 阶段出口、B 区转正、"
                "生产多袖能力或前向/日历验证"
            ),
        },
    }
    _write_json(target / "manifest.json", manifest)
    return manifest


def _write_regression_files(
    target: Path,
    *,
    pytest_targeted: Path,
    complexity_log: Path,
    complexity_src_log: Optional[Path],
    pytest_full: Optional[Path],
    flake8_log: Optional[Path],
    pending_full_reason: str,
) -> Dict[str, Any]:
    """把回归/静态门日志写入包内并返回 ``{document, pytest_full_status}``。

    双口径复杂度**分列入包**（`src` / `src+scripts`）；全量缺省写"待跑声明"并标记
    ``PENDING``（复审态证据不得冒充已运行）。
    """
    targeted = _read_bytes_or_fail(pytest_targeted, "定向回归")
    complexity_bytes = _read_bytes_or_fail(complexity_log, "复杂度（src+scripts）")
    (target / "pytest_targeted.txt").write_bytes(targeted)
    (target / "complexity.json").write_bytes(complexity_bytes)
    complexity_entries: List[Dict[str, Any]] = [
        {
            "caliber": "src+scripts",
            "path": "complexity.json",
            "bytes": len(complexity_bytes),
            "sha256": _sha256_bytes(complexity_bytes),
            "source": str(complexity_log).replace("\\", "/"),
        }
    ]
    if complexity_src_log is not None:
        src_bytes = _read_bytes_or_fail(complexity_src_log, "复杂度（src）")
        (target / "complexity_src.json").write_bytes(src_bytes)
        complexity_entries.insert(
            0,
            {
                "caliber": "src",
                "path": "complexity_src.json",
                "bytes": len(src_bytes),
                "sha256": _sha256_bytes(src_bytes),
                "source": str(complexity_src_log).replace("\\", "/"),
            },
        )
    if pytest_full is not None:
        (target / "pytest_full.txt").write_bytes(_read_bytes_or_fail(pytest_full, "全量回归"))
        full_status = "RUN"
    else:
        (target / "pytest_full.txt").write_bytes(
            _pending_full_declaration(
                reason=pending_full_reason
                or "复审通过前禁止运行全量 pytest（CLAUDE.md §7.6，2026-11-09 裁决）"
            )
        )
        full_status = "PENDING"
    if flake8_log is not None:
        # flake8 无输出即通过（exit 0）：空日志是合法成功态，但仍必须证明文件存在
        flake8_bytes = _read_bytes_or_fail(flake8_log, "flake8", allow_empty=True)
        (target / "flake8.txt").write_bytes(flake8_bytes)
        flake8_status = "RUN"
    else:
        flake8_bytes = b""
        flake8_status = "MISSING"
    document = {
        "pytest_targeted": {
            "path": "pytest_targeted.txt",
            "bytes": len(targeted),
            "sha256": _sha256_bytes(targeted),
            "source": str(pytest_targeted).replace("\\", "/"),
        },
        "pytest_full": {
            "path": "pytest_full.txt",
            "status": full_status,
            "bytes": (target / "pytest_full.txt").stat().st_size,
            "sha256": _sha256_bytes((target / "pytest_full.txt").read_bytes()),
            "source": str(pytest_full).replace("\\", "/") if pytest_full is not None else "",
        },
        "complexity": complexity_entries,
        "complexity_note": (
            "双口径分列：`src` 与 `src+scripts` 各一份日志入包；两者均须 exit=0、"
            "新增硬超限 0、基线零新增（任务 §2.2 / §9.1）"
        ),
        "flake8": {
            "path": "flake8.txt" if flake8_status == "RUN" else "",
            "status": flake8_status,
            "bytes": len(flake8_bytes),
            "sha256": _sha256_bytes(flake8_bytes) if flake8_status == "RUN" else "",
            "source": str(flake8_log).replace("\\", "/") if flake8_log is not None else "",
        },
    }
    return {"document": document, "pytest_full_status": full_status}


def _pandas_version() -> str:
    """pandas 版本（环境登记；导入失败即拒绝）。"""
    import pandas as pd

    return str(pd.__version__)


__all__ = (
    "CODE_SURFACE",
    "COMPARE_RULE",
    "EVIDENCE_ROOT",
    "ENVELOPE_BYTE_BOUNDARY",
    "P65T5EvidenceError",
    "T5_EVIDENCE_SCHEMA",
    "T5EvidenceBundle",
    "T5EvidenceVerdictError",
    "code_surface_hashes",
    "collect_t5_evidence",
    "default_run_id",
)

"""P6.5 T5 恢复矩阵：无中断参照 vs 五类恢复场景的编排与结论。

契约依据：设计 §4.5（三类恢复 + 半提交代崩溃 + 恒等式破缺终态）、§10.1 G5。

五类场景（每场景独立链根，禁止共根）：
1. ``multi_source_lot``：多来源 lot（D1 内部转账跨 L1/L2）从提交点 1 恢复；
2. ``mid_commit_point``：中途提交点（提交点 3）恢复；
3. ``transfer_after_resume``：恢复后**再次内部转账**（提交点 1/2 各中断一次）；
4. ``post_fill_watermark``：成交后水位恢复（提交点 4）——``_applied_keys``/
   ``last_fill_date`` 必须回到同代起点；
5. ``half_commit_crash``：半提交代崩溃——残骸阻断重复提交，必须从提交点前信封新建
   实例一起重放；
6. ``identity_breach_terminal``：恒等式破缺为终态——不可续跑、不可写新信封，修复后
   从破缺前完整信封重放。

边界：只声明**合成对账边界内**的候选门结论；不宣称 P6.5 出口、B 区转正或生产多袖。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.core.accounting.envelope_store import EnvelopeStoreError
from src.lazybull.v2.core.accounting.state_envelope import (
    StateEnvelopeCorruptError,
    StateEnvelopeError,
)
from src.lazybull.v2.hosts.recovery_runtime import (
    RecoverableRuntime,
    RecoveryError,
    RecoveryIntegrityError,
    RecoveryScenario,
    RestoredGeneration,
    _error,
    _integrity,
    _price_data,
    batch_id_for_day,
    build_host,
)
from src.lazybull.v2.hosts.recovery_state_digest import (
    _trade_repr,
    compare_digests,
    digest_sha256,
)
from src.lazybull.v2.hosts.virtual_native_engine import IdentityTerminatedError

#: 本模块唯一阶段标识
STAGE = "P6.5"

# ────────────────────────── 恢复矩阵 ──────────────────────────


@dataclass(frozen=True, slots=True)
class RecoveryCase:
    """单个恢复场景的结论（输入、提交点、信封哈希、逐字段比较结论）。"""

    name: str
    description: str
    scenario: RecoveryScenario
    commit_points: Tuple[int, ...]
    envelope_hashes: Tuple[Tuple[int, str], ...]
    reference_digest_sha256: str
    recovered_digest_sha256: str
    equal: bool
    differences: Tuple[str, ...]
    restored_watermarks: Tuple[Tuple[int, Mapping[str, Any]], ...]
    verdict: str
    notes: Tuple[str, ...] = ()

    def document(self) -> Dict[str, Any]:
        """证据文档投影（浮点全部位精确编码）。"""
        return {
            "case": self.name,
            "description": self.description,
            "verdict": self.verdict,
            "inputs": self.scenario.summary(),
            "commit_points": list(self.commit_points),
            "envelope_sha256": [[point, digest] for point, digest in self.envelope_hashes],
            "reference_digest_sha256": self.reference_digest_sha256,
            "recovered_digest_sha256": self.recovered_digest_sha256,
            "bitwise_equal": bool(self.equal),
            "differences": list(self.differences),
            "difference_count": len(self.differences),
            "restored_watermarks": [
                {"commit_point": point, "watermark": dict(watermark)}
                for point, watermark in self.restored_watermarks
            ],
            "notes": list(self.notes),
        }


def _watermark_of(restored: RestoredGeneration) -> Dict[str, Any]:
    """恢复代的水位投影（证据与断言共用）。"""
    return {
        "applied_keys": list(restored.applied_keys),
        "last_fill_date": restored.last_fill_date,
        "settlement_keys": list(restored.settlement_keys),
        "batch_ids": list(restored.batch_ids),
        "last_event_sequence": restored.last_event_sequence,
        "price_source_id": restored.price_source_id,
        "price_source_as_of": restored.price_source_as_of,
        "envelope_sha256": restored.envelope_sha256,
    }


def run_reference(
    *, root: Path, run_id: str, scenario: Optional[RecoveryScenario] = None
) -> Tuple[RecoverableRuntime, Dict[str, Any]]:
    """无中断参照：逐日跑完并提交，返回最终逐位投影。"""
    scenario = scenario or RecoveryScenario()
    runtime = RecoverableRuntime(scenario=scenario, root=root, run_id=run_id)
    for index in range(len(scenario.date_list())):
        runtime.run_and_commit(index)
    return runtime, runtime.digest(len(scenario.date_list()) - 1)


def _assert_bitwise_equal(
    *, case: str, reference: Mapping[str, Any], recovered: Mapping[str, Any]
) -> Tuple[str, Tuple[str, ...]]:
    """逐位比较并返回 ``(参照摘要, 差异清单)``；差异即终态破缺（禁止容差兜底）。"""
    differences = compare_digests(reference, recovered)
    return digest_sha256(reference), differences


def run_recovery_case(
    *,
    root: Path,
    run_id: str,
    case: str,
    description: str,
    reference_digest: Mapping[str, Any],
    restart_points: Sequence[int],
    scenario: Optional[RecoveryScenario] = None,
) -> RecoveryCase:
    """在指定提交点中断并重建实例，跑完剩余日并与参照逐位比较。

    Args:
        root: 本场景的独立链根（每场景一个目录，禁止共根）
        run_id: 链运行标识
        case: 场景名
        description: 场景说明
        reference_digest: 无中断参照的最终投影
        restart_points: 在这些提交点之后中断并 `restart`（升序，必须递增）
    """
    scenario = scenario or RecoveryScenario()
    days = len(scenario.date_list())
    points = tuple(int(item) for item in restart_points)
    if list(points) != sorted(set(points)):
        raise _error(f"重启提交点必须严格升序且不重复：{list(points)}")
    runtime = RecoverableRuntime(scenario=scenario, root=root, run_id=run_id)
    watermarks: List[Tuple[int, Mapping[str, Any]]] = []
    notes: List[str] = []
    next_day = 0
    for point in points + (days,):
        if point < next_day:
            raise _error(f"重启提交点 {point} 早于当前日序 {next_day}（禁止回退重放）")
        for day in range(next_day, point):
            runtime.run_and_commit(day)
        next_day = point
        if point < days:
            restored = runtime.restart(point)
            watermark = _watermark_of(restored)
            watermarks.append((point, watermark))
            notes.append(
                f"在第 {point} 个提交点新建实例并恢复（信封 {restored.envelope_sha256[:12]}…）"
            )
    final_digest = runtime.digest(days - 1)
    reference_sha, differences = _assert_bitwise_equal(
        case=case, reference=reference_digest, recovered=final_digest
    )
    return RecoveryCase(
        name=case,
        description=description,
        scenario=scenario,
        commit_points=runtime.store.commit_points(),
        envelope_hashes=runtime.envelope_hashes(),
        reference_digest_sha256=reference_sha,
        recovered_digest_sha256=digest_sha256(final_digest),
        equal=not differences,
        differences=differences,
        restored_watermarks=tuple(watermarks),
        verdict="PASS" if not differences else "FAIL",
        notes=tuple(notes),
    )


def run_half_commit_case(
    *, root: Path, run_id: str, reference_digest: Mapping[str, Any],
    scenario: Optional[RecoveryScenario] = None,
) -> RecoveryCase:
    """半提交代崩溃：残骸阻断重复提交，必须从提交点前信封新建实例一起重放。"""
    scenario = scenario or RecoveryScenario()
    days = len(scenario.date_list())
    runtime = RecoverableRuntime(scenario=scenario, root=root, run_id=run_id)
    notes: List[str] = []
    runtime.run_and_commit(0)
    crash_day = 1
    runtime.run_day(crash_day)
    final_path, temp_path = runtime.crash_after(crash_day)
    notes.append(
        f"半提交代现场：{final_path.name}（{final_path.stat().st_size} 字节）+ "
        f"{temp_path.name}（{temp_path.stat().st_size} 字节）"
    )
    # ① 最新代不可读（不静默回退）
    try:
        runtime.store.load_latest()
    except StateEnvelopeCorruptError:
        notes.append("load_latest 拒绝半提交代（未静默回退到提交点 1）")
    else:  # pragma: no cover - 防御性
        raise _integrity("半提交代被当作完整信封读取（禁止静默回退）")
    # ② 已推进的内存实例不得以同一提交点重复提交
    commit_point = crash_day + 1
    try:
        runtime.commit_day(crash_day)
    except (EnvelopeStoreError, StateEnvelopeError):
        notes.append(
            f"已推进实例的提交点 {commit_point} 提交被拒（残骸在场，禁止用已推进水位重复提交）"
        )
    else:  # pragma: no cover - 防御性
        raise _integrity("已推进的内存实例被允许重复提交（半提交代未被阻断）")
    # ③ 同一实例重放同一批次：同键同内容**不重复入账**（applied=False 且状态不变）
    cash_before = float(runtime.host.current_capital)
    trades_before = _trade_repr(runtime.host)
    positions_before = repr(sorted(runtime.host.positions))
    replay = runtime.host.execute_order_chain(
        scenario.date_list()[crash_day],
        scenario.targets_of(crash_day),
        batch_id=batch_id_for_day(crash_day),
    )
    if replay.applied:
        raise _integrity("同键同内容的重放被当作首次应用（重复入账）")
    if (
        float(runtime.host.current_capital) != cash_before
        or _trade_repr(runtime.host) != trades_before
        or repr(sorted(runtime.host.positions)) != positions_before
    ):
        raise _integrity("同键同内容的重放改变了物理账户（幂等承诺被破坏）")
    notes.append("同实例重放同一批次返回 applied=False 且现金/持仓/成交日志逐位不变")
    # ④ 隔离现场 → 从提交点 1 新建实例重放
    moved = runtime.store.quarantine_incomplete(
        commit_point, reason="half-commit crash simulation"
    )
    notes.append(f"崩溃现场已隔离到 {moved[0].parent.name}/（{len(moved)} 个残骸）")
    runtime.host = build_host(scenario, run_id=run_id)  # 丢弃已推进实例
    restored = runtime.restart(1)
    for day in range(1, days):
        runtime.run_and_commit(day)
    final_digest = runtime.digest(days - 1)
    reference_sha, differences = _assert_bitwise_equal(
        case="half_commit_crash", reference=reference_digest, recovered=final_digest
    )
    return RecoveryCase(
        name="half_commit_crash",
        description="半提交代崩溃：残骸阻断重复提交，从提交点前信封新建实例一起重放",
        scenario=scenario,
        commit_points=runtime.store.commit_points(),
        envelope_hashes=runtime.envelope_hashes(),
        reference_digest_sha256=reference_sha,
        recovered_digest_sha256=digest_sha256(final_digest),
        equal=not differences,
        differences=differences,
        restored_watermarks=((1, _watermark_of(restored)),),
        verdict="PASS" if not differences else "FAIL",
        notes=tuple(notes),
    )


def run_breach_terminal_case(
    *, root: Path, run_id: str, reference_digest: Mapping[str, Any],
    scenario: Optional[RecoveryScenario] = None,
) -> RecoveryCase:
    """恒等式破缺终态：破缺不可续跑/不可写新信封，修复后从破缺前信封重放。"""
    from src.lazybull.v2.core.accounting.reconciliation import ReconciliationError

    scenario = scenario or RecoveryScenario()
    days = len(scenario.date_list())
    runtime = RecoverableRuntime(scenario=scenario, root=root, run_id=run_id)
    notes: List[str] = []
    breach_day = 2
    runtime.run_and_commit(0)
    runtime.run_and_commit(1)
    # 破缺前信封 = 提交点 2（覆盖到第 1 日）
    before_breach = runtime.store.load(2)
    runtime.run_day(breach_day)
    # 在终态写入前注入守恒破缺（物理现金 +0.01，只有 I1 破缺）
    tampered = float(runtime.host.current_capital) + 0.01
    dates = scenario.date_list()
    runtime.host.current_capital = tampered
    try:
        runtime.host._finalize_day(
            breach_day,
            dates[breach_day],
            0,
            len(dates),
            breach_day + 1,
            runtime.host._calculate_portfolio_value(dates[breach_day]),
            list(dates),
            {item: position for position, item in enumerate(dates)},
        )
    except ReconciliationError as error:
        notes.append(f"恒等式破缺：{error.identities}")
    else:  # pragma: no cover - 防御性
        raise _integrity("注入的现金偏移未触发 I1 破缺")
    if not runtime.host.is_terminated:  # pragma: no cover - 防御性
        raise _integrity("破缺后宿主未进入终态")
    # ① 破缺态不得写入新信封（提交点与上一代摘要均不变）
    try:
        runtime.commit_day(breach_day)
    except (RecoveryIntegrityError, StateEnvelopeError):
        notes.append("破缺态写入新信封被拒")
    else:  # pragma: no cover - 防御性
        raise _integrity("破缺态被允许写入新信封")
    if runtime.store.commit_points() != (1, 2):  # pragma: no cover - 防御性
        raise _integrity(f"破缺后提交点发生变化：{runtime.store.commit_points()}")
    if runtime.store.load(2).file_sha256 != before_breach.file_sha256:  # pragma: no cover
        raise _integrity("破缺后上一代信封摘要发生变化")
    notes.append(
        f"破缺时提交点快照={list(runtime.store.commit_points())}；破缺前完整信封=提交点 2"
        f"（sha={before_breach.file_sha256[:12]}…，覆盖到第 1 日）"
    )
    # ② 同实例续跑被拒
    try:
        runtime.host.run_step(dates[breach_day + 1], list(dates), _price_data(scenario))
    except IdentityTerminatedError:
        notes.append("破缺后同实例续跑被拒（IdentityTerminatedError）")
    else:  # pragma: no cover - 防御性
        raise _integrity("破缺后同实例仍可续跑")
    # ③ 修复后从破缺前完整信封新建实例重放
    runtime.host = build_host(scenario, run_id=run_id)
    restored = runtime.restart(2)
    notes.append(
        f"修复路径：由提交点 2 新建实例重放第 {breach_day}~{days - 1} 日，"
        f"写出提交点 3~{days}（不是破缺日续写）"
    )
    for day in range(breach_day, days):
        runtime.run_and_commit(day)
    final_digest = runtime.digest(days - 1)
    reference_sha, differences = _assert_bitwise_equal(
        case="identity_breach_terminal", reference=reference_digest, recovered=final_digest
    )
    return RecoveryCase(
        name="identity_breach_terminal",
        description="恒等式破缺为终态：不可续跑/不可写新信封，修复后从破缺前信封重放",
        scenario=scenario,
        commit_points=runtime.store.commit_points(),
        envelope_hashes=runtime.envelope_hashes(),
        reference_digest_sha256=reference_sha,
        recovered_digest_sha256=digest_sha256(final_digest),
        equal=not differences,
        differences=differences,
        restored_watermarks=((2, _watermark_of(restored)),),
        verdict="PASS" if not differences else "FAIL",
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class RecoveryMatrix:
    """恢复矩阵（G5 候选证据的主体）。"""

    scenario: RecoveryScenario
    reference_digest: Mapping[str, Any]
    reference_digest_sha256: str
    reference_envelopes: Tuple[Tuple[int, str], ...]
    cases: Tuple[RecoveryCase, ...]

    @property
    def passed(self) -> bool:
        """全部场景逐位一致才通过。"""
        return all(case.equal for case in self.cases)

    def document(self) -> Dict[str, Any]:
        """证据文档投影。"""
        return {
            "case": "G5_recovery_matrix",
            "scenario": self.scenario.summary(),
            "reference_digest_sha256": self.reference_digest_sha256,
            "reference_envelopes": [[point, digest] for point, digest in self.reference_envelopes],
            "cases": [case.document() for case in self.cases],
            "case_count": len(self.cases),
            "failed_cases": [case.name for case in self.cases if not case.equal],
            "verdict": "PASS" if self.passed else "FAIL",
            "compare_rule": (
                "float64 位模式（float.hex）+ 整数股 + 批次/键序；禁止 approx/容差兜底"
            ),
            "boundary": (
                "合成对账边界内的候选门结果：不构成 P6.5 阶段出口、不代表 B 区转正、"
                "生产多袖能力或前向/日历验证"
            ),
        }


def run_recovery_matrix(
    *, root: Path, run_id: str, scenario: Optional[RecoveryScenario] = None
) -> RecoveryMatrix:
    """跑无中断参照 + 五类恢复场景（每场景独立链根，禁止共根）。"""
    scenario = scenario or RecoveryScenario()
    base = Path(root)
    reference, reference_digest = run_reference(
        root=base / "reference", run_id=f"{run_id}-ref", scenario=scenario
    )
    cases = [
        run_recovery_case(
            root=base / "multi_source_lot",
            run_id=f"{run_id}-m1",
            case="multi_source_lot",
            description="多来源 lot：D1 内部转账跨 L1/L2 两个物理来源，从提交点 1 恢复",
            reference_digest=reference_digest,
            restart_points=(1,),
            scenario=scenario,
        ),
        run_recovery_case(
            root=base / "mid_commit_point",
            run_id=f"{run_id}-m2",
            case="mid_commit_point",
            description="中途提交点恢复：提交点 3（两次转账后）中断并重建实例",
            reference_digest=reference_digest,
            restart_points=(3,),
            scenario=scenario,
        ),
        run_recovery_case(
            root=base / "transfer_after_resume",
            run_id=f"{run_id}-m3",
            case="transfer_after_resume",
            description="恢复后再次内部转账：提交点 1/2 各中断一次，恢复后连续两次转账",
            reference_digest=reference_digest,
            restart_points=(1, 2),
            scenario=scenario,
        ),
        run_recovery_case(
            root=base / "post_fill_watermark",
            run_id=f"{run_id}-m6",
            case="post_fill_watermark",
            description=(
                "成交后水位恢复：提交点 4（已有一笔物理成交）中断，恢复后 "
                "_applied_keys/last_fill_date 必须回到同代起点再继续第二笔成交"
            ),
            reference_digest=reference_digest,
            restart_points=(4,),
            scenario=scenario,
        ),
        run_half_commit_case(
            root=base / "half_commit_crash",
            run_id=f"{run_id}-m4",
            reference_digest=reference_digest,
            scenario=scenario,
        ),
        run_breach_terminal_case(
            root=base / "identity_breach",
            run_id=f"{run_id}-m5",
            reference_digest=reference_digest,
            scenario=scenario,
        ),
    ]
    return RecoveryMatrix(
        scenario=scenario,
        reference_digest=reference_digest,
        reference_digest_sha256=digest_sha256(reference_digest),
        reference_envelopes=reference.envelope_hashes(),
        cases=tuple(cases),
    )


__all__ = (
    "RecoverableRuntime",
    "RecoveryCase",
    "RecoveryError",
    "RecoveryIntegrityError",
    "RecoveryMatrix",
    "RecoveryScenario",
    "RestoredGeneration",
    "build_host",
    "compare_digests",
    "digest_sha256",
    "run_breach_terminal_case",
    "run_half_commit_case",
    "run_recovery_case",
    "run_recovery_matrix",
    "run_reference",
)

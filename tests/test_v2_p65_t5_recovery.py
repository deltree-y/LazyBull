# -*- coding: utf-8 -*-
"""P6.5 T5 同代恢复语义：三类恢复与无中断参照**逐位一致**。

覆盖（任务 §3.2）：
- 多来源 lot：D1 内部转账跨 L1/L2 两个物理来源，恢复后仍逐位一致；
- 中途提交点恢复：提交点 3 中断重建；
- 恢复后**再次内部转账**：提交点 1/2 各中断一次，恢复后连续两次转账；
- 成交后水位：恢复后 ``_applied_keys/last_fill_date`` 回到同代起点再继续第二笔成交；
- 判据为状态投影 digest（float64 位模式 + 整数股 + 键序），禁用 ``approx``/容差。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pytest

from src.lazybull.v2.hosts.recovery_cases import (
    RecoveryMatrix,
    run_recovery_matrix,
    run_reference,
)
from src.lazybull.v2.hosts.recovery_runtime import (
    RecoverableRuntime,
    RecoveryIntegrityError,
    RecoveryScenario,
)
from src.lazybull.v2.hosts.recovery_state_digest import (
    compare_digests,
    digest_sha256,
    state_digest,
)
from src.lazybull.v2.core.accounting.state_owners import decode_owners

_RUN_ID = "p65t5-recovery"


@pytest.fixture(scope="module")
def matrix(tmp_path_factory: pytest.TempPathFactory) -> RecoveryMatrix:
    """整套恢复矩阵（模块级复用：六条链各跑五日，约 1~2 秒）。"""
    root = tmp_path_factory.mktemp("p65_t5_matrix")
    return run_recovery_matrix(root=root, run_id=_RUN_ID)


def _case(matrix: RecoveryMatrix, name: str):
    """按名取恢复场景结论（缺即失败）。"""
    for case in matrix.cases:
        if case.name == name:
            return case
    raise AssertionError(f"恢复矩阵缺少场景 {name}；实际={[c.name for c in matrix.cases]}")


def test_reference_run_is_deterministic(tmp_path: Path) -> None:
    """两次独立无中断运行逐日投影完全相同（参照本身可复现）。"""
    scenario = RecoveryScenario()
    first, first_digest = run_reference(
        root=tmp_path / "a", run_id="p65t5-ref-a", scenario=scenario
    )
    second, second_digest = run_reference(
        root=tmp_path / "b", run_id="p65t5-ref-b", scenario=scenario
    )
    assert digest_sha256(first_digest) == digest_sha256(second_digest)
    assert compare_digests(first_digest, second_digest) == ()
    for index in range(len(scenario.date_list())):
        left: Dict[str, Any] = state_digest(first.host, scenario=scenario, day_index=index)
        right: Dict[str, Any] = state_digest(second.host, scenario=scenario, day_index=index)
        assert digest_sha256(left) == digest_sha256(right)


def test_reference_scenario_exercises_transfer_and_fills(tmp_path: Path) -> None:
    """参照算例本身覆盖：纯转账日（D1/D2）、物理买入日（D3/D4）、多来源 lot。"""
    scenario = RecoveryScenario()
    runtime = RecoverableRuntime(scenario=scenario, root=tmp_path / "ref", run_id="p65t5-ref")
    runtime.run_and_commit(0)  # 期初估值日（无指令）
    # D1：内部转账 250 股（跨 L1/L2）——物理零成交
    result = runtime.run_day(1)
    assert result is not None and result.plan.physical_orders() == ()
    runtime.commit_day(1)
    shares_by_source = {
        lot.source_lot_id
        for state in runtime.host.virtual_ledger.snapshot().values()
        for lot in state.lots
    }
    assert shares_by_source == {"L1", "L2"}
    # D3：物理买入 100 股（真实成交 + 费用）
    runtime.run_and_commit(2)
    runtime.run_and_commit(3)
    ledger_state = runtime.host.ledger.export_state()
    assert len(ledger_state["applied_keys"]) == 1
    assert runtime.host.positions[scenario.ts_code]["shares"] == 500


def test_multi_source_lot_recovery_is_bitwise_equal(matrix: RecoveryMatrix) -> None:
    """多来源 lot 恢复：与无中断参照逐位一致（含双层 lot 与费用）。"""
    case = _case(matrix, "multi_source_lot")
    assert case.verdict == "PASS"
    assert case.differences == ()
    assert case.reference_digest_sha256 == case.recovered_digest_sha256
    assert [point for point, _wm in case.restored_watermarks] == [1]
    watermark = dict(case.restored_watermarks[0][1])
    assert watermark["applied_keys"] == []
    assert watermark["last_fill_date"] is None
    assert watermark["batch_ids"] == []


def test_mid_commit_point_recovery_is_bitwise_equal(matrix: RecoveryMatrix) -> None:
    """中途提交点恢复（提交点 3 = 两次转账之后）逐位一致。"""
    case = _case(matrix, "mid_commit_point")
    assert case.verdict == "PASS"
    assert case.differences == ()
    watermark = dict(case.restored_watermarks[0][1])
    assert watermark["batch_ids"] == ["p65t5-d1", "p65t5-d2"]
    assert watermark["last_event_sequence"] == 4  # 2 批次 + 2 派生转账子键


def test_transfer_after_resume_is_bitwise_equal(matrix: RecoveryMatrix) -> None:
    """恢复后再次内部转账：每次重建后的下一日仍是转账日，结果逐位一致。"""
    case = _case(matrix, "transfer_after_resume")
    assert case.verdict == "PASS"
    assert [point for point, _wm in case.restored_watermarks] == [1, 2]
    first = dict(case.restored_watermarks[0][1])
    second = dict(case.restored_watermarks[1][1])
    assert first["batch_ids"] == []
    assert second["batch_ids"] == ["p65t5-d1"]
    # 批次内转账占用派生子键 ``<batch>#T<i>``：D1 一笔转账 ⇒ 恰一个派生键
    assert second["settlement_keys"] == ["p65t5-d1#T0"]


def test_post_fill_watermark_returns_to_generation_start(
    matrix: RecoveryMatrix, tmp_path: Path
) -> None:
    """成交后恢复：``_applied_keys/last_fill_date`` 回到同代起点，再继续第二笔成交。"""
    case = _case(matrix, "post_fill_watermark")
    assert case.verdict == "PASS"
    watermark = dict(case.restored_watermarks[0][1])
    assert len(watermark["applied_keys"]) == 1
    assert watermark["last_fill_date"] == "2026-01-08 00:00:00"

    # 独立复算：从提交点 4 恢复后，水位必须与信封登记的结算身份逐项相等
    scenario = RecoveryScenario()
    runtime = RecoverableRuntime(scenario=scenario, root=tmp_path / "chain", run_id="p65t5-wm")
    for index in range(4):
        runtime.run_and_commit(index)
    stored = runtime.store.load(4)
    identity = decode_owners(stored.envelope).identity
    restored = runtime.restart(4)
    assert restored.applied_keys == identity.physical_fill_keys
    assert restored.last_fill_date == identity.physical_last_fill_date
    assert restored.settlement_keys == identity.internal_settlement_keys
    assert restored.batch_ids == identity.batch_ids
    assert restored.last_event_sequence == identity.last_event_sequence
    assert restored.envelope_sha256 == stored.file_sha256

    # 继续跑第 5 日：水位只增加一笔成交键（不重放旧键）
    runtime.run_and_commit(4)
    final = runtime.host.ledger.export_state()
    assert list(final["applied_keys"])[:1] == list(identity.physical_fill_keys)
    assert len(final["applied_keys"]) == len(identity.physical_fill_keys) + 1
    assert str(final["last_fill_date"]) == "2026-01-09 00:00:00"


def test_half_commit_crash_replays_from_previous_envelope(matrix: RecoveryMatrix) -> None:
    """半提交代崩溃：残骸阻断重复提交，从提交点前信封新建实例重放后逐位一致。"""
    case = _case(matrix, "half_commit_crash")
    assert case.verdict == "PASS"
    assert case.differences == ()
    notes = "；".join(case.notes)
    assert "load_latest 拒绝半提交代" in notes
    assert "提交被拒" in notes
    assert "applied=False" in notes
    assert "崩溃现场已隔离" in notes


def test_identity_breach_terminal_case(matrix: RecoveryMatrix) -> None:
    """恒等式破缺终态：不可续跑/不可写新信封，修复后从破缺前信封重放。"""
    case = _case(matrix, "identity_breach_terminal")
    assert case.verdict == "PASS"
    assert case.differences == ()
    notes = "；".join(case.notes)
    assert "恒等式破缺" in notes
    assert "破缺态写入新信封被拒" in notes
    assert "IdentityTerminatedError" in notes
    assert [point for point, _wm in case.restored_watermarks] == [2]


def test_matrix_covers_all_required_scenarios(matrix: RecoveryMatrix) -> None:
    """场景清单不得缩减：三类恢复 + 成交后水位 + 半提交代 + 破缺终态。"""
    names = [case.name for case in matrix.cases]
    assert names == [
        "multi_source_lot",
        "mid_commit_point",
        "transfer_after_resume",
        "post_fill_watermark",
        "half_commit_crash",
        "identity_breach_terminal",
    ]
    assert matrix.passed is True
    for case in matrix.cases:
        assert case.equal is True, f"{case.name} 存在差异：{case.differences[:3]}"
        assert len(case.envelope_hashes) == 5
        assert [point for point, _sha in case.envelope_hashes] == [1, 2, 3, 4, 5]
        document = case.document()
        assert document["bitwise_equal"] is True
        assert document["difference_count"] == 0


def test_digest_covers_five_ownerships(matrix: RecoveryMatrix) -> None:
    """逐位投影覆盖五项所有权（缺一项即无法声称"完整状态一致"）。"""
    reference: Mapping[str, Any] = matrix.reference_digest
    assert set(reference) == {
        "day_index",
        "trade_date",
        "physical",
        "physical_lot_view",
        "virtual",
        "identity",
        "nav",
        "p4_state",
    }
    physical_view = reference["physical_lot_view"]
    assert set(physical_view) == {"order", "lots", "events", "origin_fingerprint"}
    for sleeve in reference["virtual"]:
        assert set(sleeve) == {
            "sleeve_id",
            "cash",
            "borrowed_credit",
            "total_value",
            "nav_contrib",
            "lots",
        }
    assert set(reference["identity"]) >= {
        "physical_applied_keys",
        "physical_last_fill_date",
        "settlement_keys",
        "batch_ids",
        "price_source_id",
        "price_source_as_of",
    }
    # 合并 NAV 与物理 NAV 位模式必须在参照内自证一致（G4 口径不放松）
    assert reference["nav"]["merged_nav_bits"] == reference["nav"]["physical_nav_bits"]


def test_scenario_rejects_inconsistent_inputs() -> None:
    """算例输入自检：日数/价格/现金合计/目标股数任一不符即拒绝。"""
    from src.lazybull.v2.hosts.recovery_runtime import RecoveryError

    with pytest.raises(RecoveryError, match="严格递增|交易日"):
        RecoveryScenario(trade_dates=("20260106", "20260105"))
    with pytest.raises(RecoveryError, match="价源日期"):
        RecoveryScenario(trade_dates=("20260105", "20260106"))
    with pytest.raises(RecoveryError, match="逐日目标数"):
        RecoveryScenario(targets=((),))
    with pytest.raises(RecoveryError, match="期初现金合计"):
        RecoveryScenario(virtual_cash=(("A", 1.0), ("B", 1.0)))
    with pytest.raises(RecoveryError, match="非负整数"):
        RecoveryScenario(
            targets=((), (("A", -1),), ((),), ((),), ((),)),
        )


def test_restart_rejects_non_sequential_day(tmp_path: Path) -> None:
    """同代推进必须顺序：跳日/回退一律拒绝（禁止用已推进水位重放）。"""
    from src.lazybull.v2.hosts.recovery_runtime import RecoveryError

    scenario = RecoveryScenario()
    runtime = RecoverableRuntime(scenario=scenario, root=tmp_path / "chain", run_id="p65t5-seq")
    runtime.run_and_commit(0)
    with pytest.raises(RecoveryError, match="日序不符"):
        runtime.run_day(2)
    with pytest.raises(RecoveryError, match="日序不符"):
        runtime.run_day(0)


def _restored_lots(runtime: RecoverableRuntime) -> Tuple[Tuple[str, str, int], ...]:
    """当前全袖 ``(袖, 物理来源 lot_id, 股数)`` 明细（I2 口径，排序）。"""
    rows = []
    for sleeve_id, state in sorted(runtime.host.virtual_ledger.snapshot().items()):
        for lot in state.lots:
            rows.append((sleeve_id, lot.source_lot_id, int(lot.shares)))
    return tuple(sorted(rows))


def test_restored_virtual_lots_keep_source_lineage(tmp_path: Path) -> None:
    """恢复后虚拟取得批次的来源谱系与股数逐项保持（不从虚拟侧反推物理成本）。"""
    scenario = RecoveryScenario()
    runtime = RecoverableRuntime(scenario=scenario, root=tmp_path / "chain", run_id="p65t5-lot")
    for index in range(2):
        runtime.run_and_commit(index)
    before = _restored_lots(runtime)
    runtime.restart(2)
    assert _restored_lots(runtime) == before
    assert runtime.host.physical_lot_view.contents() == (
        (scenario.ts_code, "L1", 200, 2005.0),
        (scenario.ts_code, "L2", 200, 2005.0),
    )


def test_quarantine_does_not_reenable_advanced_instance(tmp_path: Path) -> None:
    """隔离残骸不能解除旧实例的提交禁令（T5-Codex-C3）。

    现场：第 1 日已执行、写盘只完成一半（cp2 残骸）；隔离残骸后**同一个**已推进的
    实例不得把内存水位直接写成 cp2——必须先 restart(1) 新建实例并重放。
    """
    scenario = RecoveryScenario()
    runtime = RecoverableRuntime(scenario=scenario, root=tmp_path / "chain", run_id=_RUN_ID)
    runtime.run_and_commit(0)
    runtime.run_day(1)
    runtime.crash_after(1)
    runtime.store.quarantine_incomplete(2, reason="测试：writer crash")
    before_points = runtime.store.commit_points()
    host_before = runtime.host
    with pytest.raises(RecoveryIntegrityError, match="隔离"):
        runtime.commit_day(1)
    assert runtime.host is host_before, "被拒时不得替换宿主"
    assert runtime.store.commit_points() == before_points
    # 合法路径：从提交点 1 新建实例并重放该日之后才允许提交
    runtime.restart(1)
    runtime.run_day(1)
    stored = runtime.commit_day(1)
    assert stored.envelope.commit_point == 2
    assert runtime.store.commit_points() == (1, 2)

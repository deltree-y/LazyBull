# -*- coding: utf-8 -*-
"""P6.5 T5 失败模式与幂等语义：半提交代、破缺终态、水位回退、错误分型、键冲突。

覆盖（任务 §3.2/§3.5/§5 必测断言）：
- 半提交代崩溃：残骸阻断重复提交，只能从提交点前信封新建实例一起重放；
- 恒等式破缺为**终态**：同进程续跑被拒、破缺态写入新信封被拒、修复后从破缺前信封
  重放成功；
- 幂等：同键同内容不重复入账/入账、同键异内容拒绝、跨代重用键拒绝；
- 错误分型：``LedgerError``（重复/乱序入账）与恒等式 ``RuntimeError``（守恒破缺）
  上下文不得混用；信封层 ``StateEnvelopeError``（输入拒绝）与
  ``StateEnvelopeIntegrityError``（完整性终态）同样分离；
- 零副作用：任何拒绝都不产生新信封，上一代信封摘要不变。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.lazybull.v2.core.accounting.envelope_store import (
    CRASHED_DIR_NAME,
    EnvelopeStoreError,
)
from src.lazybull.v2.core.accounting.ledger import BookingEntry, LedgerError
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLotView
from src.lazybull.v2.core.accounting.reconciliation import ReconciliationError
from src.lazybull.v2.core.accounting.state_envelope import (
    StateEnvelopeCorruptError,
    StateEnvelopeError,
    StateEnvelopeIntegrityError,
    sha256_bytes,
)
from src.lazybull.v2.core.accounting.state_owners import (
    StateOwnerError,
    assert_fresh_instance,
    decode_owners,
)
from src.lazybull.v2.core.accounting.virtual_ledger import (
    InternalTransfer,
    LotAllocation,
    PriceSource,
    VirtualLedgerError,
)
from src.lazybull.v2.hosts.order_chain import OrderChainError
from src.lazybull.v2.hosts.recovery_runtime import (
    RecoverableRuntime,
    RecoveryError,
    RecoveryIntegrityError,
    RecoveryScenario,
)
from src.lazybull.v2.hosts.virtual_native_engine import IdentityTerminatedError


def _runtime(tmp_path: Path, *, run_id: str = "p65t5-fm") -> RecoverableRuntime:
    """新建可恢复运行时（未跑任何日）。"""
    return RecoverableRuntime(
        scenario=RecoveryScenario(), root=tmp_path / "chain", run_id=run_id
    )


def _worked(tmp_path: Path, *, days: int, run_id: str = "p65t5-fm") -> RecoverableRuntime:
    """跑到指定天数（含提交）。"""
    runtime = _runtime(tmp_path, run_id=run_id)
    for index in range(days):
        runtime.run_and_commit(index)
    return runtime


def _snapshot(root: Path) -> Dict[str, str]:
    """目录内文件内容摘要（零副作用判据）。"""
    return {
        path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _price_source(runtime: RecoverableRuntime, day_index: int) -> PriceSource:
    """当日同一价源（转账/结算与物理估值同源）。"""
    date = runtime.scenario.date_list()[day_index]
    states = runtime.host.virtual_ledger.snapshot()
    return runtime.host._build_price_source(date, states=states)  # noqa: SLF001


# ────────────────────────── 半提交代 ──────────────────────────


def test_half_commit_residue_blocks_recommit(tmp_path: Path) -> None:
    """残留临时文件（写一半）⇒ 同提交点提交被拒，既有信封摘要不变。"""
    runtime = _worked(tmp_path, days=1)
    store = runtime.store
    runtime.run_day(1)
    temp = store.temp_path_of(2)
    temp.write_bytes(b'{"schema_version": 1, "record_kind": "v2_p65_state_envelope"')
    before = _snapshot(store.root)
    with pytest.raises(EnvelopeStoreError, match="半提交代残留"):
        runtime.commit_day(1)
    assert _snapshot(store.root) == before


def test_half_commit_crash_requires_replay_and_keeps_site(tmp_path: Path) -> None:
    """截断的最终名 = 半提交代：不可读、不可重复提交、隔离后保留残骸与摘要。"""
    runtime = _worked(tmp_path, days=1)
    store = runtime.store
    runtime.run_day(1)
    final_path, temp_path = runtime.crash_after(1)
    original_final = final_path.read_bytes()
    original_temp = temp_path.read_bytes()

    with pytest.raises(StateEnvelopeCorruptError, match="不可解析"):
        store.load_latest()
    with pytest.raises(EnvelopeStoreError, match="已存在"):
        runtime.commit_day(1)
    assert store.commit_points() == (1, 2)
    assert store.load(1).commit_point == 1

    moved = store.quarantine_incomplete(2, reason="单元测试")
    assert len(moved) == 2
    assert all(path.parent.name == CRASHED_DIR_NAME for path in moved)
    assert store.commit_points() == (1,)
    assert store.temp_files() == ()
    kept = {path.name: sha256_bytes(path.read_bytes()) for path in moved}
    assert sha256_bytes(original_final) in kept.values()
    assert sha256_bytes(original_temp) in kept.values()
    notes = list(moved[0].parent.glob("*.note.json"))
    assert len(notes) == 2
    for note in notes:
        document = json.loads(note.read_text(encoding="utf-8"))
        assert document["commit_point"] == 2
        assert document["reason"] == "单元测试"

    # 隔离后可从提交点 1 新建实例重放（同一提交点可再次写入）
    runtime.host = None  # type: ignore[assignment]
    fresh = RecoverableRuntime(
        scenario=runtime.scenario, root=tmp_path / "chain", run_id=runtime.run_id
    )
    fresh.restart(1)
    for index in range(1, len(runtime.scenario.date_list())):
        fresh.run_and_commit(index)
    assert fresh.store.commit_points() == (1, 2, 3, 4, 5)


def test_half_commit_quarantine_refuses_to_overwrite(tmp_path: Path) -> None:
    """崩溃现场已存在同摘要残骸时禁止覆盖（现场只保留首次）。"""
    runtime = _worked(tmp_path, days=1)
    store = runtime.store
    runtime.run_day(1)
    runtime.crash_after(1)
    first = store.quarantine_incomplete(2, reason="第一次")
    assert first
    with pytest.raises(EnvelopeStoreError, match="没有可隔离的残骸"):
        store.quarantine_incomplete(2, reason="空现场")
    # 同一状态再落一次同一份残骸 ⇒ 目标名相同 ⇒ 拒绝覆盖既有现场
    runtime.crash_after(1)
    with pytest.raises(EnvelopeStoreError, match="已存在"):
        store.quarantine_incomplete(2, reason="第二次")


def test_load_latest_never_falls_back(tmp_path: Path) -> None:
    """最大提交点损坏时不做静默回退（必须显式隔离 + 指定旧代恢复）。"""
    runtime = _worked(tmp_path, days=2)
    store = runtime.store
    path = store.path_of(2)
    path.write_bytes(path.read_bytes()[:10])
    with pytest.raises(StateEnvelopeCorruptError):
        store.load_latest()
    assert store.load(1).commit_point == 1
    assert [status for _point, status in store.scan()] == ["ok", "corrupt"]


# ────────────────────────── 破缺终态 ──────────────────────────


def test_identity_breach_is_terminal_and_cannot_commit(tmp_path: Path) -> None:
    """恒等式破缺：实例终态、不可写新信封、上一代摘要不变、同实例续跑被拒。"""
    runtime = _worked(tmp_path, days=2)
    store = runtime.store
    before = _snapshot(store.root)
    runtime.run_day(2)
    dates = runtime.scenario.date_list()
    runtime.host.current_capital = float(runtime.host.current_capital) + 0.01
    with pytest.raises(ReconciliationError) as breach:
        runtime.host._finalize_day(  # noqa: SLF001 - 注入单恒等式破缺
            2,
            dates[2],
            0,
            len(dates),
            3,
            runtime.host._calculate_portfolio_value(dates[2]),  # noqa: SLF001
            list(dates),
            {item: index for index, item in enumerate(dates)},
        )
    assert "I1" in breach.value.identities
    assert runtime.host.is_terminated is True
    with pytest.raises(RecoveryIntegrityError, match="终态"):
        runtime.commit_day(2)
    assert _snapshot(store.root) == before
    with pytest.raises(IdentityTerminatedError):
        runtime.host.run_step(dates[3], list(dates), pd.DataFrame())
    with pytest.raises(IdentityTerminatedError):
        runtime.host.execute_order_chain(
            dates[2], {"A": {runtime.scenario.ts_code: 1}}, batch_id="x"
        )


def test_commit_rejects_poisoned_virtual_ledger(tmp_path: Path) -> None:
    """虚拟账守恒破缺（终态）后禁止写新信封（即便宿主尚未标记终态）。"""
    from src.lazybull.v2.common.types import TradeDate
    from src.lazybull.v2.core.accounting.settlement_kernel import (
        InternalSettlementBatch,
        PhysicalFillFact,
        SleeveIntent,
    )

    runtime = _worked(tmp_path, days=1)
    bad = InternalSettlementBatch(
        batch_id="POISON",
        trade_date=TradeDate(runtime.scenario.date_list()[1].date()),
        intents=(SleeveIntent("A", runtime.scenario.ts_code, "buy", 500, 400),),
        physical_fills=(
            PhysicalFillFact(runtime.scenario.ts_code, "buy", 100, 10.0, 5.0, "L9", "F9"),
        ),
        physical_cash_before=100000.0,
        physical_cash_after=98995.0,
    )
    with pytest.raises(ReconciliationError):
        runtime.host.virtual_ledger.apply_settlement(
            bad,
            price_source=PriceSource(
                as_of=TradeDate(runtime.scenario.date_list()[1].date()),
                source_id="syn:poison",
                prices={runtime.scenario.ts_code: 10.0},
            ),
            physical_view=PhysicalLotView(),  # 缺物理来源 ⇒ I2 破缺 ⇒ 终态
            identity_context=runtime.host._identity_context(  # noqa: SLF001
                runtime.scenario.date_list()[1],
                runtime.host.virtual_ledger.snapshot(),
                "syn:poison",
            ),
        )
    assert runtime.host.virtual_ledger.poisoned_reason is not None
    # 终态账本拒绝新链（前置拒绝，不产生物理成交），也拒绝写入新信封
    with pytest.raises(OrderChainError, match="进入终态"):
        runtime.run_day(1)
    with pytest.raises(RecoveryIntegrityError, match="终态"):
        runtime.commit_day(1)


# ────────────────────────── 错误分型 ──────────────────────────


def test_ledger_error_and_conservation_error_are_separated(tmp_path: Path) -> None:
    """``LedgerError``（重复/乱序入账）与恒等式 ``RuntimeError`` 不得混用。"""
    runtime = _worked(tmp_path, days=4)
    engine = runtime.host
    ledger_state = engine.ledger.export_state()
    assert ledger_state["applied_keys"], "参照算例应已产生物理成交键"
    first_key = ledger_state["applied_keys"][0]
    assert isinstance(first_key, str)

    entry = BookingEntry(
        fill_key=first_key,
        date=runtime.scenario.date_list()[1],
        stock=runtime.scenario.ts_code,
        action="buy",
        cash_delta=0.0,
        trade_record={},
        add_shares=1,
        add_cost_cash=0.0,
    )
    with pytest.raises(LedgerError, match="重复入账"):
        engine.ledger.apply_fills(engine, [entry])

    late = BookingEntry(
        fill_key="20990101|buy|600000.SH|9",
        date=pd.Timestamp("2026-01-06"),
        stock=runtime.scenario.ts_code,
        action="buy",
        cash_delta=0.0,
        trade_record={},
        add_shares=1,
        add_cost_cash=0.0,
    )
    with pytest.raises(LedgerError, match="乱序入账"):
        engine.ledger.apply_fills(engine, [late])

    # 恒等式破缺：RuntimeError 语义但**不是** LedgerError（分型不混用）
    dates = runtime.scenario.date_list()
    engine.current_capital = float(engine.current_capital) + 0.01
    with pytest.raises(ReconciliationError) as breach:
        engine._finalize_day(  # noqa: SLF001
            3,
            dates[3],
            0,
            len(dates),
            4,
            engine._calculate_portfolio_value(dates[3]),  # noqa: SLF001
            list(dates),
            {item: index for index, item in enumerate(dates)},
        )
    assert isinstance(breach.value, RuntimeError)
    assert not isinstance(breach.value, LedgerError)
    assert breach.value.context.stage == "P6.5"
    assert breach.value.context.price_source_id


def test_envelope_error_types_are_separated() -> None:
    """信封层：输入拒绝（ValueError）与完整性终态（RuntimeError）分离。"""
    assert issubclass(StateEnvelopeError, ValueError)
    assert issubclass(StateOwnerError, StateEnvelopeError)
    assert issubclass(StateEnvelopeIntegrityError, RuntimeError)
    assert not issubclass(StateEnvelopeIntegrityError, StateEnvelopeError)
    assert issubclass(StateEnvelopeCorruptError, StateEnvelopeIntegrityError)
    assert issubclass(EnvelopeStoreError, StateEnvelopeError)
    assert issubclass(RecoveryError, ValueError)
    assert issubclass(RecoveryIntegrityError, RuntimeError)


# ────────────────────────── 幂等 ──────────────────────────


def _transfer(runtime: RecoverableRuntime, settlement_id: str, shares: int) -> bool:
    """在当日价源上记一笔内部转账（同键同内容 ⇒ False）。"""
    day = 1
    source = _price_source(runtime, day)
    return runtime.host.virtual_ledger.transfer(
        InternalTransfer(
            settlement_id=settlement_id,
            trade_date=source.as_of,
            ts_code=runtime.scenario.ts_code,
            from_sleeve="A",
            to_sleeve="B",
            price=source.price_of(runtime.scenario.ts_code),
            allocations=(LotAllocation("A:L1", shares),),
        ),
        price_source=source,
    )


def test_same_key_same_content_is_not_double_booked(tmp_path: Path) -> None:
    """同键同内容重复提交：返回 False 且现金/lot 逐位不变。"""
    runtime = _worked(tmp_path, days=1)
    ledger = runtime.host.virtual_ledger
    assert _transfer(runtime, "IDEM", 100) is True
    before = _ledger_projection(ledger)
    assert _transfer(runtime, "IDEM", 100) is False
    assert _ledger_projection(ledger) == before


def _ledger_projection(ledger: Any) -> Dict[str, Any]:
    """全袖现金 + 批次 ``(lot_id, 股数)`` 明细（幂等性判据）。"""
    return {
        sleeve_id: (
            state.cash,
            [(lot.virtual_lot_id, lot.shares) for lot in state.lots],
        )
        for sleeve_id, state in ledger.snapshot().items()
    }


def test_same_key_different_content_is_rejected(tmp_path: Path) -> None:
    """同键异内容：拒绝且不改现金/lot（禁止一个身份承载两套事实）。"""
    runtime = _worked(tmp_path, days=1)
    ledger = runtime.host.virtual_ledger
    assert _transfer(runtime, "CONFLICT", 100) is True
    before = {sid: state.cash for sid, state in ledger.snapshot().items()}
    with pytest.raises(VirtualLedgerError, match="幂等键冲突"):
        _transfer(runtime, "CONFLICT", 50)
    assert {sid: state.cash for sid, state in ledger.snapshot().items()} == before


def test_cross_generation_key_reuse_is_rejected(tmp_path: Path) -> None:
    """跨代重用键：已恢复的代内同名结算身份不得承载新事实。"""
    runtime = _worked(tmp_path, days=2)
    restored = runtime.restart(2)
    assert restored.settlement_keys == ("p65t5-d1#T0",)
    with pytest.raises(VirtualLedgerError, match="已被内部结算批次占用|幂等键冲突"):
        _transfer(runtime, "p65t5-d1#T0", 10)


def test_chain_batch_id_reuse_after_restore_is_rejected(tmp_path: Path) -> None:
    """恢复后同一链批次键不得再次提交，且**同键同内容与同键异内容可区分**。"""
    runtime = _worked(tmp_path, days=2)
    runtime.restart(2)
    day = runtime.scenario.date_list()[1]
    targets = runtime.scenario.targets_of(1)
    with pytest.raises(OrderChainError, match="同一请求"):
        runtime.host.execute_order_chain(day, targets, batch_id="p65t5-d1")
    other = {
        sleeve: {code: shares + 100 for code, shares in mapping.items()}
        for sleeve, mapping in targets.items()
    }
    with pytest.raises(OrderChainError, match="另一套请求"):
        runtime.host.execute_order_chain(day, other, batch_id="p65t5-d1")


def test_chain_fingerprints_do_not_shrink_across_restore(tmp_path: Path) -> None:
    """链级批次指纹跨代只增不减：恢复后新一代必须保留上一代已登记的身份。"""
    runtime = _worked(tmp_path, days=2)
    before = decode_owners(runtime.store.load(2).envelope).identity.chain_batch_fingerprints
    assert [batch_id for batch_id, _fp in before] == ["p65t5-d1"]
    runtime.restart(2)
    runtime.run_and_commit(2)
    after = decode_owners(runtime.store.load(3).envelope).identity.chain_batch_fingerprints
    assert [batch_id for batch_id, _fp in after] == ["p65t5-d1", "p65t5-d2"]
    assert dict(before).items() <= dict(after).items()


def test_stale_generation_commit_is_rejected(tmp_path: Path) -> None:
    """restart(n) 后不跑第 n+1 日直接提交：必须拒绝且目录零变化。

    该场景下同代内的一致性校验全部通过（P4 账本 = 结算身份 = 虚拟账），只有
    "提交日序 + 跨代水位"两道门能发现，因此二者都属于写入路径的必过关。
    """
    runtime = _worked(tmp_path, days=3)
    store = runtime.store
    runtime.restart(2)
    before = _snapshot(store.root)
    with pytest.raises(RecoveryError, match="提交日序不符"):
        runtime.commit_day(3)
    assert _snapshot(store.root) == before
    assert store.commit_points() == (1, 2, 3)


def test_restore_from_old_instance_is_rejected(tmp_path: Path) -> None:
    """恢复只能新建实例：把信封状态写入已演化实例必须被拒（零写入）。"""
    runtime = _worked(tmp_path, days=2)
    restored = decode_owners(runtime.store.load(2).envelope)
    cash = float(runtime.host.current_capital)
    trades = list(runtime.host.trades)
    with pytest.raises(StateEnvelopeError, match="不是新建实例"):
        assert_fresh_instance(runtime.host)
    assert float(runtime.host.current_capital) == cash
    assert list(runtime.host.trades) == trades
    del restored


def test_commit_requires_decodable_owners(tmp_path: Path) -> None:
    """提交前必须证明信封可恢复：不可解码的所有者在落盘前被拒（零副作用）。"""
    runtime = _worked(tmp_path, days=1)
    store = runtime.store
    before = _snapshot(store.root)
    from src.lazybull.v2.core.accounting.state_envelope import (
        MEDIA_TYPE_CANONICAL_JSON,
        OWNER_VIRTUAL_LEDGER,
        build_envelope,
        make_owner,
    )
    from src.lazybull.v2.core.accounting.state_owners import (
        build_identity_owner,
        build_p4_owner,
        build_physical_owner,
    )

    date = runtime.scenario.date_list()[0]
    states = runtime.host.virtual_ledger.snapshot()
    source = runtime.host._build_price_source(date, states=states)  # noqa: SLF001
    owners: Dict[str, Any] = {
        "p4_engine_state": build_p4_owner(runtime.host),
        "physical_lot_view": build_physical_owner(runtime.host.physical_lot_view),
        "virtual_ledger": make_owner(
            OWNER_VIRTUAL_LEDGER, b'{"bogus":1}', MEDIA_TYPE_CANONICAL_JSON
        ),
        "settlement_identity": build_identity_owner(
            host=runtime.host,
            ledger=runtime.host.virtual_ledger,
            price_source_id=source.source_id,
            price_source_as_of=str(source.as_of),
        ),
    }
    envelope = build_envelope(
        run_id=runtime.run_id,
        commit_point=2,
        trade_date=str(source.as_of),
        previous_commit_point=1,
        previous_envelope_sha256=store.load(1).file_sha256,
        state_owners=owners,
    )
    with pytest.raises(StateEnvelopeError, match="虚拟账所有者"):
        store.commit(envelope)
    assert _snapshot(store.root) == before


def test_rejected_commit_keeps_previous_sha(tmp_path: Path) -> None:
    """任一拒绝都不改变上一代信封摘要（"未生成新信封 + 上一代 SHA 不变"）。"""
    runtime = _worked(tmp_path, days=2)
    store = runtime.store
    first = store.load(1)
    second = store.load(2)
    runtime.run_day(2)
    runtime.commit_day(2)  # 正常写出提交点 3
    third = store.load(3)
    for _attempt in range(2):
        with pytest.raises(EnvelopeStoreError, match="已存在"):
            runtime.commit_day(2)  # 提交点 3 已存在：禁止覆盖
    assert store.load(1).file_sha256 == first.file_sha256
    assert store.load(2).file_sha256 == second.file_sha256
    assert store.load(3).file_sha256 == third.file_sha256
    assert store.commit_points() == (1, 2, 3)


def test_restart_older_than_existing_chain_is_explicit(tmp_path: Path) -> None:
    """恢复必须显式指定提交点（不得靠"最新"隐式推断）：可回到更早的完整代。"""
    runtime = _worked(tmp_path, days=3)
    keys_three = runtime.store.load(3).file_sha256
    restored = runtime.restart(1)
    assert restored.commit_point == 1
    assert runtime.store.load(3).file_sha256 == keys_three
    assert runtime.store.commit_points() == (1, 2, 3)


def test_state_owner_errors_listed_for_reviewers() -> None:
    """错误类型清单固定（评审可直接核对分型，不必读实现）。"""
    names: List[str] = [
        cls.__name__
        for cls in (
            LedgerError,
            ReconciliationError,
            StateEnvelopeError,
            StateEnvelopeIntegrityError,
            StateEnvelopeCorruptError,
            EnvelopeStoreError,
            StateOwnerError,
            RecoveryError,
            RecoveryIntegrityError,
            VirtualLedgerError,
            OrderChainError,
        )
    ]
    assert len(set(names)) == len(names)
    assert ReconciliationError.__mro__[1] is RuntimeError
    assert LedgerError.__mro__[1] is RuntimeError

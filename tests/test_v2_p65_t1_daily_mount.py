# -*- coding: utf-8 -*-
"""P6.5 T1：合成宿主 ``_finalize_day`` 逐日挂载。

覆盖：校验先于父类、无成交日仍执行、I1→I2→I3 顺序、失败零副作用
（父类未调用 / 无新快照哈希 / 状态不推进）、父类与 ``run_loop.py`` 零改动守卫、
引擎派生价格源的缺价拒绝。全部为显式合成输入，不读真实配置或行情。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
import pytest

from src.lazybull.universe.base import BasicUniverse
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.core.accounting.engine_state import (
    dumps_blob,
    export_state,
    state_to_blob,
)
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLot, PhysicalLotView
from src.lazybull.v2.core.accounting.reconciliation import ReconciliationError
from src.lazybull.v2.core.accounting.virtual_ledger import (
    PriceSource,
    VirtualLedger,
    VirtualLot,
)
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger
from src.lazybull.v2.core.execution.engine import BacktestEngine
from src.lazybull.v2.core.execution.run_loop import BacktestRunLoopMixin
from src.lazybull.v2.core.signal.base import EqualWeightSignal
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML
from src.lazybull.v2.hosts.virtual_native_engine import (
    IdentityTerminatedError,
    VirtualNativeMergeBacktestEngineML,
)

_D0 = TradeDate(pd.Timestamp("2026-01-05").date())
_D1 = pd.Timestamp("2026-01-06")
_D2 = pd.Timestamp("2026-01-07")
_X = "600000.SH"
_Y = "000001.SZ"
_INITIAL = 100000.0
_PORTFOLIO_VALUE = 104800.0

_REPO_ROOT = Path(__file__).resolve().parents[1]
#: 父类/日循环内容级守卫（LF 归一后的 SHA-256）：变更必须显式更新并同步契约说明
_FROZEN_PARENT_SOURCES = {
    "src/lazybull/v2/core/execution/run_loop.py": (
        "f3919b2b3b5469f28dfd687ebeec67c8d42f4e3d1b97eba21a53adff44d1aec0"
    ),
    "src/lazybull/v2/hosts/native_engine.py": (
        "5e198a00d06735cd34b0ee6e4e73321eb0634eebb80f013fe39d8f8604fa2582"
    ),
}


def _stock_basic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [_X, "000001.SZ"],
            "name": ["甲", "乙"],
            "industry": ["I1", "I2"],
        }
    )


def _base_kwargs() -> Dict[str, Any]:
    return {
        "features_by_date": {},
        "universe": BasicUniverse(_stock_basic_df(), verbose=False),
        "signal": EqualWeightSignal(top_n=5),
        "initial_capital": _INITIAL,
        "cost_model": CostModel(
            commission_rate=0.0003, min_commission=5.0, stamp_tax=0.0005, slippage=0.0
        ),
        "verbose": False,
        "portfolio_merger": SingleSleevePortfolioMerger(),
    }


def _position(shares: int) -> Dict[str, Any]:
    return {
        "shares": shares,
        "buy_date": pd.Timestamp("2025-12-31"),
        "buy_cost_cash": shares * 10.0,
        "buy_trade_price": 10.0,
    }


def _seed_engine(engine: Any) -> Any:
    """装配 400 股物理持仓 + 现金 100,000，并接线两日收盘价索引（X/Y 均 12.0）。"""
    engine.positions = {_X: _position(400)}
    engine.current_capital = _INITIAL
    engine._emit_immediate_log = lambda *args, **kwargs: None
    engine._prepare_price_index(_price_data())
    return engine


def _ledger() -> VirtualLedger:
    ledger = VirtualLedger()
    ledger.open_account("A", 50000.0, [VirtualLot("A:L1", _X, "L1", 400, _D0, 10.0)])
    ledger.open_account("B", 50000.0)
    return ledger


def _view() -> PhysicalLotView:
    return PhysicalLotView([PhysicalLot(_X, "L1", 400)])


def _provider(
    prices: Optional[Dict[str, float]] = None
) -> Callable[[pd.Timestamp], PriceSource]:
    table = {_X: 12.0} if prices is None else prices

    def build(date: pd.Timestamp) -> PriceSource:
        return PriceSource(
            as_of=TradeDate(pd.Timestamp(date).date()),
            source_id="syn:t1",
            prices=table,
        )

    return build


def _make_virtual_engine(
    *,
    ledger: Optional[VirtualLedger] = None,
    view: Optional[PhysicalLotView] = None,
    provider: Optional[Callable[[pd.Timestamp], PriceSource]] = _provider(),
    run_id: str = "T1R1",
) -> VirtualNativeMergeBacktestEngineML:
    engine = VirtualNativeMergeBacktestEngineML(
        **_base_kwargs(),
        virtual_ledger=ledger or _ledger(),
        physical_lot_view=view or _view(),
        price_source_provider=provider,
        run_id=run_id,
    )
    return _seed_engine(engine)


def _make_parent_engine() -> NativeMergeBacktestEngineML:
    return _seed_engine(NativeMergeBacktestEngineML(**_base_kwargs()))


def _finalize(engine: Any, *, date: pd.Timestamp = _D1, value: float = _PORTFOLIO_VALUE) -> None:
    engine._finalize_day(0, date, 0, 1, 1, value, [date], {date: 0})


def _snapshot_hash(engine: Any) -> str:
    blob = state_to_blob(export_state(engine))
    return hashlib.sha256(dumps_blob(blob)).hexdigest()


def _price_data() -> pd.DataFrame:
    """两日合成行情（X/Y 收盘价均为 12；本用例不触发信号生成与买卖）。"""
    return pd.DataFrame(
        {
            "ts_code": [_X, _Y, _X, _Y],
            "trade_date": [_D1, _D1, _D2, _D2],
            "close": [12.0, 12.0, 12.0, 12.0],
        }
    )


def _wire_daily_loop(engine: Any, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """按 ``run_step`` 前提装配日循环接线（日历/信号日/总天数/价格索引）。"""
    engine._trade_date_index = {date: idx for idx, date in enumerate(dates)}
    engine._signal_dates = {}
    engine._run_total_days = len(dates)
    engine._cycle_anchor_idx = 0
    price_data = _price_data()
    engine._prepare_price_index(price_data)
    return price_data


def test_run_step_drives_gate_once_per_day() -> None:
    """驱动级证据：日循环每天恰好触达一次挂载点（不是只直接调 _finalize_day）。"""
    dates = [_D1, _D2]
    engine = _make_virtual_engine()
    price_data = _wire_daily_loop(engine, dates)

    engine.run_step(_D1, dates, price_data)
    engine.run_step(_D2, dates, price_data)

    assert engine.identity_check_count == 2
    assert [report.identities for report in engine.identity_report_snapshot()] == [
        ("I1", "I2", "I3"),
        ("I1", "I2", "I3"),
    ]
    assert len(engine.portfolio_values) == 2


def test_run_step_stops_on_second_day_breakage() -> None:
    dates = [_D1, _D2]
    ledger = _ledger()
    engine = _make_virtual_engine(ledger=ledger)
    price_data = _wire_daily_loop(engine, dates)
    engine.run_step(_D1, dates, price_data)
    assert engine.identity_check_count == 1

    ledger.open_account("C", 100.0)  # 次日 I1 破缺
    with pytest.raises(ReconciliationError) as caught:
        engine.run_step(_D2, dates, price_data)

    assert caught.value.identity == "I1"
    assert engine.identity_check_count == 1
    assert len(engine.portfolio_values) == 1
    assert engine.is_terminated

    # 终态实例：同日重试与下一日推进都被拒绝（不产生新证据）
    with pytest.raises(IdentityTerminatedError):
        engine.run_step(_D2, dates, price_data)
    with pytest.raises(IdentityTerminatedError):
        _finalize(engine, date=_D2)
    assert engine.identity_check_count == 1
    assert len(engine.portfolio_values) == 1


def test_breakage_terminates_instance_and_evidence_reset_does_not_lift_it() -> None:
    dates = [_D1, _D2]
    engine = _make_virtual_engine()
    price_data = _wire_daily_loop(engine, dates)

    with pytest.raises(ReconciliationError) as caught:
        _finalize(engine, value=104700.0)  # 仅 I3 破缺（物理总值与价源不一致）
    assert caught.value.identities == ("I3",)
    assert engine.is_terminated
    assert engine.identity_check_count == 0
    before_hash = _snapshot_hash(engine)

    with pytest.raises(IdentityTerminatedError) as terminated:
        engine.run_step(_D2, dates, price_data)
    assert terminated.value.original.identities == ("I3",)
    assert _snapshot_hash(engine) == before_hash

    engine.reset_identity_evidence()
    with pytest.raises(IdentityTerminatedError):
        engine.run_step(_D2, dates, price_data)
    assert engine.identity_check_count == 0


def test_terminated_instance_rejects_continuous_run_without_mutating_state() -> None:
    """终态后调用继承的 ``run()`` 必须在父类改写日历/排期/价源现场之前拒绝。"""
    dates = [_D1, _D2]
    ledger = _ledger()
    engine = _make_virtual_engine(ledger=ledger)
    price_data = _wire_daily_loop(engine, dates)
    ledger.open_account("C", 100.0)  # 首日 I1 破缺
    with pytest.raises(ReconciliationError):
        engine.run_step(_D1, dates, price_data)
    assert engine.is_terminated

    blob_before = _snapshot_hash(engine)
    signals_before = dict(engine._signal_dates)
    calendar_before = dict(engine._trade_date_index)

    with pytest.raises(IdentityTerminatedError):
        engine.run(_D2, _D2, dates, price_data)

    assert _snapshot_hash(engine) == blob_before
    assert dict(engine._signal_dates) == signals_before
    assert dict(engine._trade_date_index) == calendar_before
    assert engine.identity_check_count == 0
    assert engine.portfolio_values == []


def test_explicit_price_source_must_match_physical_prices_per_stock() -> None:
    """组合总值恰好相同的逐股错价（X=11/Y=13 vs 12/12）必须在写日终前拒绝。"""
    ledger = VirtualLedger()
    ledger.open_account("A", 50000.0, [VirtualLot("AX", _X, "LX", 100, _D0, 10.0)])
    ledger.open_account("B", 50000.0, [VirtualLot("BY", _Y, "LY", 100, _D0, 10.0)])
    view = PhysicalLotView([PhysicalLot(_X, "LX", 100), PhysicalLot(_Y, "LY", 100)])
    engine = _make_virtual_engine(
        ledger=ledger,
        view=view,
        provider=_provider({_X: 11.0, _Y: 13.0}),
    )
    engine.positions = {_X: _position(100), _Y: _position(100)}

    with pytest.raises(ReconciliationError, match="不同源") as caught:
        _finalize(engine, value=102400.0)  # 51100 + 51300 恰好等于物理总值

    assert caught.value.identities == ("I3",)
    assert engine.portfolio_values == []
    assert engine.identity_check_count == 0


def test_explicit_price_source_requires_engine_price_index() -> None:
    """无引擎价格索引时无法证明同源，必须拒绝而不是静默通过。"""
    engine = _make_virtual_engine(provider=_provider())
    engine.trade_price_index = None

    with pytest.raises(ReconciliationError, match="价格索引未就绪") as caught:
        _finalize(engine)
    assert caught.value.identities == ("I3",)
    assert engine.portfolio_values == []


def test_parent_sources_are_content_frozen() -> None:
    """内容级守卫：父类日循环与原生宿主文件内容不得被 T1 改动。"""
    for relative, expected in _FROZEN_PARENT_SOURCES.items():
        text = (_REPO_ROOT / relative).read_text(encoding="utf-8").replace("\r\n", "\n")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert actual == expected, (
            f"{relative} 内容已变更：确认不属于 T1 改动后，显式更新 _FROZEN_PARENT_SOURCES"
        )


def test_reset_identity_evidence_clears_instance_state() -> None:
    engine = _make_virtual_engine()
    _finalize(engine)
    assert engine.identity_check_count == 1

    engine.reset_identity_evidence()
    assert engine.identity_check_count == 0
    assert list(engine.identity_report_snapshot()) == []


def test_gate_runs_before_parent_and_records_report() -> None:
    engine = _make_virtual_engine()
    _finalize(engine)

    assert engine.identity_check_count == 1
    assert engine.portfolio_values == [
        {
            "date": _D1,
            "portfolio_value": _PORTFOLIO_VALUE,
            "capital": _INITIAL,
            "market_value": _PORTFOLIO_VALUE - _INITIAL,
        }
    ]
    report = engine.identity_reports[0]
    assert report.identities == ("I1", "I2", "I3")
    assert report.price_source_id == "syn:t1"
    assert report.run_id == "T1R1"
    assert engine.identity_report_snapshot()[0] is report


def test_no_trade_day_still_runs_all_identities() -> None:
    engine = _make_virtual_engine()
    assert engine.trades == []
    _finalize(engine)

    report = engine.identity_reports[0]
    assert report.identities == ("I1", "I2", "I3")
    assert report.results[1].right == 400
    assert isinstance(report.results[1].right, int)
    assert engine.identity_check_count == 1
    assert engine.trades == []


def test_failure_blocks_parent_and_keeps_snapshot_unchanged() -> None:
    """破缺当日不追加日终快照/证据、不推进；当日成交写入不回滚（见宿主 docstring）。"""
    ledger = VirtualLedger()
    ledger.open_account("A", 51000.0, [VirtualLot("A:L1", _X, "L1", 400, _D0, 10.0)])
    ledger.open_account("B", 50000.0)
    engine = _make_virtual_engine(ledger=ledger)
    before = _snapshot_hash(engine)

    with pytest.raises(ReconciliationError) as caught:
        _finalize(engine)

    assert caught.value.identity == "I1"
    assert engine.portfolio_values == []
    assert engine.holdings_snapshots == []
    assert engine.identity_reports == []
    assert engine.identity_check_count == 0
    assert _snapshot_hash(engine) == before


def test_mount_reports_all_breakage_identities() -> None:
    """联动故障（I1/I2/I3 同时破缺）必须在写日终前报出全部编号。"""
    ledger = VirtualLedger()
    ledger.open_account("A", 1.0, [VirtualLot("A:L9", _X, "L9", 400, _D0, 10.0)])
    ledger.open_account("B", 50000.0)
    engine = _make_virtual_engine(ledger=ledger)

    with pytest.raises(ReconciliationError) as caught:
        _finalize(engine)

    error = caught.value
    assert error.identities == ("I1", "I2", "I3")
    message = str(error)
    for identity in ("I1", "I2", "I3"):
        assert f"[{identity}]" in message
    assert engine.portfolio_values == []
    assert engine.identity_check_count == 0
    assert engine.is_terminated


def test_i3_only_failure_from_inconsistent_physical_value() -> None:
    engine = _make_virtual_engine()

    with pytest.raises(ReconciliationError) as caught:
        _finalize(engine, value=104700.0)

    assert caught.value.identity == "I3"
    assert engine.portfolio_values == []
    assert engine.identity_check_count == 0


def test_parent_behaviour_is_identical_when_identities_pass() -> None:
    parent = _make_parent_engine()
    child = _make_virtual_engine()
    assert child.portfolio_values == [] == parent.portfolio_values

    _finalize(parent)
    _finalize(child)

    assert child.portfolio_values == parent.portfolio_values
    assert child.current_capital == parent.current_capital == _INITIAL
    assert child.trades == parent.trades == []
    assert child.holdings_snapshots == parent.holdings_snapshots == []


def test_parent_and_run_loop_are_unmodified() -> None:
    assert "_finalize_day" not in NativeMergeBacktestEngineML.__dict__
    assert NativeMergeBacktestEngineML._finalize_day is BacktestRunLoopMixin._finalize_day
    assert BacktestEngine._finalize_day is BacktestRunLoopMixin._finalize_day
    assert VirtualNativeMergeBacktestEngineML._finalize_day is not (
        BacktestRunLoopMixin._finalize_day
    )
    assert issubclass(VirtualNativeMergeBacktestEngineML, NativeMergeBacktestEngineML)


def test_second_day_breakage_stops_after_first_success() -> None:
    ledger = _ledger()
    engine = _make_virtual_engine(ledger=ledger)
    _finalize(engine, date=_D1)
    assert engine.identity_check_count == 1

    ledger.open_account("C", 100.0)
    with pytest.raises(ReconciliationError) as caught:
        _finalize(engine, date=_D2)

    assert caught.value.identity == "I1"
    assert engine.identity_check_count == 1
    assert len(engine.portfolio_values) == 1
    assert engine.portfolio_values[0]["date"] == _D1


def test_engine_derived_price_source_uses_close_price() -> None:
    engine = _make_virtual_engine(provider=None)
    index = pd.MultiIndex.from_tuples([(_D1, _X)], names=["trade_date", "ts_code"])
    engine.trade_price_index = pd.Series([12.0], index=index)
    _finalize(engine)
    assert engine.identity_reports[0].price_source_id == "engine_trade_price:20260106"


def test_engine_derived_price_source_rejects_missing_price() -> None:
    engine = _make_virtual_engine(provider=None)
    engine.trade_price_index = pd.Series(
        dtype=float,
        index=pd.MultiIndex.from_tuples([], names=["trade_date", "ts_code"]),
    )

    with pytest.raises(ReconciliationError, match="缺少") as caught:
        _finalize(engine)

    assert caught.value.identity == "I3"
    assert engine.portfolio_values == []


def test_constructor_rejects_non_ledger_types() -> None:
    with pytest.raises(TypeError, match="VirtualLedger"):
        VirtualNativeMergeBacktestEngineML(
            **_base_kwargs(),
            virtual_ledger=object(),
            physical_lot_view=_view(),
        )
    with pytest.raises(TypeError, match="PhysicalLotView"):
        VirtualNativeMergeBacktestEngineML(
            **_base_kwargs(),
            virtual_ledger=_ledger(),
            physical_lot_view=object(),
        )

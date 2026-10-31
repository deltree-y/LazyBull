# -*- coding: utf-8 -*-
"""v2 P4 T1 引擎状态（EngineState）契约测试（tests/ 全量豁免复杂度）。

锁定 D2 状态外置接缝（§7 测试矩阵「EngineState 往返」行 + T1 验收增项②④）：
- export→restore→export **字节一致**（合成区间多断点：首日/中途/末日）；
- **fail-fast**：缺字段 / 多字段 / schema 版本不符 / 子对象键集不符 /
  子对象形态（None 与否）与引擎配置不符 ⇒ ``EngineStateError``；
- 恢复逐字段保真（每个快照字段恢复到全新引擎后与源逐值相等）；
- 字段归类注册表（FIELD_CLASSIFICATION）：AST 枚举口径全量覆盖 +
  类别闭集 + 快照字段 ≡ evolving ∪ sub_object（AST 枚举+人工复核的机器锁定）；
- 快照与引擎脱钩（导出后引擎演化不污染快照；恢复后快照不受引擎后续影响）。
"""

import copy
from pathlib import Path
from typing import Dict, Set

import pandas as pd
import pytest

from scripts.v2_p4.t05b_spike_runner import (
    _DeterministicSignal,
    _make_engine_factory,
    _make_price_data,
    synthetic_cost_settings,
)
from src.lazybull.v2.core.accounting.engine_state import (
    ENGINE_STATE_SCHEMA_VERSION,
    FIELD_CLASSIFICATION,
    SNAPSHOT_FIELDS,
    SNAPSHOT_SUB_OBJECTS,
    EngineStateError,
    dumps_blob,
    export_state,
    loads_blob,
    restore_state,
    state_from_blob,
    state_to_blob,
)
from src.lazybull.v2.core.accounting.ledger import LedgerError

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _synthetic_costs():
    with synthetic_cost_settings():
        yield


def _make_engine(n_days: int = 12):
    dates = list(pd.bdate_range("2026-01-05", periods=n_days))
    stocks = ["A", "B"]
    signal = _DeterministicSignal(2, {}, [("A", 0.9), ("B", 0.8)])
    factory = _make_engine_factory(stocks, signal)
    closes = {s: [10.0 * (1 + 0.001 * i) for i in range(n_days)] for s in stocks}
    price_data = _make_price_data(dates, closes)
    return dates, factory, price_data


def _wired_engine(factory, dates, price_data, with_schedule: bool = True):
    engine = factory()
    engine._trade_date_index = {d: i for i, d in enumerate(dates)}
    engine._prepare_price_index(price_data)
    engine.price_data_cache = price_data
    engine._run_total_days = len(dates)
    if with_schedule:
        engine._signal_dates = engine._get_rebalance_dates(dates)
    return engine


def _roundtrip_bytes(blob: Dict) -> bytes:
    """blob → 恢复进全新引擎 → 再导出 → 字节。"""
    dates, factory, price_data = _make_engine()
    probe = _wired_engine(factory, dates, price_data)
    restore_state(probe, state_from_blob(loads_blob(dumps_blob(blob))))
    return dumps_blob(state_to_blob(export_state(probe)))


class TestRoundTripByteIdentical:
    def test_initial_state_roundtrip(self):
        dates, factory, price_data = _make_engine()
        engine = _wired_engine(factory, dates, price_data)
        blob = state_to_blob(export_state(engine))
        assert dumps_blob(blob) == _roundtrip_bytes(blob)

    def test_mid_run_state_roundtrip(self):
        """演化后的状态（含持仓/成交/排期改写）往返字节一致。"""
        dates, factory, price_data = _make_engine(12)
        engine = _wired_engine(factory, dates, price_data)
        for date in dates[:6]:  # 驱动 6 日（跨信号日 0/5，含持仓与成交）
            engine.run_step(date, dates, price_data)
        blob = state_to_blob(export_state(engine))
        assert blob["fields"]["trades"], "前提：6 日内已产生成交"
        assert dumps_blob(blob) == _roundtrip_bytes(blob)

    def test_roundtrip_after_second_restore(self):
        """多断点：恢复态继续演化一天后再往返仍字节一致。"""
        dates, factory, price_data = _make_engine(12)
        engine = _wired_engine(factory, dates, price_data)
        for date in dates[:5]:
            engine.run_step(date, dates, price_data)
        blob1 = state_to_blob(export_state(engine))
        engine2 = _wired_engine(factory, dates, price_data)
        restore_state(engine2, state_from_blob(loads_blob(dumps_blob(blob1))))
        engine2.run_step(dates[5], dates, price_data)  # 断点后续跑
        blob2 = state_to_blob(export_state(engine2))
        assert dumps_blob(blob2) == _roundtrip_bytes(blob2)


class TestRestoreFailFast:
    def _blob(self) -> Dict:
        dates, factory, price_data = _make_engine()
        engine = _wired_engine(factory, dates, price_data)
        return state_to_blob(export_state(engine))

    def test_missing_field_rejected(self):
        blob = self._blob()
        blob["fields"].pop("positions")
        with pytest.raises(EngineStateError, match="缺"):
            state_from_blob(blob)

    def test_extra_field_rejected(self):
        blob = self._blob()
        blob["fields"]["unknown_field"] = 1
        with pytest.raises(EngineStateError, match="多"):
            state_from_blob(blob)

    def test_schema_version_mismatch_rejected(self):
        blob = self._blob()
        blob["schema_version"] = ENGINE_STATE_SCHEMA_VERSION + 1
        with pytest.raises(EngineStateError, match="schema 版本不符"):
            state_from_blob(blob)

    def test_malformed_blob_rejected(self):
        with pytest.raises(EngineStateError):
            state_from_blob({"fields": {}})
        with pytest.raises(EngineStateError):
            state_from_blob({"schema_version": ENGINE_STATE_SCHEMA_VERSION})
        with pytest.raises(EngineStateError):
            state_from_blob("not-a-dict")

    def test_sub_object_key_set_mismatch_rejected(self):
        blob = self._blob()
        blob["fields"]["pending_order_manager"] = {"pending_orders": {}}  # 缺三计数器
        with pytest.raises(EngineStateError, match="pending_order_manager"):
            state_from_blob(blob)
        blob2 = self._blob()
        blob2["fields"]["ledger"] = {"applied_keys": []}  # 缺 last_fill_date
        with pytest.raises(EngineStateError, match="ledger"):
            state_from_blob(blob2)

    def test_sub_object_shape_mismatch_with_engine_config_rejected(self):
        """子对象形态（None 与否）与引擎配置不符 ⇒ 恢复拒绝。"""
        dates, factory, price_data = _make_engine()
        engine = _wired_engine(factory, dates, price_data)
        blob = state_to_blob(export_state(engine))
        assert blob["fields"]["pending_order_manager"] is not None
        # 篡改为 None（等价于引擎 enable_pending_order=False 的快照）
        blob["fields"]["pending_order_manager"] = None
        engine2 = _wired_engine(factory, dates, price_data)
        with pytest.raises(EngineStateError, match="形态不符"):
            restore_state(engine2, state_from_blob(blob))

    def test_rejected_restore_leaves_engine_untouched(self):
        """R2-P4-T1-05：恢复被拒绝 ⇒ 目标引擎**逐位不变**（不产生混合状态）。"""
        dates, factory, price_data = _make_engine()
        engine = _wired_engine(factory, dates, price_data)
        # 目标引擎先演化出非平凡状态
        for date in dates[:6]:
            engine.run_step(date, dates, price_data)
        before_bytes = dumps_blob(state_to_blob(export_state(engine)))
        # 构造一个会被拒绝的快照（形态不符：引擎有 manager 而快照 None）
        src = _wired_engine(factory, dates, price_data)
        blob = state_to_blob(export_state(src))
        blob["fields"]["pending_order_manager"] = None
        with pytest.raises(EngineStateError):
            restore_state(engine, state_from_blob(blob))
        after_bytes = dumps_blob(state_to_blob(export_state(engine)))
        assert after_bytes == before_bytes, "恢复被拒绝后目标引擎状态必须逐位不变"

    def test_malformed_sub_object_values_rejected_and_untouched(self):
        """复审处置（r1-P2-B / R2-P4-T1-05 保留项）：子对象**值**非法 ⇒
        恢复拒绝且目标引擎逐位不变（校验全部先于写入；覆盖账本/管理器/监控器）。"""
        from src.lazybull.v2.common.rules.stop_loss import StopLossConfig

        dates, factory, price_data = _make_engine()
        # 源引擎启用止损（覆盖监控器值校验路径；两侧配置一致）
        factory_sl = _make_engine_factory(
            ["A", "B"],
            _DeterministicSignal(2, {}, [("A", 0.9), ("B", 0.8)]),
            stop_loss_config=StopLossConfig(enabled=True, drawdown_pct=5.0),
        )
        src = _wired_engine(factory_sl, dates, price_data)
        engine = _wired_engine(factory_sl, dates, price_data)
        for date in dates[:6]:
            engine.run_step(date, dates, price_data)
        before_bytes = dumps_blob(state_to_blob(export_state(engine)))
        for tamper in (
            lambda f: f["ledger"].update(applied_keys=0),  # 非 list
            lambda f: f["ledger"].update(applied_keys=["a", "a"]),  # 重复键
            lambda f: f["ledger"].update(last_fill_date="20260105"),  # 水位类型
            lambda f: f["pending_order_manager"].update(total_added="x"),  # 计数器类型
            lambda f: f["pending_order_manager"].update(pending_orders=[]),  # 队列类型
            lambda f: f["stop_loss_monitor"].update(consecutive_limit_down_days=[]),
        ):
            blob = state_to_blob(export_state(src))
            tamper(blob["fields"])
            with pytest.raises((EngineStateError, LedgerError)):
                restore_state(engine, state_from_blob(blob))
            after_bytes = dumps_blob(state_to_blob(export_state(engine)))
            assert after_bytes == before_bytes, "子对象值非法被拒绝后目标引擎必须逐位不变"

    def test_field_type_validation(self):
        """r1-P3-4：顶层字段类型非法 ⇒ fail-fast（含 bool 不是合法 int）。"""
        blob = self._blob()
        blob["fields"]["positions"] = ["not", "a", "dict"]
        with pytest.raises(EngineStateError, match="类型非法"):
            state_from_blob(blob)
        blob2 = self._blob()
        blob2["fields"]["_cycle_anchor_idx"] = True  # bool 不是合法 int
        with pytest.raises(EngineStateError, match="类型非法"):
            state_from_blob(blob2)


class TestRestoreFidelity:
    def test_every_snapshot_field_restored(self):
        """恢复逐字段保真：每个快照字段恢复到全新引擎后与源逐值相等。"""
        dates, factory, price_data = _make_engine(12)
        engine = _wired_engine(factory, dates, price_data)
        for date in dates[:6]:
            engine.run_step(date, dates, price_data)
        blob = state_to_blob(export_state(engine))
        engine2 = _wired_engine(factory, dates, price_data, with_schedule=False)
        restore_state(engine2, state_from_blob(loads_blob(dumps_blob(blob))))
        for name in SNAPSHOT_FIELDS:
            assert dumps_blob(engine2.__dict__[name]) == dumps_blob(
                blob["fields"][name]
            ), f"字段 {name} 恢复不保真"
        # 子对象
        assert engine2.pending_order_manager.total_added == (
            blob["fields"]["pending_order_manager"]["total_added"]
        )
        assert engine2.live_anchor_state is None  # 回放形态恒 None

    def test_snapshot_detached_both_directions(self):
        """脱钩双向性：导出后源引擎演化不污染快照；恢复后新引擎演化不污染快照。"""
        dates, factory, price_data = _make_engine()
        engine = _wired_engine(factory, dates, price_data)
        state = export_state(engine)
        engine.run_step(dates[0], dates, price_data)  # 源引擎演化
        assert state.current_capital == 1_000_000.0
        engine2 = _wired_engine(factory, dates, price_data, with_schedule=False)
        restore_state(engine2, state)
        engine2.run_step(dates[0], dates, price_data)  # 新引擎演化
        assert state.current_capital == 1_000_000.0
        assert state.trades == []


class TestFieldClassification:
    def test_categories_closed_set(self):
        allowed = {"evolving", "sub_object", "config", "wiring", "daily_transient"}
        for attr, (category, note) in FIELD_CLASSIFICATION.items():
            assert category in allowed, f"{attr} 类别非法: {category}"
            assert note, f"{attr} 缺复核注记"

    def test_snapshot_fields_exactly_evolving_plus_sub_object(self):
        evolving = {k for k, (c, _n) in FIELD_CLASSIFICATION.items() if c == "evolving"}
        sub_object = {k for k, (c, _n) in FIELD_CLASSIFICATION.items() if c == "sub_object"}
        assert set(SNAPSHOT_FIELDS) == evolving
        assert set(SNAPSHOT_SUB_OBJECTS) == sub_object

    def test_classification_covers_all_engine_attr_writes(self):
        """AST 枚举口径机器锁定：引擎模块（core/execution + core/accounting）
        的引擎属性写点（Backtest* 类 + 模块级 engine/self 形参函数）必须全部
        在归类注册表内（新增写点未归类 ⇒ 本测试失败）。"""
        from scripts.v2_p4 import t1_ast_enum

        rows = t1_ast_enum.scan(_REPO_ROOT)
        covered: Set[str] = set()
        for row in rows:
            module = row["module"]
            if not (
                "v2/core/execution/" in module
                or "v2/core/accounting/" in module
            ):
                continue
            cls = row["class"]
            if cls and not cls.startswith("Backtest"):
                continue  # 子对象类（PendingOrderManager 等）内部属性不归引擎面
            covered.add(row["attr"])
        missing = covered - set(FIELD_CLASSIFICATION)
        assert missing == set(), f"引擎属性写点未归类: {sorted(missing)}"

    def test_config_and_wiring_not_in_snapshot(self):
        """配置型/接线型/日界暂态字段一律不入快照清单。"""
        for attr, (category, _note) in FIELD_CLASSIFICATION.items():
            if category in ("config", "wiring", "daily_transient"):
                assert attr not in SNAPSHOT_FIELDS
                assert attr not in SNAPSHOT_SUB_OBJECTS

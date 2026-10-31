# -*- coding: utf-8 -*-
"""v2 P4 T1 账本（Ledger）契约测试（tests/ 全量豁免复杂度）。

锁定 D2 调用链冻结与 §7 测试矩阵「入账唯一入口」行：
- 唯一入账：v2 core 内直接改现金/持仓的写点只存在于 ``ledger.py``
  （AST 静态扫描；state.py 构造初始化为例外登记）；
- 执行器产出 Fill 前后账户不变；只有 Ledger 入账改变现金/股数/成本
  （spy 账本 + 逐日现金恒等式核验）；
- 重复入账/乱序入账拒绝；指令形状与前置持仓非法拒绝；
- 账本状态随 EngineState 快照导出/恢复（恢复后重复提交仍拒绝）；
- decide 无副作用：决策相位（_generate_signal/_check_and_sell）不改变
  账户（现金/持仓及嵌套对象），同输入重复调用输出一致。
"""

import ast
import copy
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from src.lazybull.v2.core.accounting.ledger import BookingEntry, Ledger, LedgerError

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ── 静态扫描：直接改账户写点只允许在 ledger.py（+ state.py 构造初始化登记例外）──

_ACCOUNT_WRITE_MODULES_EXCEPT = {
    # 唯一入账入口本体
        "src/lazybull/v2/core/accounting/ledger.py",
}
_CONSTRUCTION_EXCEPTION = {
    # 构造期初始化（init_engine_state 族；非运行期入账）
    ("src/lazybull/v2/core/accounting/state.py", "current_capital"),
    ("src/lazybull/v2/core/accounting/state.py", "positions"),
}


def _iter_account_writes(path: Path) -> List[str]:
    """枚举对 (self|engine).current_capital / .positions 的直接写点。

    覆盖：赋值/注解赋值/增减赋值/删除（属性级与项级）+ 容器方法调用
    （update/pop/clear/setdefault/append 等，r1-P2-3 处置）。**边界登记**：
    经局部别名的写（如 ``info = positions[stock]; info[...] = ...``）不在
    静态覆盖内——当前唯一别名写 = ``last_known_price`` 估值缓存（D10 登记例外）。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: List[str] = []

    def _base_name(node: ast.AST) -> str:
        while isinstance(node, ast.Subscript):
            node = node.value
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        return ""

    _CONTAINER_WRITE_METHODS = {
        "update", "pop", "popitem", "clear", "setdefault", "append",
        "extend", "insert", "add", "discard", "remove",
    }

    for node in ast.walk(tree):
        targets: List[ast.AST] = []
        kind = ""
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            kind = "assign"
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            kind = "ann_assign"
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
            kind = "aug_assign"
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
            kind = "delete"
        for target in targets:
            name = _base_name(target)
            if name in ("self.current_capital", "engine.current_capital"):
                offenders.append(f"{path.name}:{node.lineno} {kind} {name}")
            if name in ("self.positions", "engine.positions") and (
                isinstance(target, ast.Subscript) or kind == "delete"
            ):
                offenders.append(f"{path.name}:{node.lineno} {kind} {name}")
        # 容器方法调用写点（self.positions.update(...) 等）
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            recv = node.func.value
            name = _base_name(recv)
            if name in ("self.positions", "engine.positions") and (
                node.func.attr in _CONTAINER_WRITE_METHODS
            ):
                offenders.append(
                    f"{path.name}:{node.lineno} method_call:{node.func.attr} {name}"
                )
    return offenders


class TestUniqueBookingEntry:
    """入账唯一入口（静态扫描 + 构造例外登记）。"""

    def test_no_direct_account_mutation_outside_ledger(self):
        offenders: List[str] = []
        for path in (_REPO_ROOT / "src" / "lazybull" / "v2" / "core").rglob("*.py"):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in _ACCOUNT_WRITE_MODULES_EXCEPT:
                continue
            for hit in _iter_account_writes(path):
                attr = hit.rsplit(" ", 1)[-1].split(".", 1)[-1]
                if (rel, attr) in _CONSTRUCTION_EXCEPTION:
                    continue
                offenders.append(f"{rel}:{hit}")
        assert offenders == [], f"v2 core 存在绕过 Ledger 的直接改账户写点: {offenders}"


class _StubEngine:
    """最小引擎桩（账本落账载体：现金/持仓/成交日志）。"""

    def __init__(self, capital: float = 1_000_000.0) -> None:
        self.current_capital = capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []


def _buy_entry(
    date: str,
    stock: str,
    shares: int,
    price: float,
    seq: int,
    cost: float = 5.0,
) -> BookingEntry:
    total = shares * price + cost
    return BookingEntry(
        fill_key=f"{date}|buy|{stock}|{seq}",
        date=pd.Timestamp(date),
        stock=stock,
        action="buy",
        cash_delta=-total,
        trade_record={"date": pd.Timestamp(date), "stock": stock, "action": "buy"},
        new_position={
            "shares": shares,
            "buy_date": pd.Timestamp(date),
            "signal_date": pd.Timestamp(date),
            "buy_trade_price": price,
            "buy_pnl_price": price,
            "buy_cost_cash": total,
        },
    )


def _sell_entry(
    date: str, stock: str, shares: int, price: float, seq: int, remaining: int = 0
) -> BookingEntry:
    amount = shares * price
    return BookingEntry(
        fill_key=f"{date}|sell|{stock}|{seq}",
        date=pd.Timestamp(date),
        stock=stock,
        action="sell",
        cash_delta=amount - 5.0,
        trade_record={"date": pd.Timestamp(date), "stock": stock, "action": "sell"},
        remaining_shares=remaining,
        remaining_cost_cash=0.0,
    )


class TestApplyFills:
    def test_buy_new_position(self):
        engine = _StubEngine()
        ledger = Ledger()
        entry = _buy_entry("2026-01-05", "A", 100, 10.0, seq=0)
        ledger.apply_fills(engine, [entry])
        assert engine.current_capital == pytest.approx(1_000_000.0 - 1005.0)
        assert engine.positions["A"]["shares"] == 100
        assert engine.positions["A"]["buy_cost_cash"] == pytest.approx(1005.0)
        assert len(engine.trades) == 1 and engine.trades[0]["action"] == "buy"

    def test_buy_add_to_position_weighted_cost(self):
        engine = _StubEngine()
        ledger = Ledger()
        ledger.apply_fills(engine, [_buy_entry("2026-01-05", "A", 100, 10.0, seq=0)])
        add = BookingEntry(
            fill_key="2026-01-06|buy|A|1",
            date=pd.Timestamp("2026-01-06"),
            stock="A",
            action="buy",
            cash_delta=-(100 * 11.0 + 5.0),
            trade_record={"action": "buy"},
            add_shares=100,
            add_cost_cash=1105.0,
        )
        ledger.apply_fills(engine, [add])
        assert engine.positions["A"]["shares"] == 200
        assert engine.positions["A"]["buy_cost_cash"] == pytest.approx(1005.0 + 1105.0)
        # 买入日/持有期不重置
        assert engine.positions["A"]["buy_date"] == pd.Timestamp("2026-01-05")

    def test_sell_full_and_partial(self):
        engine = _StubEngine()
        ledger = Ledger()
        ledger.apply_fills(engine, [_buy_entry("2026-01-05", "A", 300, 10.0, seq=0)])
        cost_before = engine.positions["A"]["buy_cost_cash"]
        # 部分卖出 100/300：剩余 200，成本按剩余比例缩减
        partial = _sell_entry("2026-01-06", "A", 100, 11.0, seq=1, remaining=200)
        partial = BookingEntry(
            **{**partial.__dict__, "remaining_cost_cash": cost_before * 200 / 300}
        )
        ledger.apply_fills(engine, [partial])
        assert engine.positions["A"]["shares"] == 200
        assert engine.positions["A"]["buy_cost_cash"] == pytest.approx(cost_before * 200 / 300)
        # 整仓卖出：删除持仓
        full = _sell_entry("2026-01-07", "A", 200, 11.0, seq=2, remaining=0)
        ledger.apply_fills(engine, [full])
        assert "A" not in engine.positions

    def test_duplicate_fill_rejected(self):
        engine = _StubEngine()
        ledger = Ledger()
        entry = _buy_entry("2026-01-05", "A", 100, 10.0, seq=0)
        ledger.apply_fills(engine, [entry])
        with pytest.raises(LedgerError, match="重复入账"):
            ledger.apply_fills(engine, [entry])
        # 拒绝后账户不变
        assert engine.positions["A"]["shares"] == 100
        assert len(engine.trades) == 1

    def test_out_of_order_fill_rejected(self):
        engine = _StubEngine()
        ledger = Ledger()
        ledger.apply_fills(engine, [_buy_entry("2026-01-06", "A", 100, 10.0, seq=0)])
        stale = _buy_entry("2026-01-05", "B", 100, 10.0, seq=1)
        with pytest.raises(LedgerError, match="乱序入账"):
            ledger.apply_fills(engine, [stale])

    def test_shape_validation(self):
        engine = _StubEngine()
        ledger = Ledger()
        # 形态缺失（无持仓变更）
        bad = BookingEntry(
            fill_key="k0",
            date=pd.Timestamp("2026-01-05"),
            stock="A",
            action="buy",
            cash_delta=-1.0,
            trade_record={},
        )
        with pytest.raises(LedgerError, match="形态非法"):
            ledger.apply_fills(engine, [bad])
        # 建仓前置非法（已有持仓）
        ledger.apply_fills(engine, [_buy_entry("2026-01-05", "A", 100, 10.0, seq=0)])
        with pytest.raises(LedgerError, match="建仓入账前置非法"):
            ledger.apply_fills(engine, [_buy_entry("2026-01-06", "A", 100, 10.0, seq=1)])
        # 卖出前置非法（无持仓）
        with pytest.raises(LedgerError, match="卖出入账前置非法"):
            ledger.apply_fills(engine, [_sell_entry("2026-01-07", "B", 100, 10.0, seq=2)])

    def test_ledger_state_export_restore_and_duplicate_guard_after_restore(self):
        engine = _StubEngine()
        ledger = Ledger()
        ledger.apply_fills(engine, [_buy_entry("2026-01-05", "A", 100, 10.0, seq=0)])
        state = ledger.export_state()
        ledger2 = Ledger()
        ledger2.restore_state(copy.deepcopy(state))
        # 恢复后重复提交同一幂等键仍拒绝
        with pytest.raises(LedgerError, match="重复入账"):
            ledger2.apply_fills(engine, [_buy_entry("2026-01-05", "A", 100, 10.0, seq=0)])
        # 缺键/多键 fail-fast
        with pytest.raises(LedgerError):
            ledger2.restore_state({"applied_keys": []})
        with pytest.raises(LedgerError):
            ledger2.restore_state({"applied_keys": [], "last_fill_date": None, "extra": 1})


# ── 运行时守卫：执行器经账本入账 + 现金恒等式 + decide 无副作用 ──


def _make_synthetic_engine_factory():
    """合成引擎工厂（复用 spike runner 的合成基建；全合成不依赖真实数据）。"""
    from scripts.v2_p4.t05b_spike_runner import (
        _DeterministicSignal,
        _make_engine_factory,
        _make_price_data,
    )

    dates = list(pd.bdate_range("2026-01-05", periods=15))
    stocks = ["A", "B", "C", "D"]
    plan = {}
    for i, d in enumerate(dates):
        plan[d.strftime("%Y%m%d")] = (
            [("A", 0.9), ("B", 0.8)] if (i // 5) % 2 == 0 else [("C", 0.95), ("D", 0.85)]
        )
    signal = _DeterministicSignal(2, plan, plan[dates[0].strftime("%Y%m%d")])
    factory = _make_engine_factory(stocks, signal)
    price_data = _make_price_data(
        dates, {s: [10.0 * (1 + 0.001 * i) for i in range(len(dates))] for s in stocks}
    )
    return dates, factory, price_data


@pytest.fixture(autouse=True)
def _synthetic_costs():
    from scripts.v2_p4.t05b_spike_runner import synthetic_cost_settings

    with synthetic_cost_settings():
        yield


class TestRuntimeGuard:
    def test_executors_route_through_ledger_and_account_untouched_before_apply(self):
        """执行器产出 Fill 前后账户不变；只有 Ledger 入账改变现金/股数/成本。"""
        dates, factory, price_data = _make_synthetic_engine_factory()
        engine = factory()
        engine._prepare_price_index(price_data)
        engine.price_data_cache = price_data

        calls: List[tuple] = []
        real_apply = engine.ledger.apply_fills

        def _spy_apply(eng, fills):
            # 入账前账户快照（现金/持仓深拷贝）
            before = (eng.current_capital, copy.deepcopy(eng.positions))
            calls.append((before, list(fills)))
            real_apply(eng, fills)

        engine.ledger.apply_fills = _spy_apply  # type: ignore[assignment]
        # 直接驱动一次买入（执行器内部：计算 → 构造指令 → apply）
        engine._buy_stock_direct(dates[1], "A", 10000.0)
        assert len(calls) == 1, "买入执行必须且仅经 Ledger.apply_fills 一次"
        before, fills = calls[0]
        # 关键断言（P2-3/r3-P2-1 处置）：apply_fills 被调用前账户与初始一致
        assert before == (1_000_000.0, {}), "执行器在入账前不得改动账户（现金/持仓）"
        entry = fills[0]
        assert entry.action == "buy" and entry.new_position is not None
        assert entry.cash_delta == -entry.new_position["buy_cost_cash"]
        assert engine.positions["A"]["shares"] > 0

    def test_sell_path_account_untouched_before_apply(self):
        """卖出路径同型断言：apply 前账户（含已有持仓）不被执行器改动。"""
        dates, factory, price_data = _make_synthetic_engine_factory()
        engine = factory()
        engine._prepare_price_index(price_data)
        engine.price_data_cache = price_data
        engine._buy_stock_direct(dates[1], "A", 10000.0)
        capital_mid = engine.current_capital
        positions_mid = copy.deepcopy(engine.positions)

        calls: List[tuple] = []
        real_apply = engine.ledger.apply_fills

        def _spy_apply(eng, fills):
            calls.append(((eng.current_capital, copy.deepcopy(eng.positions)), list(fills)))
            real_apply(eng, fills)

        engine.ledger.apply_fills = _spy_apply  # type: ignore[assignment]
        engine._sell_stock_direct(dates[2], "A")
        assert len(calls) == 1
        before, fills = calls[0]
        assert before == (capital_mid, positions_mid), "卖出执行器在入账前不得改动账户"
        assert fills[0].action == "sell"
        assert "A" not in engine.positions

    def test_ledger_owns_booked_objects(self):
        """R2-P4-T1-04：入账后修改调用方保留的入账对象不得反向改账户/成交日志。"""
        engine = _StubEngine()
        ledger = Ledger()
        entry = _buy_entry("2026-01-05", "A", 100, 10.0, seq=0)
        ledger.apply_fills(engine, [entry])
        # 调用方保留的可变引用被改写（模拟外部持有者的意外/恶意修改）
        entry.new_position["shares"] = 777
        entry.trade_record["action"] = "corrupted"
        # 账户与成交日志不受影响
        assert engine.positions["A"]["shares"] == 100
        assert engine.trades[0]["action"] == "buy"

    def test_batch_prevalidate_zero_side_effect_on_key_error(self):
        """r3-P2-3：批量内重复键/重复提交 ⇒ 预检失败且零副作用（一笔都不入账）。"""
        engine = _StubEngine()
        ledger = Ledger()
        e0 = _buy_entry("2026-01-05", "A", 100, 10.0, seq=0)
        ledger.apply_fills(engine, [e0])
        # 批量内第二笔与已入账键重复 ⇒ 预检拒绝，批量内第一笔也不入账
        e1 = _buy_entry("2026-01-06", "B", 100, 10.0, seq=1)
        e_dup = _buy_entry("2026-01-05", "A", 100, 10.0, seq=0)  # 与 e0 同幂等键
        with pytest.raises(LedgerError, match="重复入账"):
            ledger.apply_fills(engine, [e1, e_dup])
        assert "B" not in engine.positions, "预检失败时不得有任何入账副作用"
        assert len(engine.trades) == 1
        capital = engine.current_capital
        # 批量内自重复键同样拒绝
        e2 = _buy_entry("2026-01-06", "B", 100, 10.0, seq=1)
        with pytest.raises(LedgerError, match="重复幂等键"):
            ledger.apply_fills(engine, [e2, e2])
        assert engine.current_capital == capital

    def test_batch_out_of_order_rejected_empty_ledger(self):
        """复审处置（r1-P2-A / R2-P4-T1-R1-01）：空账本批内日期倒序 ⇒ 预检拒绝
        且零副作用（账户/日志/幂等键/水位全不变）。"""
        engine = _StubEngine()
        ledger = Ledger()
        e_new = _buy_entry("2026-01-06", "A", 100, 10.0, seq=0)
        e_old = _buy_entry("2026-01-05", "B", 100, 10.0, seq=1)
        with pytest.raises(LedgerError, match="乱序入账"):
            ledger.apply_fills(engine, [e_new, e_old])
        assert engine.positions == {} and engine.trades == []
        assert engine.current_capital == 1_000_000.0
        assert ledger.export_state() == {"applied_keys": [], "last_fill_date": None}

    def test_batch_out_of_order_rejected_with_history(self):
        """批内倒序（已有历史日期）⇒ 拒绝且水位不回退。"""
        engine = _StubEngine()
        ledger = Ledger()
        ledger.apply_fills(engine, [_buy_entry("2026-01-05", "A", 100, 10.0, seq=0)])
        before = ledger.export_state()
        e7 = _buy_entry("2026-01-07", "B", 100, 10.0, seq=1)
        e6 = _buy_entry("2026-01-06", "C", 100, 10.0, seq=2)  # 早于前笔
        with pytest.raises(LedgerError, match="乱序入账"):
            ledger.apply_fills(engine, [e7, e6])
        assert ledger.export_state() == before
        assert set(engine.positions) == {"A"}

    def test_batch_same_day_and_increasing_accepted(self):
        """正例：同日多笔 + 递增日期批量均合法（预检 + 应用单调性不误伤）。"""
        engine = _StubEngine()
        ledger = Ledger()
        e1 = _buy_entry("2026-01-05", "A", 100, 10.0, seq=0)
        e2 = _buy_entry("2026-01-05", "B", 100, 10.0, seq=1)  # 同日
        e3 = _buy_entry("2026-01-06", "C", 100, 10.0, seq=2)  # 递增
        ledger.apply_fills(engine, [e1, e2, e3])
        assert set(engine.positions) == {"A", "B", "C"}

    def test_ledger_validate_state_matrix(self):
        """账本投影纯校验（validate_state）：键集/类型/元素/重复/水位类型全覆盖。"""
        good = {"applied_keys": ["k0"], "last_fill_date": pd.Timestamp("2026-01-05")}
        Ledger.validate_state(good)  # 正例通过
        for bad in (
            {"applied_keys": []},  # 缺键
            {"applied_keys": [], "last_fill_date": None, "x": 1},  # 多键
            {"applied_keys": 0, "last_fill_date": None},  # 非 list
            {"applied_keys": ["a", 1], "last_fill_date": None},  # 非 str 元素
            {"applied_keys": ["a", "a"], "last_fill_date": None},  # 重复键
            {"applied_keys": [], "last_fill_date": "20260105"},  # 水位类型非法
        ):
            with pytest.raises(LedgerError):
                Ledger.validate_state(bad)

    def test_daily_cash_identity_via_ledger(self):
        """逐日现金恒等式：每日现金变化 == 当日入账 cash_delta 合计（逐日步进驱动）。"""
        from scripts.v2_p4.t05b_step_driver import run_stepped

        dates, factory, price_data = _make_synthetic_engine_factory()
        ledger_calls: List[tuple] = []

        def _factory_with_spy():
            engine = factory()
            real_apply = engine.ledger.apply_fills

            def _spy(eng, fills):
                ledger_calls.append((eng, list(fills)))
                return real_apply(eng, fills)

            engine.ledger.apply_fills = _spy  # type: ignore[assignment]
            return engine

        result = run_stepped(
            _factory_with_spy, dates[0], dates[-1], dates, price_data
        )
        assert result.snapshot_roundtrip_ok
        # 汇总：全部入账 cash_delta 合计 == 末日现金 − 初始现金
        total_delta = sum(e.cash_delta for _eng, fills in ledger_calls for e in fills)
        final_capital = result.engine_state["fields"]["current_capital"]
        assert final_capital == pytest.approx(1_000_000.0 + total_delta)
        # 且全部入账经 spy 通道（笔数 == 成交日志笔数）
        n_fills = sum(len(fills) for _eng, fills in ledger_calls)
        assert n_fills == len(result.engine_state["fields"]["trades"])

    def test_decide_phase_no_side_effect_and_idempotent(self):
        """决策相位无副作用：_generate_signal 不改变账户（现金/持仓及嵌套对象），
        同输入重复调用输出一致（pending_signals 计划逐值相等）。"""
        dates, factory, price_data = _make_synthetic_engine_factory()
        engine = factory()
        date = dates[0]
        engine._trade_date_index = {d: i for i, d in enumerate(dates)}
        engine._prepare_price_index(price_data)
        engine.price_data_cache = price_data
        engine._signal_dates = engine._get_rebalance_dates(dates)
        account_before = (engine.current_capital, copy.deepcopy(engine.positions))

        engine._generate_signal(
            date, dates, price_data, engine._trade_date_index, tranche_idx=0
        )
        assert date in engine.pending_signals  # 信号日入队
        first_plan = copy.deepcopy(engine.pending_signals[date])
        account_after = (engine.current_capital, copy.deepcopy(engine.positions))
        assert account_after == account_before, "决策相位不得改变账户（现金/持仓）"

        # 同输入重复 decide：输出一致
        engine._generate_signal(
            date, dates, price_data, engine._trade_date_index, tranche_idx=0
        )
        assert engine.pending_signals[date] == first_plan

        # _check_and_sell 同样不得改变账户
        engine._check_and_sell(date, dates, engine._trade_date_index)
        assert (engine.current_capital, engine.positions) == account_before

    def test_decide_phase_with_existing_positions_registered_exception(self):
        """决策相位 + 非空持仓：账户逐值不变，**唯一例外 = last_known_price
        估值缓存写回**（D10 回退链语义，P2-2/R2-P4-T1-03 登记例外——
        现金/股数/成本/买入日/成交/队列一律不变）。

        本测试同时把「例外面边界」机器锁定：任何新增的嵌套写点（除
        last_known_price 外）都会被本断言检出。
        """
        dates, factory, price_data = _make_synthetic_engine_factory()
        engine = factory()
        engine._trade_date_index = {d: i for i, d in enumerate(dates)}
        engine._prepare_price_index(price_data)
        engine.price_data_cache = price_data
        engine._signal_dates = engine._get_rebalance_dates(dates)
        # 先建仓（非空持仓形态）
        engine._buy_stock_direct(dates[1], "A", 10000.0)
        assert "A" in engine.positions

        def account_without_valuation_cache(eng):
            # 先深拷贝再剔除登记例外键（R2-P4-T1-03 复审处置：一层浅拷贝会共享
            # 嵌套扩展字段，导致变异漏检）
            positions = copy.deepcopy(eng.positions)
            for info in positions.values():
                info.pop("last_known_price", None)
            return eng.current_capital, positions

        before = account_without_valuation_cache(engine)
        decide_date = dates[2]
        engine._generate_signal(
            decide_date, dates, price_data, engine._trade_date_index, tranche_idx=0
        )
        engine._check_and_sell(decide_date, dates, engine._trade_date_index)
        after = account_without_valuation_cache(engine)
        assert after == before, "现金/持仓（除 last_known_price 例外）/成本/买入日不得改变"
        assert len(engine.trades) == 1, "决策相位不得产生成交"

    def test_decide_test_detects_nested_extension_mutation(self):
        """反例（R2-P4-T1-03 复审处置）：持仓嵌套扩展字段被改写 ⇒ 比对必须检出
        （证明 decide 测试的嵌套守卫不是恒真）。"""
        dates, factory, price_data = _make_synthetic_engine_factory()
        engine = factory()
        engine._trade_date_index = {d: i for i, d in enumerate(dates)}
        engine._prepare_price_index(price_data)
        engine.price_data_cache = price_data
        engine._buy_stock_direct(dates[1], "A", 10000.0)

        def account_without_valuation_cache(eng):
            positions = copy.deepcopy(eng.positions)
            for info in positions.values():
                info.pop("last_known_price", None)
            return eng.current_capital, positions

        # 快照前建立嵌套扩展对象、快照后修改其中已有值（r2 二次复审非阻塞项：
        # 该形态更精确——浅拷贝下 before 会随引擎同步变化，断言恒不触发）
        engine.positions["A"]["extra_meta"] = {"note": "original"}
        before = account_without_valuation_cache(engine)
        engine.positions["A"]["extra_meta"]["note"] = "tampered"
        after = account_without_valuation_cache(engine)
        assert after != before, "嵌套扩展字段变异必须被检出"

    def test_iter_account_writes_covers_method_calls(self):
        """静态扫描面覆盖方法调用（update/pop/clear 等）与项级写（r1-P2-3 处置）。"""
        import textwrap

        src = textwrap.dedent(
            """
            class _X:
                def f(self):
                    self.positions["A"] = {}
                    self.positions.update({})
                    self.positions.pop("A", None)
                    self.current_capital += 1.0
                    engine.current_capital = 0.0
                    del self.positions["A"]
            """
        )
        import ast as _ast
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(src)
            path = Path(fh.name)
        hits = _iter_account_writes(path)
        assert len(hits) == 6, f"方法调用/项级写/删除未全覆盖: {hits}"

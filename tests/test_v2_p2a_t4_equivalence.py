"""P2a-T4 新旧逐值一致等价测试（engine.py 函数级拆分 + 整模块复制件）。

T0 实施规划 v4 §4-T4 验收口径（仿 T1 ``test_v2_p2a_t1_equivalence.py`` 风格）：
- 方法转函数：同一合成输入下，旧引擎方法（绑定到属性桩）与新纯函数输出逐值比对；
- ``__init__`` 状态初始化等价（D8 提取式拆分）：属性集与逐值比对，
  exposure 关联字段缺席断言 = D6 摘除生效；
- 估值回退 3 处逐点等价（D4：旧 engine.py:743 / :777 / holdings_snapshot.py:62-66）；
- 整模块复制件（pending_order / signal.base / holdings_snapshot）行为序列比对。

不依赖真实配置与真实数据：所有输入为合成 fixture；行业约束分支的
``DataLoader`` / ``get_shenwan_level`` 经 monkeypatch 打为合成 stub。
"""

import copy
import dataclasses
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

# ── 旧模块（生产在用，迁移期不修改） ──
from src.lazybull.backtest import holdings_snapshot as old_snapshot
from src.lazybull.backtest.engine import BacktestEngine as OldEngine
from src.lazybull.common import cost as old_cost
from src.lazybull.common.cost import CostModel as OldCostModel
from src.lazybull.execution import pending_order as old_pending
from src.lazybull.risk.stop_loss import StopLossConfig as OldStopLossConfig
from src.lazybull.signals import base as old_signal_base
from src.lazybull.universe.base import BasicUniverse
from src.lazybull.v2.common import cost as new_cost
from src.lazybull.v2.common.cost import CostModel as NewCostModel
from src.lazybull.v2.common.rules.stop_loss import StopLossConfig as NewStopLossConfig
from src.lazybull.v2.core.accounting import exports as new_exports
from src.lazybull.v2.core.accounting import holdings_snapshot as new_snapshot
from src.lazybull.v2.core.accounting import nav as new_nav
from src.lazybull.v2.core.accounting import state as new_state
from src.lazybull.v2.core.decision import params as new_params
from src.lazybull.v2.core.decision import weighting as new_weighting
from src.lazybull.v2.core.execution import engine_utils as new_engine_utils

# ── v2 部件（P2a-T4 交付） ──
from src.lazybull.v2.core.execution import pending_order as new_pending
from src.lazybull.v2.core.execution import price_index as new_price_index
from src.lazybull.v2.core.signal import base as new_signal_base

_MISSING = dataclasses.MISSING

# D6 摘除清单：旧 __init__ 中 exposure mixin 关联的全部状态字段
_EXPOSURE_ATTRS = [
    "exposure_table",
    "exposure_stats",
    "_exposure_missing_dates",
    "pending_exposure_trims",
    "exposure_trim_tolerance",
    "exposure_trim_stats",
    "exposure_release_budget",
    "pending_exposure_replenishes",
    "exposure_replenish_enabled",
    "exposure_replenish_stats",
]


@pytest.fixture()
def synthetic_cost_settings(monkeypatch):
    """合成成本配置（R2-T4-01 处置，范式同 T1 ``synthetic_cost_settings``）。

    新旧 ``CostModel`` 构造器即使四参全显式传入，也会**无条件**调用
    ``get_cost_settings()`` 读取共享配置单例——显式传参不等于隔离配置访问。
    本 fixture 把新旧两侧成本模块各自绑定的 ``get_cost_settings`` 打为同一
    合成读取函数，pytest 结束自动恢复；返回读取计数供反例断言。"""
    settings = {
        "commission_rate": 0.0003,
        "min_commission": 5.0,
        "stamp_tax": 0.0005,
        "slippage": 0.001,
    }
    reads: List[int] = []

    def _stub() -> dict:
        reads.append(1)
        return dict(settings)

    monkeypatch.setattr(old_cost, "get_cost_settings", _stub)
    monkeypatch.setattr(new_cost, "get_cost_settings", _stub)
    return {"settings": settings, "reads": reads}


def _assert_dataclass_fields_equal(old_cls, new_cls) -> None:
    """dataclass 字段集 + 默认值逐项比对（同 T1 口径）。"""
    old_fields = old_cls.__dataclass_fields__
    new_fields = new_cls.__dataclass_fields__
    assert list(old_fields) == list(new_fields), "字段集或顺序不一致"
    for name in old_fields:
        of, nf = old_fields[name], new_fields[name]
        assert of.type == nf.type, f"字段 {name} 类型不一致"
        if of.default is _MISSING:
            assert nf.default is _MISSING, f"字段 {name} 默认值缺失状态不一致"
        else:
            assert of.default == nf.default, f"字段 {name} 默认值不一致"
        if of.default_factory is _MISSING:  # type: ignore[misc]
            assert nf.default_factory is _MISSING, (  # type: ignore[misc]
                f"字段 {name} default_factory 缺失状态不一致"
            )
        else:
            same_factory = of.default_factory() == nf.default_factory()  # type: ignore[misc]
            assert same_factory, f"字段 {name} default_factory 产物不一致"


# ══════════════════ core/decision/params ══════════════════


class _SignalWithTopN:
    def __init__(self, top_n):
        self.top_n = top_n


def _params_stub(signal, positions, stagger_tranches=1, min_buy_value_ratio=0.0):
    stub = SimpleNamespace(
        signal=signal,
        positions=positions,
        stagger_tranches=stagger_tranches,
        min_buy_value_ratio=min_buy_value_ratio,
    )
    # 旧方法的内部委托链：绑定旧实现保持逐字语义
    stub._get_target_position_count = lambda: OldEngine._get_target_position_count(stub)
    return stub


def test_get_target_position_count_equivalence():
    cases = [
        (_SignalWithTopN(5), {}),
        (_SignalWithTopN(5), {"A": {}, "B": {}}),
        (_SignalWithTopN(0), {"A": {}}),  # top_n 非正 → 回退持仓数
        (_SignalWithTopN("5"), {"A": {}}),  # 非 int → 回退持仓数
        (SimpleNamespace(), {"A": {}, "B": {}, "C": {}}),  # 无 top_n 属性
    ]
    for signal, positions in cases:
        stub = _params_stub(signal, positions)
        assert OldEngine._get_target_position_count(stub) == new_params.get_target_position_count(
            signal, positions
        )


def test_tranche_target_count_and_capital_fraction_equivalence():
    signal = _SignalWithTopN(10)
    positions = {"A": {}, "B": {}}
    for k in (1, 2, 3, 5):
        for idx in range(k):
            stub = _params_stub(signal, positions, stagger_tranches=k)
            assert OldEngine._get_tranche_target_count(
                stub, idx
            ) == new_params.get_tranche_target_count(idx, k, signal, positions)
            # 显式 target_count 分支
            assert OldEngine._get_tranche_target_count(
                stub, idx, target_count=7
            ) == new_params.get_tranche_target_count(idx, k, signal, positions, target_count=7)
            assert OldEngine._get_tranche_capital_fraction(
                stub, idx
            ) == new_params.get_tranche_capital_fraction(idx, k, signal, positions)


def test_get_min_buy_value_threshold_equivalence():
    date = pd.Timestamp("2026-01-05")
    for ratio in (0.0, 0.3, 0.5):
        for total_assets, target_count in ((1_000_000.0, 5), (0.0, 5), (1_000_000.0, 0)):
            stub = _params_stub(_SignalWithTopN(5), {}, min_buy_value_ratio=ratio)
            stub._calculate_portfolio_value = lambda d, v=total_assets: v
            stub._get_target_position_count = lambda n=target_count: n
            old_val = OldEngine._get_min_buy_value_threshold(stub, date)
            new_val = new_params.get_min_buy_value_threshold(ratio, total_assets, target_count)
            assert old_val == new_val
            if ratio <= 0 or target_count <= 0 or total_assets <= 0:
                assert new_val == 0.0
            else:
                assert new_val == pytest.approx(total_assets / target_count * ratio)


# ══════════════════ core/execution/price_index ══════════════════


def _price_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["A", "B", "A", "B"],
            "trade_date": ["20260105", "20260105", "20260106", "20260106"],
            "close": [10.0, 20.0, 10.5, 21.0],
            "open": [9.9, 19.8, 10.2, 20.5],
            "close_adj": [11.0, 22.0, 11.5, 23.0],
            "open_adj": [10.9, 21.8, 11.2, 22.5],
        }
    )


def _assert_price_indexes_equal(old_stub, new_idx) -> None:
    pd.testing.assert_series_equal(old_stub.trade_price_index, new_idx.trade_price_index)
    pd.testing.assert_series_equal(old_stub.pnl_price_index, new_idx.pnl_price_index)
    pd.testing.assert_series_equal(old_stub.trade_price_open_index, new_idx.trade_price_open_index)
    pd.testing.assert_series_equal(old_stub.pnl_price_open_index, new_idx.pnl_price_open_index)


def test_prepare_price_index_full_equivalence():
    old_stub = SimpleNamespace()
    OldEngine._prepare_price_index(old_stub, _price_df())
    new_idx = new_price_index.prepare_price_index(_price_df())
    _assert_price_indexes_equal(old_stub, new_idx)


def test_prepare_price_index_fallback_branches_equivalence():
    variants = {
        "缺 close_adj": _price_df().drop(columns=["close_adj"]),
        "缺 open": _price_df().drop(columns=["open"]),
        "open 全 NaN": _price_df().assign(open=np.nan),
        "缺 open_adj": _price_df().drop(columns=["open_adj"]),
        "缺 open 与 open_adj": _price_df().drop(columns=["open", "open_adj"]),
        "open_adj 全 NaN": _price_df().assign(open_adj=np.nan),
        "open_adj 全 NaN 且缺 open": _price_df().drop(columns=["open"]).assign(open_adj=np.nan),
        "字符串日期": _price_df(),
        "datetime 日期": _price_df().assign(
            trade_date=pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"])
        ),
    }
    for name, df in variants.items():
        old_stub = SimpleNamespace()
        OldEngine._prepare_price_index(old_stub, df)
        new_idx = new_price_index.prepare_price_index(df)
        _assert_price_indexes_equal(old_stub, new_idx)


def test_prepare_price_index_missing_close_error():
    df = _price_df().drop(columns=["close"])
    with pytest.raises(ValueError) as old_exc:
        OldEngine._prepare_price_index(SimpleNamespace(), df)
    with pytest.raises(ValueError) as new_exc:
        new_price_index.prepare_price_index(df)
    assert str(old_exc.value) == str(new_exc.value)


def test_get_price_functions_equivalence():
    old_stub = SimpleNamespace()
    OldEngine._prepare_price_index(old_stub, _price_df())
    new_idx = new_price_index.prepare_price_index(_price_df())
    d1, d_missing = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-03-01")
    for date, stock in ((d1, "A"), (d1, "B"), (d1, "MISSING"), (d_missing, "A")):
        assert OldEngine._get_trade_price(old_stub, date, stock) == new_price_index.get_trade_price(
            new_idx, date, stock
        )
        assert OldEngine._get_pnl_price(old_stub, date, stock) == new_price_index.get_pnl_price(
            new_idx, date, stock
        )
        assert OldEngine._get_trade_price_open(
            old_stub, date, stock
        ) == new_price_index.get_trade_price_open(new_idx, date, stock)
        assert OldEngine._get_pnl_price_open(
            old_stub, date, stock
        ) == new_price_index.get_pnl_price_open(new_idx, date, stock)
    assert new_price_index.get_trade_price(new_idx, d1, "MISSING") is None


# ══════════════════ core/decision/weighting（D5 回测侧语义） ══════════════════


def _pnl_price_series(n_days: int = 30) -> pd.Series:
    dates = pd.bdate_range("2026-01-05", periods=n_days)
    records = {}
    for i, d in enumerate(dates):
        records[(d, "A")] = 10.0 * (1.01**i)
        records[(d, "B")] = 20.0 * (1.0 + 0.005 * ((-1) ** i))
    return pd.Series(records).rename_axis(["trade_date", "ts_code"])


def _weighting_stub(**overrides) -> SimpleNamespace:
    base = dict(
        pnl_price_index=_pnl_price_series(),
        vol_window=20,
        vol_epsilon=0.001,
        position_sizing="equal",
        _normalize_log_count=0,
        verbose=False,
        kelly_max_leverage=0.25,
        price_data_cache=None,
        kelly_vol_window=60,
        TRADING_DAYS_PER_YEAR=252,  # 旧实现读类常量；桩以实例属性供给同值
    )
    base.update(overrides)
    stub = SimpleNamespace(**base)
    # 旧方法的内部委托链：绑定旧实现保持逐字语义
    stub._calculate_volatility = lambda stock, end_date: OldEngine._calculate_volatility(
        stub, stock, end_date
    )
    stub._estimate_stock_variance = lambda stock, date: OldEngine._estimate_stock_variance(
        stub, stock, date
    )
    stub._kelly_weights = lambda signals, date, half=False: OldEngine._kelly_weights(
        stub, signals, date, half=half
    )
    return stub


def _price_cache(n_days: int = 25) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=n_days).strftime("%Y%m%d")
    rows = []
    for i, d in enumerate(dates):
        rows.append({"ts_code": "A", "trade_date": d, "close_adj": 10.0 * (1.01**i)})
        rows.append({"ts_code": "B", "trade_date": d, "close_adj": 20.0 * (1.002**i)})
    return pd.DataFrame(rows)


def test_calculate_volatility_equivalence():
    stub = _weighting_stub()
    idx = stub.pnl_price_index
    end = pd.Timestamp("2026-02-10")
    for stock, end_date in (
        ("A", end),
        ("B", end),
        ("A", pd.Timestamp("2026-01-06")),
        ("ZZZ", end),
    ):
        assert OldEngine._calculate_volatility(
            stub, stock, end_date
        ) == new_weighting.calculate_volatility(idx, stock, end_date, 20, 0.001)
    # 数据不足与未知股票回退到 epsilon
    assert (
        new_weighting.calculate_volatility(idx, "A", pd.Timestamp("2026-01-06"), 20, 0.001) == 0.001
    )
    assert new_weighting.calculate_volatility(idx, "ZZZ", end, 20, 0.001) == 0.001


def test_apply_risk_budget_equivalence():
    stub = _weighting_stub()
    signals = {"A": 0.6, "B": 0.4}
    date = pd.Timestamp("2026-02-10")
    assert OldEngine._apply_risk_budget(stub, signals, date) == new_weighting.apply_risk_budget(
        signals, date, stub.pnl_price_index, 20, 0.001
    )
    # 空信号分支
    assert (
        OldEngine._apply_risk_budget(stub, {}, date)
        == new_weighting.apply_risk_budget({}, date, stub.pnl_price_index, 20, 0.001)
        == {}
    )


def test_estimate_stock_variance_equivalence():
    cache = _price_cache()
    date = pd.Timestamp("2026-02-05")
    # 缓存为 None → None
    stub_none = _weighting_stub(price_data_cache=None)
    assert (
        OldEngine._estimate_stock_variance(stub_none, "A", date)
        is new_weighting.estimate_stock_variance(None, "A", date, 60)
        is None
    )
    # 数据充足 → 方差一致
    stub = _weighting_stub(price_data_cache=cache)
    assert OldEngine._estimate_stock_variance(
        stub, "A", date
    ) == new_weighting.estimate_stock_variance(cache, "A", date, 60)
    # 数据不足 20 行 → None
    small = cache.head(10)
    stub_small = _weighting_stub(price_data_cache=small)
    assert (
        OldEngine._estimate_stock_variance(stub_small, "A", date)
        is new_weighting.estimate_stock_variance(small, "A", date, 60)
        is None
    )
    # 无 close_adj 列 → 回退 close 列
    cache_close = cache.rename(columns={"close_adj": "close"})
    stub_close = _weighting_stub(price_data_cache=cache_close)
    assert OldEngine._estimate_stock_variance(
        stub_close, "A", date
    ) == new_weighting.estimate_stock_variance(cache_close, "A", date, 60)


def test_kelly_weights_equivalence():
    cache = _price_cache()
    signals = {"A": 0.8, "B": 0.5}
    date = pd.Timestamp("2026-02-05")
    for half in (False, True):
        stub = _weighting_stub(price_data_cache=cache)
        old_res = OldEngine._kelly_weights(stub, signals, date, half=half)
        new_res = new_weighting.kelly_weights(
            signals,
            date,
            kelly_max_leverage=0.25,
            price_data_cache=cache,
            kelly_vol_window=60,
            verbose=False,
            half=half,
        )
        assert old_res == new_res


def test_normalize_signals_equivalence():
    cache = _price_cache()
    date = pd.Timestamp("2026-02-05")
    signals = {"A": 0.8, "B": 0.5, "C": 0.3}
    for sizing in ("equal", "score", "kelly", "half_kelly", "unknown_mode"):
        stub = _weighting_stub(position_sizing=sizing, price_data_cache=cache)
        old_res = OldEngine._normalize_signals(stub, dict(signals), date)
        new_res, new_count = new_weighting.normalize_signals(
            dict(signals),
            date,
            position_sizing=sizing,
            normalize_log_count=0,
            verbose=False,
            kelly_max_leverage=0.25,
            price_data_cache=cache,
            kelly_vol_window=60,
        )
        assert old_res == new_res, f"sizing={sizing} 权重不一致"
        assert stub._normalize_log_count == new_count, f"sizing={sizing} 诊断计数不一致"
    # 空信号分支
    stub = _weighting_stub()
    assert (
        OldEngine._normalize_signals(stub, {}, date)
        == new_weighting.normalize_signals(
            {},
            date,
            position_sizing="equal",
            normalize_log_count=0,
            verbose=False,
            kelly_max_leverage=0.25,
            price_data_cache=None,
            kelly_vol_window=60,
        )[0]
        == {}
    )
    # score 全负 → 静默等权回退（D5 回测侧语义）
    stub = _weighting_stub(position_sizing="score", verbose=True)
    neg = {"A": -0.1, "B": -0.2}
    old_res = OldEngine._normalize_signals(stub, dict(neg), date)
    new_res, new_count = new_weighting.normalize_signals(
        dict(neg),
        date,
        position_sizing="score",
        normalize_log_count=0,
        verbose=True,
        kelly_max_leverage=0.25,
        price_data_cache=cache,
        kelly_vol_window=60,
    )
    assert old_res == new_res == {"A": 0.5, "B": 0.5}
    assert stub._normalize_log_count == new_count


def test_normalize_signals_log_count_increment():
    """诊断日志计数语义：equal/score 分支 <5 时递增、到 5 停止；kelly 分支不动。"""
    for sizing in ("equal", "score"):
        stub = _weighting_stub(position_sizing=sizing)
        count = 0
        for _ in range(7):
            old_res = OldEngine._normalize_signals(
                stub, {"A": 0.6, "B": 0.4}, pd.Timestamp("2026-02-05")
            )
            new_res, count = new_weighting.normalize_signals(
                {"A": 0.6, "B": 0.4},
                pd.Timestamp("2026-02-05"),
                position_sizing=sizing,
                normalize_log_count=count,
                verbose=False,
                kelly_max_leverage=0.25,
                price_data_cache=None,
                kelly_vol_window=60,
            )
            assert old_res == new_res
            assert stub._normalize_log_count == count
        assert count == 5  # 只计前 5 次


# ══════════════════ core/accounting/nav（估值回退 D4 第 1/2 处） ══════════════════


def _nav_price_index() -> new_price_index.PriceIndexes:
    idx = new_price_index.prepare_price_index(_price_df())
    return idx


def _nav_positions() -> Dict[str, Dict]:
    return {
        # A：当日有价 → 用之并写回 last_known_price
        "A": {"shares": 100, "buy_trade_price": 9.0},
        # B：当日无价（合成行情仅覆盖 2026-01-05/06）→ 回退缓存价
        "B": {"shares": 200, "buy_trade_price": 19.0, "last_known_price": 18.5},
        # C：无价且无缓存 → 回退买入价（告警分支）
        "C": {"shares": 50, "buy_trade_price": 5.0},
        # D：无价无缓存且买入价为 0 → 0 估值（不告警分支）
        "D": {"shares": 10, "buy_trade_price": 0.0},
    }


def _nav_old_stub(positions, current_capital, trade_price_index) -> SimpleNamespace:
    """旧侧估值桩：绑定 _get_trade_price 内部委托（旧实现逐字语义）。"""
    stub = SimpleNamespace(
        positions=positions,
        current_capital=current_capital,
        trade_price_index=trade_price_index,
    )
    stub._get_trade_price = lambda date, stock: OldEngine._get_trade_price(stub, date, stock)
    return stub


def test_calculate_portfolio_value_fallback_equivalence():
    """估值回退第 1 处（旧 engine.py:743）：当日价 → 缓存价 → 买入价。"""
    date = pd.Timestamp("2026-02-10")  # 行情索引中不存在 ⇒ 全部走回退
    old_positions = _nav_positions()
    old_stub = _nav_old_stub(old_positions, 1000.0, _nav_price_index().trade_price_index)
    old_val = OldEngine._calculate_portfolio_value(old_stub, date)

    new_positions = _nav_positions()
    new_val = new_nav.calculate_portfolio_value(new_positions, 1000.0, _nav_price_index(), date)
    assert old_val == new_val
    assert old_positions == new_positions  # last_known_price 写回副作用一致

    # 当日有价分支：A 命中行情，写回 last_known_price
    date2 = pd.Timestamp("2026-01-05")
    old_positions2 = _nav_positions()
    old_stub2 = _nav_old_stub(old_positions2, 1000.0, _nav_price_index().trade_price_index)
    old_val2 = OldEngine._calculate_portfolio_value(old_stub2, date2)
    new_positions2 = _nav_positions()
    new_val2 = new_nav.calculate_portfolio_value(new_positions2, 1000.0, _nav_price_index(), date2)
    assert old_val2 == new_val2
    assert old_positions2 == new_positions2
    assert new_positions2["A"]["last_known_price"] == 10.0
    # 手工核对：A/B 当日有价（10.0/20.0）并写回缓存，C 回退买入价，D 买入价 0 → 0
    assert new_val2 == 1000.0 + 100 * 10.0 + 200 * 20.0 + 50 * 5.0


def test_position_market_value_fallback_equivalence():
    """估值回退第 2 处（旧 engine.py:777）：口径与组合估值完全一致。"""
    date = pd.Timestamp("2026-02-10")
    idx = _nav_price_index()

    def old_call(positions, stock):
        stub = _nav_old_stub(positions, 0.0, idx.trade_price_index)
        return OldEngine._position_market_value(stub, date, stock)

    positions = _nav_positions()
    for stock in ("A", "B", "C", "D", "NOT_HELD"):
        assert old_call(positions, stock) == new_nav.position_market_value(
            positions, idx, date, stock
        )
    # 锚定语义：B 回退缓存价、C 回退买入价、D 买入价 0 → None、非持仓 → None
    assert new_nav.position_market_value(positions, idx, date, "B") == 200 * 18.5
    assert new_nav.position_market_value(positions, idx, date, "C") == 50 * 5.0
    assert new_nav.position_market_value(positions, idx, date, "D") is None
    assert new_nav.position_market_value(positions, idx, date, "NOT_HELD") is None
    # 股数为 0 的持仓 → None
    zero = {"Z": {"shares": 0, "buy_trade_price": 3.0}}
    assert old_call(zero, "Z") == new_nav.position_market_value(zero, idx, date, "Z") is None


def test_generate_nav_curve_equivalence():
    portfolio_values = [
        {"date": pd.Timestamp("2026-01-05"), "portfolio_value": 1_000_000.0},
        {"date": pd.Timestamp("2026-01-06"), "portfolio_value": 1_010_000.0},
    ]
    old_stub = SimpleNamespace(portfolio_values=portfolio_values, initial_capital=1_000_000.0)
    old_df = OldEngine._generate_nav_curve(old_stub)
    new_df = new_nav.generate_nav_curve(portfolio_values, 1_000_000.0)
    pd.testing.assert_frame_equal(old_df, new_df)
    assert list(new_df["nav"]) == [1.0, pytest.approx(1.01)]
    assert list(new_df["return"]) == [0.0, pytest.approx(0.01)]


# ══════════════════ core/accounting/exports ══════════════════


def test_exports_equivalence():
    trades = [
        {"date": "2026-01-06", "stock": "A", "action": "buy", "price": 10.0, "shares": 100},
        {"date": "2026-01-11", "stock": "A", "action": "sell", "price": 10.5, "shares": 100},
    ]
    attribution = [{"signal_date": "2026-01-05", "stock": "A", "slot": 0}]
    old_stub = SimpleNamespace(trades=trades, execution_attribution_records=attribution)
    pd.testing.assert_frame_equal(OldEngine.get_trades(old_stub), new_exports.get_trades(trades))
    pd.testing.assert_frame_equal(
        OldEngine.get_execution_attribution(old_stub),
        new_exports.get_execution_attribution(attribution),
    )
    # 空记录分支
    empty_stub = SimpleNamespace(trades=[], execution_attribution_records=[])
    pd.testing.assert_frame_equal(OldEngine.get_trades(empty_stub), new_exports.get_trades([]))
    pd.testing.assert_frame_equal(
        OldEngine.get_execution_attribution(empty_stub),
        new_exports.get_execution_attribution([]),
    )


# ══════════════════ core/execution/engine_utils（组装体 T5 前置部件） ══════════════════


class _FakeStorage:
    """Storage 桩：load_raw_by_date 返回合成 suspend 数据（免真实数据依赖）。"""

    def __init__(self, data):
        self._data = data

    def load_raw_by_date(self, dataset: str, trade_date: str):
        assert dataset == "suspend"
        return self._data.get(trade_date)


def _suspend_fixture() -> dict:
    return {
        "20260105": pd.DataFrame(
            {
                "ts_code": ["A", "B", "C"],
                "suspend_type": ["S", "R", "X"],
            }
        ),
    }


def test_ensure_suspend_calendar_equivalence():
    fake = _FakeStorage(_suspend_fixture())
    old_stub = SimpleNamespace(_suspend_calendar=None, data_storage=fake)
    old_cal = OldEngine._get_suspend_calendar(old_stub)
    new_cal, new_storage = new_engine_utils.ensure_suspend_calendar(None, fake)
    # 存储透写一致
    assert old_stub.data_storage is fake and new_storage is fake
    # 日历行为一致
    for ts_code in ("A", "B", "C", "NO_RECORD"):
        assert old_cal.is_suspended(ts_code, "20260105") == new_cal.is_suspended(
            ts_code, "20260105"
        )
    # 已缓存分支：原样返回
    old_again = OldEngine._get_suspend_calendar(old_stub)
    new_again, new_storage2 = new_engine_utils.ensure_suspend_calendar(new_cal, new_storage)
    assert old_again is old_cal and new_again is new_cal and new_storage2 is fake


def test_get_rebalance_dates_equivalence():
    dates = list(pd.bdate_range("2026-01-05", periods=40))
    for freq, k in ((5, 1), (5, 2), (20, 4)):
        stub = SimpleNamespace(rebalance_freq=freq, stagger_tranches=k)
        assert OldEngine._get_rebalance_dates(stub, dates) == new_engine_utils.get_rebalance_dates(
            dates, freq, k
        )


def test_early_rebalance_snapshot_restore_equivalence():
    date = pd.Timestamp("2026-01-20")
    candidates = [("A", 0.9), ("B", 0.8)]
    signal_date = pd.Timestamp("2026-01-16")
    # 快照：净值非 None 与 None 两分支
    for nav in (1.234, None):
        old_stub = SimpleNamespace(
            _last_ranked_candidates=list(candidates),
            _last_signal_date=signal_date,
        )
        if nav is not None:
            old_stub._last_rebalance_nav = nav
        # 旧实现 getattr 防御：缺属性按 None；新侧由调用方显式传 None
        old_snap = OldEngine._snapshot_early_rebalance_state(old_stub, date)
        new_snap = new_engine_utils.snapshot_early_rebalance_state(
            candidates, signal_date, getattr(old_stub, "_last_rebalance_nav", None)
        )
        assert old_snap == new_snap
        # 快照为独立副本：原列表后续修改不影响快照
        candidates.append(("C", 0.7))
        assert old_snap["last_ranked_candidates"] == [("A", 0.9), ("B", 0.8)]
        candidates.pop()

        # 回滚：旧方法就地改写属性；新函数返回待回写字段，由组装体赋值
        old_target = SimpleNamespace(
            _last_ranked_candidates=[("X", 0.1)],
            _last_signal_date=pd.Timestamp("2026-01-19"),
            _last_rebalance_nav=9.99,
        )
        OldEngine._restore_early_rebalance_state(old_target, date, old_snap)
        new_target = SimpleNamespace(
            _last_ranked_candidates=[("X", 0.1)],
            _last_signal_date=pd.Timestamp("2026-01-19"),
            _last_rebalance_nav=9.99,
        )
        for key, value in new_engine_utils.restore_early_rebalance_state(new_snap).items():
            setattr(new_target, "_" + key, value)
        assert vars(old_target) == vars(new_target)
        # nav=None 快照不回写净值（保留现场值 9.99）
        if nav is None:
            assert new_target._last_rebalance_nav == 9.99
            assert "last_rebalance_nav" not in new_engine_utils.restore_early_rebalance_state(
                new_snap
            )


# ══════════════════ core/accounting/state（__init__ 状态初始化，D8 拆分 + D6 摘除） ══════════════════


def _stock_basic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["A", "B", "C"],
            "name": ["甲", "乙", "丙"],
            "industry": ["I1", "I2", "I1"],
        }
    )


def _make_old_engine(**overrides):
    """构造旧引擎（合成 universe/signal/成本模型，不触真实配置与数据）。"""
    kwargs = dict(
        universe=BasicUniverse(_stock_basic_df(), verbose=False),
        signal=old_signal_base.EqualWeightSignal(top_n=5),
        verbose=False,
        cost_model=OldCostModel(
            commission_rate=0.0003, min_commission=5.0, stamp_tax=0.0005, slippage=0.001
        ),
    )
    kwargs.update(overrides)
    return OldEngine(**kwargs)


def _make_new_engine(**overrides):
    """用新状态初始化函数填充属性桩（与 _make_old_engine 同参数）。"""
    kwargs = dict(
        universe=BasicUniverse(_stock_basic_df(), verbose=False),
        signal=new_signal_base.EqualWeightSignal(top_n=5),
        verbose=False,
        cost_model=NewCostModel(
            commission_rate=0.0003, min_commission=5.0, stamp_tax=0.0005, slippage=0.001
        ),
        # 旧实现固定传 self._record_pending_order_event（非 None）；新侧以合成回调对应
        pending_order_event_sink=lambda event: None,
    )
    kwargs.update(overrides)
    engine = SimpleNamespace()
    new_state.init_engine_state(engine, **kwargs)
    return engine


# 需要特殊比对的属性（对象身份/跨类实例）
_SPECIAL_ATTRS = {
    "universe",
    "signal",
    "cost_model",
    "pending_order_manager",
    "stop_loss_config",
    "stop_loss_monitor",
    "stock_basic",
}


def _norm_special(engine, name: str):
    """跨类实例属性归一化为可比值。"""
    value = getattr(engine, name)
    if name in ("universe", "signal"):
        return (type(value).__name__, getattr(value, "name", None), getattr(value, "top_n", None))
    if name == "cost_model":
        return (
            value.commission_rate,
            value.min_commission,
            value.stamp_tax,
            value.slippage,
        )
    if name == "pending_order_manager":
        if value is None:
            return None
        return (
            value.max_retry_count,
            value.max_retry_days,
            value.get_statistics(),
            value.event_sink is not None,
        )
    if name == "stop_loss_config":
        return asdict(value) if value is not None else None
    if name == "stop_loss_monitor":
        if value is None:
            return None
        return (type(value).__name__, asdict(value.config))
    if name == "stock_basic":
        return None if value is None else value.to_dict()
    raise AssertionError(f"未登记的特殊属性: {name}")


def _assert_engine_state_equal(old_engine, new_engine) -> None:
    old_attrs = set(vars(old_engine))
    new_attrs = set(vars(new_engine))
    # D6 摘除生效：新状态无 exposure 关联字段，且与旧状态的差集恰为摘除清单
    for attr in _EXPOSURE_ATTRS:
        assert attr not in new_attrs, f"exposure 字段 {attr} 应已摘除"
        assert attr in old_attrs, f"旧引擎应含 exposure 字段 {attr}（前置假设失效）"
    assert old_attrs - new_attrs == set(_EXPOSURE_ATTRS)
    assert new_attrs - old_attrs == set()
    # 逐值比对共享属性
    for name in sorted(new_attrs):
        old_val = getattr(old_engine, name)
        new_val = getattr(new_engine, name)
        if name in _SPECIAL_ATTRS:
            assert _norm_special(old_engine, name) == _norm_special(
                new_engine, name
            ), f"属性 {name} 不一致"
        else:
            assert type(old_val) is type(new_val), f"属性 {name} 类型不一致"
            assert old_val == new_val, f"属性 {name} 取值不一致"


def test_init_engine_state_default_equivalence(synthetic_cost_settings):
    """默认参数：属性集（减 exposure 摘除集）与逐值一致。"""
    _assert_engine_state_equal(_make_old_engine(), _make_new_engine())
    # 锚定（R2-T4-01）：成本构造确实经由合成 stub（无条件默认读取被替换）
    assert len(synthetic_cost_settings["reads"]) >= 2


def test_init_engine_state_non_default_equivalence(synthetic_cost_settings):
    """非默认组合：分批/持有期/卖出时机/止损/kelly/最小买入阈值。"""
    common = dict(
        initial_capital=500000.0,
        rebalance_freq=10,
        holding_period=7,
        sell_timing="close",
        stagger_tranches=2,
        enable_risk_budget=True,
        vol_window=30,
        vol_epsilon=0.002,
        enable_pending_order=True,
        max_retry_count=3,
        max_retry_days=6,
        enable_position_completion=False,
        completion_window_days=5,
        position_sizing="half_kelly",
        kelly_vol_window=30,
        kelly_max_leverage=0.2,
        enable_early_rebalance_on_empty=False,
        min_buy_value_ratio=0.3,
        max_weight_per_stock=0.25,
    )
    old_engine = _make_old_engine(
        stop_loss_config=OldStopLossConfig(enabled=True, drawdown_pct=15.0), **common
    )
    new_engine = _make_new_engine(
        stop_loss_config=NewStopLossConfig(enabled=True, drawdown_pct=15.0), **common
    )
    _assert_engine_state_equal(old_engine, new_engine)


def test_init_engine_state_pending_disabled_and_defaults(synthetic_cost_settings):
    """延迟订单关闭分支 + holding_period 默认跟随调仓频率。"""
    old_engine = _make_old_engine(enable_pending_order=False)
    new_engine = _make_new_engine(enable_pending_order=False)
    _assert_engine_state_equal(old_engine, new_engine)
    assert new_engine.pending_order_manager is None
    assert new_engine.holding_period == new_engine.rebalance_freq


def test_init_engine_state_validation_errors_equivalence(synthetic_cost_settings):
    """参数校验异常：类型与消息逐字一致（D8 拆分不改变校验语义）。"""
    bad_cases = [
        {"max_weight_per_stock": 0.0},
        {"max_weight_per_stock": 1.5},
        {"max_per_industry": 2},  # 缺 stock_basic
        {"rebalance_freq": 2.5},
        {"rebalance_freq": 0},
        {"sell_timing": "mid"},
        {"stagger_tranches": 1.5},
        {"stagger_tranches": 0},
        {"rebalance_freq": 5, "stagger_tranches": 6},
        {"stagger_tranches": 6},  # 超过调仓频率（freq=5）先触发
        {"rebalance_freq": 10, "stagger_tranches": 6},  # 超过目标持仓数 top_n=5（R1-2 补）
        {"max_weight_per_stock": 0.1},  # 0.1 * top_n=5 < 1
        {"position_sizing": "bad"},
        {"min_buy_value_ratio": -0.1},
    ]
    for kwargs in bad_cases:
        with pytest.raises((ValueError, TypeError)) as old_exc:
            _make_old_engine(**kwargs)
        with pytest.raises((ValueError, TypeError)) as new_exc:
            _make_new_engine(**kwargs)
        assert type(old_exc.value) is type(new_exc.value), f"参数 {kwargs} 异常类型不一致"
        assert str(old_exc.value) == str(new_exc.value), f"参数 {kwargs} 异常消息不一致"


def test_init_engine_state_industry_constraint_branch(monkeypatch, synthetic_cost_settings):
    """行业约束分支：DataLoader 打为合成 stub，行业映射构建结果一致。"""
    shenwan = pd.DataFrame(
        {
            "ts_code": ["A", "B", "C"],
            "sw_l1": ["食品饮料", "银行", "食品饮料"],
        }
    )

    class _FakeLoader:
        def __init__(self, storage):
            self.storage = storage

        def load_shenwan_industry(self):
            return shenwan

    # 两侧 DataLoader 绑定同一假类；行业口径层级打为固定值（免真实配置）
    import src.lazybull.backtest.engine as old_engine_mod
    import src.lazybull.portfolio.industry_constraint as old_ic

    monkeypatch.setattr(old_engine_mod, "DataLoader", _FakeLoader)
    monkeypatch.setattr(new_state, "DataLoader", _FakeLoader)
    monkeypatch.setattr(old_ic, "get_shenwan_level", lambda: "l1")

    common = dict(max_per_industry=2, stock_basic=_stock_basic_df())
    old_engine = _make_old_engine(**common)
    new_engine = _make_new_engine(**common)
    _assert_engine_state_equal(old_engine, new_engine)
    assert old_engine.industry_mapping == new_engine.industry_mapping != {}


class _PassedCounter:
    """inner pytest 运行的通过数记录器（复收证据：复跑非空跑）。"""

    def __init__(self) -> None:
        self.passed = 0

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" and report.outcome == "passed":
            self.passed += 1


def test_t4_suite_forbids_real_config_read(monkeypatch):
    """永久回归（R2-T4-01 复收条件）：禁止读取替身 + 计数断言。

    把新旧两侧成本模块绑定的 ``get_cost_settings`` 替换为**立即抛
    RuntimeError 的禁止读取替身**（评审 §1 复现手法），在该替身下复跑整个
    T4 测试文件（本项除外，防递归）：必须全绿、真实配置读取 0 次。
    任一测试绕过合成 fixture 直接读取真实配置 ⇒ 替身抛出 ⇒ 该测试失败。
    """
    real_reads: List[int] = []

    def _reject_configuration_read():
        real_reads.append(1)
        raise RuntimeError("reviewer2: unexpected configuration read")

    monkeypatch.setattr(old_cost, "get_cost_settings", _reject_configuration_read)
    monkeypatch.setattr(new_cost, "get_cost_settings", _reject_configuration_read)

    counter = _PassedCounter()
    exit_code = pytest.main(
        [
            "-q",
            "-k",
            "not test_t4_suite_forbids_real_config_read",
            str(Path(__file__)),
        ],
        plugins=[counter],
    )
    assert exit_code == 0, "禁止真实配置读取下 T4 套件存在失败项"
    assert real_reads == [], f"真实配置读取发生 {len(real_reads)} 次"
    # 复跑非空跑：当前文件 38 项（含本项），inner 应跑 37 项；下限防未来漂移
    assert counter.passed >= 36, f"inner 复跑通过数异常: {counter.passed}"


# ══════════════════ core/execution/pending_order（整模块复制件） ══════════════════


def test_pending_order_dataclass_fields():
    _assert_dataclass_fields_equal(old_pending.PendingOrder, new_pending.PendingOrder)
    d0 = pd.Timestamp("2026-01-05")
    old_order = old_pending.PendingOrder("A", "buy", 10000.0, d0, d0)
    new_order = new_pending.PendingOrder("A", "buy", 10000.0, d0, d0)
    assert repr(old_order) == repr(new_order)
    assert asdict(old_order) == asdict(new_order)


def test_pending_order_manager_sequence_equivalence():
    """相同操作序列下事件流、统计与终态逐项一致（含 sell 不过期语义）。"""
    old_events: List[Dict] = []
    new_events: List[Dict] = []
    old_mgr = old_pending.PendingOrderManager(
        max_retry_count=2, max_retry_days=5, event_sink=old_events.append
    )
    new_mgr = new_pending.PendingOrderManager(
        max_retry_count=2, max_retry_days=5, event_sink=new_events.append
    )
    d0 = pd.Timestamp("2026-01-05")
    d1 = pd.Timestamp("2026-01-06")
    for mgr in (old_mgr, new_mgr):
        mgr.add_order("A", "buy", d0, d0, target_value=10000.0, reason="涨停")
        mgr.add_order("A", "buy", d1, d0, reason="停牌")  # 重试 +1
        mgr.add_order("B", "sell", d0, d0, reason="停牌")  # sell 订单
        mgr.add_order("C", "buy", d0, d0, target_value=5000.0, reason="跌停")
    assert old_events == new_events
    assert old_mgr.get_statistics() == new_mgr.get_statistics()
    assert old_mgr.get_pending_count() == new_mgr.get_pending_count() == 3

    # 重试轮次：A 再重试 2 次后超过 max_retry_count=2
    old_retry, old_expired = old_mgr.get_orders_to_retry(d1)
    new_retry, new_expired = new_mgr.get_orders_to_retry(d1)
    assert [o.stock for o in old_retry] == [o.stock for o in new_retry]
    assert old_expired == [] and new_expired == []
    for mgr in (old_mgr, new_mgr):
        mgr.add_order("A", "buy", d1, d0, reason="停牌")  # retry=3 > 2
    old_retry, old_expired = old_mgr.get_orders_to_retry(d1)
    new_retry, new_expired = new_mgr.get_orders_to_retry(d1)
    assert [o.stock for o in old_retry] == [o.stock for o in new_retry] == ["B", "C"]
    assert [o.stock for o in old_expired] == [o.stock for o in new_expired] == ["A"]
    assert old_events == new_events
    assert old_events[-1]["type"] == "expired_retry"

    # 超期分支：C 创建于 d0，20 天后 > max_retry_days=5；sell 的 B 不过期
    d_late = d0 + pd.Timedelta(days=20)
    old_retry, old_expired = old_mgr.get_orders_to_retry(d_late)
    new_retry, new_expired = new_mgr.get_orders_to_retry(d_late)
    assert [o.stock for o in old_retry] == [o.stock for o in new_retry] == ["B"]
    assert [o.stock for o in old_expired] == [o.stock for o in new_expired] == ["C"]
    assert old_events == new_events
    assert old_events[-1]["type"] == "expired_days"

    # 成功/手动移除/清空
    for mgr in (old_mgr, new_mgr):
        mgr.mark_success(d1, "B", "sell")
        assert mgr.has_order("B", "sell") is False
        mgr.add_order("D", "buy", d1, d1, target_value=100.0, reason="停牌")
        mgr.remove_order("D", "buy")
    assert old_events == new_events
    assert old_mgr.get_statistics() == new_mgr.get_statistics()
    for mgr in (old_mgr, new_mgr):
        mgr.add_order("E", "buy", d1, d1, target_value=100.0, reason="停牌")
        mgr.clear_all()
    assert old_mgr.get_pending_count() == new_mgr.get_pending_count() == 0
    assert old_mgr.get_statistics() == new_mgr.get_statistics()
    assert old_mgr.get_all_orders() == new_mgr.get_all_orders() == []


def test_pending_order_manager_no_event_sink_branch():
    """无 event_sink 分支（日志路径）：行为与统计一致。"""
    old_mgr = old_pending.PendingOrderManager(max_retry_count=1, max_retry_days=1)
    new_mgr = new_pending.PendingOrderManager(max_retry_count=1, max_retry_days=1)
    d0 = pd.Timestamp("2026-01-05")
    for mgr in (old_mgr, new_mgr):
        mgr.add_order("A", "buy", d0, d0, target_value=1.0, reason="停牌")
        mgr.add_order("A", "buy", d0, d0, reason="停牌")  # retry=2 > 1
    old_retry, old_expired = old_mgr.get_orders_to_retry(d0)
    new_retry, new_expired = new_mgr.get_orders_to_retry(d0)
    assert [o.stock for o in old_retry] == [o.stock for o in new_retry] == []
    assert [o.stock for o in old_expired] == [o.stock for o in new_expired] == ["A"]
    assert old_mgr.get_statistics() == new_mgr.get_statistics()


# ══════════════════ core/signal/base（整模块复制件，T5 组装前置 V3R-03） ══════════════════


def test_signal_base_abstract_equivalence():
    with pytest.raises(TypeError):
        old_signal_base.Signal("x")
    with pytest.raises(TypeError):
        new_signal_base.Signal("x")


def test_equal_weight_signal_equivalence():
    universe = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    date = pd.Timestamp("2026-01-05")
    for top_n in (1, 5, 30):  # 含 top_n > len(universe) 分支
        old_sig = old_signal_base.EqualWeightSignal(top_n=top_n)
        new_sig = new_signal_base.EqualWeightSignal(top_n=top_n)
        assert old_sig.name == new_sig.name and old_sig.top_n == new_sig.top_n
        assert old_sig.generate(date, universe, {}) == new_sig.generate(date, universe, {})
        # 默认 generate_ranked（按权重降序，等权下稳定保序）
        assert old_sig.generate_ranked(date, universe, {}) == new_sig.generate_ranked(
            date, universe, {}
        )
    # 空股票池分支
    assert (
        old_signal_base.EqualWeightSignal(5).generate(date, [], {})
        == new_signal_base.EqualWeightSignal(5).generate(date, [], {})
        == {}
    )


def test_factor_signal_equivalence():
    universe = ["S1", "S2", "S3"]
    date = pd.Timestamp("2026-01-05")
    for weight_method in ("equal", "score"):
        old_sig = old_signal_base.FactorSignal(top_n=2, weight_method=weight_method)
        new_sig = new_signal_base.FactorSignal(top_n=2, weight_method=weight_method)
        assert old_sig.name == new_sig.name
        assert old_sig.weight_method == new_sig.weight_method
        assert old_sig.generate(date, universe, {}) == new_sig.generate(date, universe, {})


def test_generate_ranked_default_sorting_equivalence():
    """默认 generate_ranked：从 generate() 结果按权重降序推导。"""

    class _OldCustom(old_signal_base.Signal):
        def generate(self, date, universe, data):
            return {"A": 0.1, "B": 0.5, "C": 0.3}

    class _NewCustom(new_signal_base.Signal):
        def generate(self, date, universe, data):
            return {"A": 0.1, "B": 0.5, "C": 0.3}

    date = pd.Timestamp("2026-01-05")
    old_ranked = _OldCustom("c").generate_ranked(date, [], {})
    new_ranked = _NewCustom("c").generate_ranked(date, [], {})
    assert old_ranked == new_ranked == [("B", 0.5), ("C", 0.3), ("A", 0.1)]


# ══════════════════ core/accounting/holdings_snapshot（估值回退 D4 第 3 处） ══════════════════


def _snapshot_mixin(
    cls,
    prices: Dict[str, Optional[float]],
    positions: Dict[str, Dict],
    holding_period: int = 5,
    record: bool = True,
):
    """构造快照 mixin 实例（_get_trade_price 以合成价格表注入）。"""
    obj = cls()
    obj.record_holdings_snapshot = record
    obj.positions = positions
    obj.holding_period = holding_period
    obj.holdings_snapshots = []
    obj._get_trade_price = lambda date, stock: prices.get(stock)
    return obj


def _snapshot_dates(n: int = 10):
    dates = list(pd.bdate_range("2026-01-05", periods=n))
    return dates, {d: i for i, d in enumerate(dates)}


def _snapshot_case(cls, prices, positions, holding_period: int = 5, record: bool = True):
    """构造快照 mixin 实例：持仓输入 deepcopy 隔离，并返回调用前独立副本（R2-T4-02）。

    快照是只读旁路（不读写任何参与买卖判断的状态）；返回的 pristine 副本
    供调用方在记录后断言持仓状态未被污染。
    """
    obj = _snapshot_mixin(cls, prices, copy.deepcopy(positions), holding_period, record)
    return obj, copy.deepcopy(positions)


def test_holdings_snapshot_fallback_equivalence():
    """估值回退第 3 处（旧 holdings_snapshot.py:62-66）：成交价 → 缓存价 → 买入价。"""
    dates, date_to_idx = _snapshot_dates()
    date = dates[4]
    prices = {"A": 10.0}  # B/C 无价
    positions = {
        "A": {"shares": 100, "buy_date": dates[1], "buy_trade_price": 9.0, "signal_date": dates[0]},
        # B：无价 → 回退 last_known_price
        "B": {
            "shares": 200,
            "buy_date": dates[0],
            "buy_trade_price": 19.0,
            "last_known_price": 18.5,
        },
        # C：无价无缓存 → 回退买入价
        "C": {"shares": 50, "buy_date": dates[2], "buy_trade_price": 5.0},
        # D：buy_date 不在交易日映射 → holding 字段全 None
        "D": {"shares": 10, "buy_date": pd.Timestamp("1999-01-04"), "buy_trade_price": 3.0},
    }
    prices["D"] = 3.1
    old_obj, old_pristine = _snapshot_case(
        old_snapshot.BacktestHoldingsSnapshotMixin, prices, positions
    )
    new_obj, new_pristine = _snapshot_case(
        new_snapshot.BacktestHoldingsSnapshotMixin, prices, positions
    )
    old_obj._record_holdings_snapshot(date, 5000.0, dates, date_to_idx)
    new_obj._record_holdings_snapshot(date, 5000.0, dates, date_to_idx)
    assert old_obj.holdings_snapshots == new_obj.holdings_snapshots
    # 只读旁路硬约束（R2-T4-02）：快照记录之外持仓状态不变（新旧两侧各自锁定）
    assert old_obj.positions == old_pristine
    assert new_obj.positions == new_pristine
    by_stock = {r["ts_code"]: r for r in new_obj.holdings_snapshots}
    assert by_stock["A"]["market_value"] == 100 * 10.0
    assert by_stock["B"]["market_value"] == 200 * 18.5  # 回退缓存价
    assert by_stock["C"]["market_value"] == 50 * 5.0  # 回退买入价
    assert by_stock["A"]["holding_days"] == 3
    assert by_stock["A"]["planned_exit_date"] == dates[1 + 5]
    assert by_stock["A"]["remaining_intervals"] == (1 + 5) - 4 - 1
    assert by_stock["D"]["holding_days"] is None
    assert by_stock["D"]["planned_exit_date"] is None
    assert by_stock["D"]["remaining_intervals"] is None


def test_holdings_snapshot_exit_beyond_window_equivalence():
    """到期日超出回测窗口 → 记 None；组合总值为 0 → weight 记 None。"""
    dates, date_to_idx = _snapshot_dates(n=5)
    date = dates[4]
    prices = {"A": 10.0}
    positions = {"A": {"shares": 100, "buy_date": dates[2], "buy_trade_price": 9.0}}
    old_obj, old_pristine = _snapshot_case(
        old_snapshot.BacktestHoldingsSnapshotMixin, prices, positions, holding_period=20
    )
    new_obj, new_pristine = _snapshot_case(
        new_snapshot.BacktestHoldingsSnapshotMixin, prices, positions, holding_period=20
    )
    old_obj._record_holdings_snapshot(date, 0.0, dates, date_to_idx)
    new_obj._record_holdings_snapshot(date, 0.0, dates, date_to_idx)
    assert old_obj.holdings_snapshots == new_obj.holdings_snapshots
    assert old_obj.positions == old_pristine
    assert new_obj.positions == new_pristine
    record = new_obj.holdings_snapshots[0]
    assert record["planned_exit_date"] is None and record["remaining_intervals"] is None
    assert record["weight"] is None


def test_holdings_snapshot_guard_branches_equivalence():
    """开关关闭 / 空仓 → 不记录；交易日映射缺失 → 同消息 ValueError。"""
    dates, date_to_idx = _snapshot_dates()
    positions = {"A": {"shares": 100, "buy_date": dates[0], "buy_trade_price": 9.0}}
    for record, pos in ((False, positions), (True, {})):
        old_obj, old_pristine = _snapshot_case(
            old_snapshot.BacktestHoldingsSnapshotMixin, {"A": 10.0}, pos, record=record
        )
        new_obj, new_pristine = _snapshot_case(
            new_snapshot.BacktestHoldingsSnapshotMixin, {"A": 10.0}, pos, record=record
        )
        old_obj._record_holdings_snapshot(dates[1], 1000.0, dates, date_to_idx)
        new_obj._record_holdings_snapshot(dates[1], 1000.0, dates, date_to_idx)
        assert old_obj.holdings_snapshots == new_obj.holdings_snapshots == []
        assert old_obj.positions == old_pristine
        assert new_obj.positions == new_pristine
    # 映射缺失
    missing = pd.Timestamp("2026-06-01")
    for cls in (
        old_snapshot.BacktestHoldingsSnapshotMixin,
        new_snapshot.BacktestHoldingsSnapshotMixin,
    ):
        obj = _snapshot_mixin(cls, {"A": 10.0}, positions)
        with pytest.raises(ValueError) as exc:
            obj._record_holdings_snapshot(missing, 1000.0, dates, date_to_idx)
        assert str(exc.value) == f"持仓快照需要 {missing} 在交易日序列中的位置，当前映射缺失"


def test_get_holdings_snapshot_equivalence():
    dates, date_to_idx = _snapshot_dates()
    positions = {"A": {"shares": 100, "buy_date": dates[0], "buy_trade_price": 9.0}}
    old_obj, old_pristine = _snapshot_case(
        old_snapshot.BacktestHoldingsSnapshotMixin, {"A": 10.0}, positions
    )
    new_obj, new_pristine = _snapshot_case(
        new_snapshot.BacktestHoldingsSnapshotMixin, {"A": 10.0}, positions
    )
    # 空记录：列齐空帧
    old_empty = old_obj.get_holdings_snapshot()
    new_empty = new_obj.get_holdings_snapshot()
    assert list(old_empty.columns) == list(new_empty.columns)
    assert len(old_empty) == len(new_empty) == 0
    # 有记录：帧一致
    old_obj._record_holdings_snapshot(dates[1], 1000.0, dates, date_to_idx)
    new_obj._record_holdings_snapshot(dates[1], 1000.0, dates, date_to_idx)
    pd.testing.assert_frame_equal(old_obj.get_holdings_snapshot(), new_obj.get_holdings_snapshot())
    assert old_obj.positions == old_pristine
    assert new_obj.positions == new_pristine


def test_holdings_snapshot_readonly_detector_catches_write():
    """变异自证（R2-T4-02）：注入错误写入后，只读断言必须失败。

    评审反例手法：包装新侧 ``_record_holdings_snapshot``，保留原快照输出后
    故意向首个持仓追加错误写入——输出仍然正确，但持仓已被污染；
    deepcopy 前副本比对必须捕获（AssertionError），否则断言形同虚设。
    """
    dates, date_to_idx = _snapshot_dates()
    positions = {"A": {"shares": 100, "buy_date": dates[0], "buy_trade_price": 9.0}}
    obj, pristine = _snapshot_case(
        new_snapshot.BacktestHoldingsSnapshotMixin, {"A": 10.0}, positions
    )
    original = new_snapshot.BacktestHoldingsSnapshotMixin._record_holdings_snapshot

    def _poisoned(self, date, portfolio_value, trading_dates, date_to_idx):
        original(self, date, portfolio_value, trading_dates, date_to_idx)
        if getattr(self, "record_holdings_snapshot", False) and self.positions:
            next(iter(self.positions.values()))["last_known_price"] = 12345.0

    obj._record_holdings_snapshot = _poisoned.__get__(obj)
    obj._record_holdings_snapshot(dates[1], 1000.0, dates, date_to_idx)
    # 快照输出本身仍正确（污染不影响输出）
    assert len(obj.holdings_snapshots) == 1
    # 错误写入确已发生
    assert obj.positions["A"]["last_known_price"] == 12345.0
    # 只读断言（与等价测试同款）必须失败 ⇒ 检测器有效
    with pytest.raises(AssertionError):
        assert obj.positions == pristine, "快照只读约束被污染"

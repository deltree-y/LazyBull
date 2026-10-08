"""P2a-T1 新旧逐值一致等价测试（双源漂移锁定，B5/C5）。

T0 实施规划 v4 §4-T1 验收口径：
- dataclass = 字段集 + 默认值逐项比对；
- 纯函数 = 输出比对（同输入同输出，含异常类型与消息）；
- 双源一致性 = 本文件的全部断言（切换前新旧模块双源并存，本测试锁漂移）。

不依赖真实配置与真实数据：所有输入为合成 fixture；成本默认值分支经
``synthetic_cost_settings`` fixture 把新旧两侧的 ``get_cost_settings`` 打为同一
合成 stub（R2-T1-02 处置），并保留"两侧绑定同一读取函数"的身份断言（B5）。
"""

import argparse
import dataclasses
import warnings
from dataclasses import asdict
from typing import List

import numpy as np
import pandas as pd
import pytest

# ── 旧模块（生产在用，迁移期不修改） ──
from src.lazybull.common import cost as old_cost
from src.lazybull.common import date_utils as old_date_utils
from src.lazybull.common import feature_utils as old_feature_utils
from src.lazybull.common import sidecar_schema as old_sidecar
from src.lazybull.common import suspend_calendar as old_suspend
from src.lazybull.common import trade_status as old_trade_status
from src.lazybull.common import trading_config as old_trading_config
from src.lazybull.common import xgboost_compat as old_xgb_compat
from src.lazybull.risk import stop_loss as old_stop_loss
from src.lazybull.risk import stop_loss_checker as old_checker
from src.lazybull.trading import sell_rules as old_sell_rules
from src.lazybull.trading import stagger as old_stagger

# ── v2 复制件（P2a-T1 交付） ──
from src.lazybull.v2.common import cost as new_cost
from src.lazybull.v2.common import date_utils as new_date_utils
from src.lazybull.v2.common import feature_utils as new_feature_utils
from src.lazybull.v2.common import suspend_calendar as new_suspend
from src.lazybull.v2.common import table_schema as new_table_schema
from src.lazybull.v2.common import trade_status as new_trade_status
from src.lazybull.v2.common import trading_config as new_trading_config
from src.lazybull.v2.common import xgboost_compat as new_xgb_compat
from src.lazybull.v2.common.rules import sell_rules as new_sell_rules
from src.lazybull.v2.common.rules import stagger as new_stagger
from src.lazybull.v2.common.rules import stop_loss as new_stop_loss
from src.lazybull.v2.common.rules import stop_loss_checker as new_checker

_MISSING = dataclasses.MISSING


def _assert_dataclass_fields_equal(old_cls, new_cls) -> None:
    """dataclass 字段集 + 默认值逐项比对（T1 验收口径）。"""
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


# ══════════════════ rules/sell_rules ══════════════════


def test_sell_rules_dataclass_fields():
    _assert_dataclass_fields_equal(old_sell_rules.RebalanceSellDecision,
                                   new_sell_rules.RebalanceSellDecision)


def test_is_holding_period_exit_due_grid():
    for holding_days in range(0, 26):
        for holding_period in (1, 2, 5, 20):
            assert old_sell_rules.is_holding_period_exit_due(
                holding_days, holding_period
            ) == new_sell_rules.is_holding_period_exit_due(holding_days, holding_period)


def test_min_holding_days_for_rebalance_sell_grid():
    for freq in (1, 2, 5, 20):
        for floor in (0, 1, 3):
            assert old_sell_rules.min_holding_days_for_rebalance_sell(
                freq, floor
            ) == new_sell_rules.min_holding_days_for_rebalance_sell(freq, floor)


def test_select_rebalance_sell_candidates_equivalence():
    holding_days_map = {
        "A": 25,  # 在卖出队列 → 跳过
        "B": 25,  # 受保护 → 跳过
        "C": 25,  # 在新目标中 → 跳过
        "D": 3,   # 未满持有期 → 跳过
        "E": 25,  # 应卖出
        "F": None,  # 持有天数未知 → 不做年轻过滤，应卖出
    }
    kwargs = dict(
        min_holding_days=19,
        target_codes={"C"},
        protected_codes={"B"},
        queued_codes={"A"},
    )
    old_decision = old_sell_rules.select_rebalance_sell_candidates(holding_days_map, **kwargs)
    new_decision = new_sell_rules.select_rebalance_sell_candidates(holding_days_map, **kwargs)
    assert old_decision.sells == new_decision.sells == ["E", "F"]
    assert asdict(old_decision) == asdict(new_decision)
    # 缺省可选集合（None 分支）
    old_bare = old_sell_rules.select_rebalance_sell_candidates({"X": 25}, 19)
    new_bare = new_sell_rules.select_rebalance_sell_candidates({"X": 25}, 19)
    assert asdict(old_bare) == asdict(new_bare)


# ══════════════════ rules/stagger ══════════════════


def test_stagger_tranche_count_and_fraction_grid():
    for total in (1, 2, 5, 20, 30):
        for k in range(1, total + 1):
            for idx in range(k):
                assert old_stagger.get_tranche_target_count(
                    idx, total, k
                ) == new_stagger.get_tranche_target_count(idx, total, k)
                assert old_stagger.get_tranche_capital_fraction(
                    idx, total, k
                ) == new_stagger.get_tranche_capital_fraction(idx, total, k)


def test_stagger_invalid_input_errors():
    bad_calls = [
        (4, 20, 4),    # 批次索引越界（== 批次数）
        (-1, 20, 4),   # 批次索引为负
        (0, 0, 1),     # 总目标非法
        (0, 20, 21),   # 批次数超总目标
    ]
    for args in bad_calls:
        with pytest.raises(ValueError) as old_exc:
            old_stagger.get_tranche_target_count(*args)
        with pytest.raises(ValueError) as new_exc:
            new_stagger.get_tranche_target_count(*args)
        assert str(old_exc.value) == str(new_exc.value)


def test_compute_tranche_schedule_equivalence():
    dates = [f"2026{(i // 28) + 1:02d}{(i % 28) + 1:02d}" for i in range(60)]
    for freq in (5, 20):
        for k in (1, 2, 3, 4):
            if k > freq:
                continue
            assert old_stagger.compute_tranche_schedule(
                dates, freq, k
            ) == new_stagger.compute_tranche_schedule(dates, freq, k)


def test_build_tranche_schedule_from_anchor_equivalence():
    dates = [f"2026{(i // 28) + 1:02d}{(i % 28) + 1:02d}" for i in range(60)]
    for anchor in (dates[0], dates[10], dates[-1], "19990101"):  # 末项 = 锚不在列表
        assert old_stagger.build_tranche_schedule_from_anchor(
            anchor, dates, 20, 3
        ) == new_stagger.build_tranche_schedule_from_anchor(anchor, dates, 20, 3)


# ══════════════════ rules/stop_loss + stop_loss_checker ══════════════════


def test_stop_loss_config_fields_and_factory():
    _assert_dataclass_fields_equal(old_stop_loss.StopLossConfig, new_stop_loss.StopLossConfig)
    cfg_dict = {
        "stop_loss_enabled": True,
        "stop_loss_drawdown_pct": 15.0,
        "stop_loss_consecutive_limit_down": 3,
        "stop_loss_post_action": "buy_alternative",
    }
    old_cfg = old_stop_loss.create_stop_loss_config_from_dict(cfg_dict)
    new_cfg = new_stop_loss.create_stop_loss_config_from_dict(cfg_dict)
    assert asdict(old_cfg) == asdict(new_cfg)
    # 空字典默认值分支
    assert asdict(old_stop_loss.create_stop_loss_config_from_dict({})) == asdict(
        new_stop_loss.create_stop_loss_config_from_dict({})
    )


def test_stop_loss_monitor_sequence_equivalence():
    old_monitor = old_stop_loss.StopLossMonitor(
        old_stop_loss.StopLossConfig(enabled=True, drawdown_pct=20.0,
                                     consecutive_limit_down_days=2)
    )
    new_monitor = new_stop_loss.StopLossMonitor(
        new_stop_loss.StopLossConfig(enabled=True, drawdown_pct=20.0,
                                     consecutive_limit_down_days=2)
    )
    sequence = [
        ("A", 10.0, 9.0, False),    # 未触发
        ("A", 10.0, 7.9, False),    # 回撤 >20% → 触发
        ("B", 10.0, 9.5, True),     # 跌停第 1 天
        ("B", 10.0, 9.0, True),     # 跌停第 2 天 → 触发
        ("C", 10.0, 9.5, True),     # 跌停第 1 天
        ("C", 10.0, 9.6, False),    # 非跌停 → 计数重置
        ("C", 10.0, 9.5, True),     # 重新计第 1 天
        ("D", 0.0, 9.0, False),     # 买入价无效
    ]
    for stock, buy, cur, limit_down in sequence:
        old_ret = old_monitor.check_stop_loss(stock, buy, cur, limit_down)
        new_ret = new_monitor.check_stop_loss(stock, buy, cur, limit_down)
        old_norm = (old_ret[0], old_ret[1].value if old_ret[1] else None, old_ret[2])
        new_norm = (new_ret[0], new_ret[1].value if new_ret[1] else None, new_ret[2])
        assert old_norm == new_norm
    assert old_monitor.consecutive_limit_down_days == new_monitor.consecutive_limit_down_days
    # 禁用分支
    old_off = old_stop_loss.StopLossMonitor(old_stop_loss.StopLossConfig(enabled=False))
    new_off = new_stop_loss.StopLossMonitor(new_stop_loss.StopLossConfig(enabled=False))
    assert old_off.check_stop_loss("A", 10.0, 1.0, True) == new_off.check_stop_loss(
        "A", 10.0, 1.0, True
    ) == (False, None, None)
    # remove_position / reset
    old_monitor.remove_position("B")
    new_monitor.remove_position("B")
    assert old_monitor.consecutive_limit_down_days == new_monitor.consecutive_limit_down_days
    old_monitor.reset()
    new_monitor.reset()
    assert old_monitor.consecutive_limit_down_days == new_monitor.consecutive_limit_down_days == {}


class _FakeSuspendCalendar:
    """停牌日历桩（仅实现 check_positions_stop_loss 消费的接口）。"""

    def __init__(self, suspended: set):
        self._suspended = suspended

    def is_suspended(self, ts_code: str, trade_date: str) -> bool:
        return ts_code in self._suspended


def test_stop_loss_checker_action_dataclass_fields():
    _assert_dataclass_fields_equal(old_checker.StopLossAction, new_checker.StopLossAction)


def test_check_positions_stop_loss_equivalence():
    positions = {
        "SUSP": {"buy_trade_price": 10.0},          # 停牌跳过
        "NOPRICE": {"buy_trade_price": 10.0},       # 无行情跳过
        "BADBUY": {"buy_trade_price": 0.0},         # 买入价无效跳过
        "TRIG": {"buy_trade_price": 10.0},          # 回撤触发
        "OK": {"buy_trade_price": 10.0},            # 正常
    }
    prices = {"TRIG": 6.0, "OK": 10.5, "BADBUY": 9.0}
    limit_down = {"TRIG": False, "OK": False, "BADBUY": False}
    calendar = _FakeSuspendCalendar({"SUSP"})
    old_monitor = old_stop_loss.StopLossMonitor(
        old_stop_loss.StopLossConfig(enabled=True, drawdown_pct=20.0)
    )
    new_monitor = new_stop_loss.StopLossMonitor(
        new_stop_loss.StopLossConfig(enabled=True, drawdown_pct=20.0)
    )
    old_actions = old_checker.check_positions_stop_loss(
        positions, old_monitor, prices, limit_down, calendar, "20260105", verbose=False
    )
    new_actions = new_checker.check_positions_stop_loss(
        positions, new_monitor, prices, limit_down, calendar, "20260105", verbose=False
    )
    assert [asdict(a) for a in old_actions] == [asdict(a) for a in new_actions]
    assert [a.ts_code for a in new_actions] == ["TRIG"]
    # 空持仓与 None 日历分支
    assert old_checker.check_positions_stop_loss(
        {}, old_monitor, prices, limit_down, None, "20260105", verbose=False
    ) == new_checker.check_positions_stop_loss(
        {}, new_monitor, prices, limit_down, None, "20260105", verbose=False
    ) == []


def test_check_positions_stop_loss_suspension_abort_equivalence():
    """停牌数据异常 ⇒ 整轮中止并返回已收集动作（D8 提取后逐点等价）。"""
    # 先排一只可触发的持仓，再排一只触发中止的：返回列表应只含前者
    positions = {
        "TRIG": {"buy_trade_price": 10.0},
        "ABORT": {"buy_trade_price": 10.0},
    }
    prices = {"TRIG": 6.0, "ABORT": 9.0}
    limit_down = {"TRIG": False, "ABORT": False}

    class _SelectiveCalendar:
        def is_suspended(self, ts_code: str, trade_date: str) -> bool:
            if ts_code == "ABORT":
                raise FileNotFoundError("停牌数据文件缺失")
            return False

    old_monitor = old_stop_loss.StopLossMonitor(
        old_stop_loss.StopLossConfig(enabled=True, drawdown_pct=20.0)
    )
    new_monitor = new_stop_loss.StopLossMonitor(
        new_stop_loss.StopLossConfig(enabled=True, drawdown_pct=20.0)
    )
    old_actions = old_checker.check_positions_stop_loss(
        positions, old_monitor, prices, limit_down, _SelectiveCalendar(),
        "20260105", verbose=False
    )
    new_actions = new_checker.check_positions_stop_loss(
        positions, new_monitor, prices, limit_down, _SelectiveCalendar(),
        "20260105", verbose=False
    )
    assert [asdict(a) for a in old_actions] == [asdict(a) for a in new_actions]
    assert [a.ts_code for a in new_actions] == ["TRIG"]


# ══════════════════ common/date_utils ══════════════════


def test_date_utils_scalar_functions_grid():
    inputs = [
        "20230101", "2023-01-01", "2023/01/01",
        pd.Timestamp("2023-01-01"), np.datetime64("2023-01-01"),
    ]
    for value in inputs:
        assert old_date_utils.to_trade_date_str(value) == new_date_utils.to_trade_date_str(value)
        assert old_date_utils.to_timestamp(value) == new_date_utils.to_timestamp(value)
    for bad in ("not-a-date", "2023011"):
        for fn_name in ("to_trade_date_str", "to_timestamp"):
            with pytest.raises(ValueError) as old_exc:
                getattr(old_date_utils, fn_name)(bad)
            with pytest.raises(ValueError) as new_exc:
                getattr(new_date_utils, fn_name)(bad)
            assert str(old_exc.value) == str(new_exc.value)


def test_normalize_to_yyyymmdd_grid():
    values = [
        None, np.nan, pd.NaT, pd.Timestamp("2023-01-01"),
        "2023-01-01", "20230101", " 20230101 ", "", "nan", "2023011", "abc",
        20230101,
    ]
    for value in values:
        assert old_date_utils.normalize_to_yyyymmdd(value) == new_date_utils.normalize_to_yyyymmdd(
            value
        )


def test_normalize_date_columns_equivalence():
    df = pd.DataFrame({
        "d1": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "d2": ["2023-01-01", None],
        "v": [1.0, 2.0],
    })
    for to_str in (True, False):
        old_out = old_date_utils.normalize_date_columns(df, ["d1", "d2", "missing"], to_str)
        new_out = new_date_utils.normalize_date_columns(df, ["d1", "d2", "missing"], to_str)
        pd.testing.assert_frame_equal(old_out, new_out)


def test_normalize_series_and_holding_days():
    series = pd.Series(["2023-01-01", None, "bad", pd.Timestamp("2023-01-03")])
    pd.testing.assert_series_equal(
        old_date_utils.normalize_series_to_yyyymmdd(series),
        new_date_utils.normalize_series_to_yyyymmdd(series),
    )
    trade_dates = ["20230102", "20230103", "20230104", "20230105"]
    cases = [
        ("20230102", "20230105"), ("20230105", "20230102"),
        ("20230101", "20230105"), ("", "20230105"), ("20230102", ""),
    ]
    for buy, cur in cases:
        assert old_date_utils.calc_holding_trade_days(
            buy, cur, trade_dates
        ) == new_date_utils.calc_holding_trade_days(buy, cur, trade_dates)


def test_is_recent_date_str_equivalence():
    recent = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    for value in ("20200101", recent, "abc"):
        assert old_date_utils.is_recent_date_str(value) == new_date_utils.is_recent_date_str(value)


# ══════════════════ common/cost ══════════════════


def test_cost_get_cost_settings_identity():
    """双源锁定（B5）：新旧成本模块绑定同一个只读 config 读取函数对象。"""
    assert old_cost.get_cost_settings is new_cost.get_cost_settings


@pytest.fixture()
def synthetic_cost_settings(monkeypatch):
    """合成成本配置（R2-T1-02 处置）：新旧两侧共同消费的 ``get_cost_settings``
    一律打为同一 stub，测试不再触碰真实 config 单例；monkeypatch 结束自动恢复。"""
    settings = {
        "commission_rate": 0.0003,
        "min_commission": 5.0,
        "stamp_tax": 0.0005,
        "slippage": 0.001,
    }

    def _stub() -> dict:
        return dict(settings)

    monkeypatch.setattr(old_cost, "get_cost_settings", _stub)
    monkeypatch.setattr(new_cost, "get_cost_settings", _stub)
    return settings


def test_cost_model_explicit_params_equivalence(synthetic_cost_settings):
    kwargs = dict(commission_rate=0.0003, min_commission=5.0, stamp_tax=0.0005, slippage=0.001)
    old_model = old_cost.CostModel(**kwargs)
    new_model = new_cost.CostModel(**kwargs)
    for amount in (0.0, 1000.0, 25600.0, 3333.33, 1e6):
        assert old_model.calculate_commission(amount) == new_model.calculate_commission(amount)
        assert old_model.calculate_stamp_tax(amount) == new_model.calculate_stamp_tax(amount)
        assert old_model.calculate_slippage(amount) == new_model.calculate_slippage(amount)
        assert old_model.calculate_buy_cost(amount) == new_model.calculate_buy_cost(amount)
        assert old_model.calculate_sell_cost(amount) == new_model.calculate_sell_cost(amount)
        assert old_model.calculate_total_cost(amount, amount) == new_model.calculate_total_cost(
            amount, amount
        )


def test_cost_model_default_params_relational(synthetic_cost_settings):
    """默认值回退分支：新旧两侧消费同一合成配置 stub，断言相等且取值来自合成配置。"""
    old_model = old_cost.CostModel()
    new_model = new_cost.CostModel()
    for attr in ("commission_rate", "min_commission", "stamp_tax", "slippage"):
        assert getattr(old_model, attr) == getattr(new_model, attr)
        # 锚定合成配置（而非仅“两侧相等”），确保默认值确实走了被 stub 的回退分支
        assert getattr(new_model, attr) == synthetic_cost_settings[attr]
    # 部分参数覆盖分支
    old_partial = old_cost.CostModel(commission_rate=0.001)
    new_partial = new_cost.CostModel(commission_rate=0.001)
    assert old_partial.commission_rate == new_partial.commission_rate == 0.001
    for attr in ("min_commission", "stamp_tax", "slippage"):
        assert getattr(old_partial, attr) == getattr(new_partial, attr)
        assert getattr(new_partial, attr) == synthetic_cost_settings[attr]


# ══════════════════ common/trade_status ══════════════════


def _quote_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ts_code": ["A", "B", "C", "D"],
        "trade_date": ["20260105"] * 4,
        "is_suspended": [1, 0, 0, 0],
        "is_limit_up": [0, 1, 0, 0],
        "is_limit_down": [0, 0, 1, 0],
        "vol": [0, 100, 100, 100],
        "close": [10.0, 11.0, 9.0, 10.5],
        "pct_chg": [0.0, 10.0, -10.0, 1.0],
    })


def test_evaluate_trade_status_grid():
    statuses = [
        {"is_suspended": 1},
        {"is_limit_up": 1},
        {"is_limit_down": 1},
        {"tradable": 0},
        {},
    ]
    for status in statuses:
        for action in ("buy", "sell"):
            for require in (False, True):
                assert old_trade_status.evaluate_trade_status(
                    status, action, require_tradable=require
                ) == new_trade_status.evaluate_trade_status(
                    status, action, require_tradable=require
                )


def test_trade_status_query_functions_equivalence():
    df = _quote_df()
    for ts_code in ("A", "B", "C", "D", "MISSING"):
        for fn_name in ("is_suspended", "is_limit_up", "is_limit_down"):
            assert getattr(old_trade_status, fn_name)(
                ts_code, "20260105", df
            ) == getattr(new_trade_status, fn_name)(ts_code, "20260105", df)
        for action in ("buy", "sell"):
            assert old_trade_status.is_tradeable(
                ts_code, "20260105", df, action
            ) == new_trade_status.is_tradeable(ts_code, "20260105", df, action)
        assert old_trade_status.get_trade_status_info(
            ts_code, "20260105", df
        ) == new_trade_status.get_trade_status_info(ts_code, "20260105", df)
    # 空行情分支
    empty = df.iloc[0:0]
    assert old_trade_status.is_tradeable("A", "20260105", empty) == new_trade_status.is_tradeable(
        "A", "20260105", empty
    )
    assert old_trade_status.get_trade_status_info(
        "A", "20260105", empty
    ) == new_trade_status.get_trade_status_info("A", "20260105", empty)


# ══════════════════ common/suspend_calendar ══════════════════


class _FakeStorage:
    """Storage 桩：load_raw_by_date 返回合成 suspend 数据（免真实数据依赖）。"""

    def __init__(self, data):
        self._data = data

    def load_raw_by_date(self, dataset: str, trade_date: str):
        assert dataset == "suspend"
        return self._data.get(trade_date)


def _suspend_fixture() -> dict:
    return {
        "20260105": pd.DataFrame({
            "ts_code": ["A", "B", "C"],
            "suspend_type": ["S", "R", "X"],
        }),
        "20260106": pd.DataFrame({"ts_code": [], "suspend_type": []}),
        # 20260107 缺失 → FileNotFoundError 分支
    }


def test_suspend_calendar_equivalence():
    old_cal = old_suspend.SuspendCalendar(_FakeStorage(_suspend_fixture()))
    new_cal = new_suspend.SuspendCalendar(_FakeStorage(_suspend_fixture()))
    for ts_code in ("A", "B", "C", "NO_RECORD"):
        assert old_cal.is_suspended(ts_code, "20260105") == new_cal.is_suspended(
            ts_code, "20260105"
        )
        assert old_cal.get_status_reason(ts_code, "20260105") == new_cal.get_status_reason(
            ts_code, "20260105"
        )
    # 空表分支
    assert old_cal.is_suspended("A", "20260106") == new_cal.is_suspended("A", "20260106") is False
    # 批量接口
    codes = ["A", "B", "C", "NO_RECORD"]
    assert old_cal.batch_is_suspended(codes, "20260105") == new_cal.batch_is_suspended(
        codes, "20260105"
    )
    # 数据缺失严格模式分支
    for cal_cls in (old_suspend.SuspendCalendar, new_suspend.SuspendCalendar):
        cal = cal_cls(_FakeStorage(_suspend_fixture()))
        with pytest.raises(FileNotFoundError):
            cal.is_suspended("A", "20260107")


def test_get_suspend_calendar_passthrough():
    fake = _FakeStorage(_suspend_fixture())
    old_cal, old_storage = old_suspend.get_suspend_calendar(fake)
    new_cal, new_storage = new_suspend.get_suspend_calendar(fake)
    assert old_storage is fake and new_storage is fake
    assert old_cal.is_suspended("A", "20260105") == new_cal.is_suspended("A", "20260105")


# ══════════════════ common/xgboost_compat ══════════════════


def test_xgboost_compat_patterns_and_suppression():
    assert old_xgb_compat.XGBOOST_PICKLE_WARNING_PATTERNS == (
        new_xgb_compat.XGBOOST_PICKLE_WARNING_PATTERNS
    )
    for module in (old_xgb_compat, new_xgb_compat):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            with module.suppress_xgboost_pickle_warning():
                warnings.warn("loading a serialized model from old version")  # 应被抑制
                warnings.warn("No visible GPU is found")  # 应被抑制
                warnings.warn("一条不在白名单的告警")  # 应透传
        assert len(recorded) == 1
        assert "不在白名单" in str(recorded[0].message)


# ══════════════════ common/feature_utils ══════════════════


def test_feature_utils_series_functions_equivalence():
    series = pd.Series([1.0, 2.0, 3.0, 100.0, 200.0, np.nan, 5.0])
    pd.testing.assert_series_equal(
        old_feature_utils.winsorize_series(series, limits=(0.2, 0.2)),
        new_feature_utils.winsorize_series(series, limits=(0.2, 0.2)),
    )
    for base in (None, 10.0):
        pd.testing.assert_series_equal(
            old_feature_utils.log1p_transform(series, base),
            new_feature_utils.log1p_transform(series, base),
        )
    for ddof in (0, 1):
        pd.testing.assert_series_equal(
            old_feature_utils.zscore_transform(series, ddof=ddof),
            new_feature_utils.zscore_transform(series, ddof=ddof),
        )
    # 全 NaN / 空序列 / 零方差分支
    for edge in (pd.Series([np.nan, np.nan]), pd.Series([], dtype=float),
                 pd.Series([3.0, 3.0, 3.0])):
        pd.testing.assert_series_equal(
            old_feature_utils.zscore_transform(edge), new_feature_utils.zscore_transform(edge)
        )
        pd.testing.assert_series_equal(
            old_feature_utils.winsorize_series(edge), new_feature_utils.winsorize_series(edge)
        )


def test_cross_sectional_zscore_equivalence():
    df = pd.DataFrame({
        "trade_date": ["20230101", "20230101", "20230102", "20230102"],
        "ts_code": ["A", "B", "A", "B"],
        "return": [0.05, 0.10, -0.02, 0.03],
    })
    for group_col in (None, "trade_date"):
        for wins in (None, (0.01, 0.01)):
            pd.testing.assert_series_equal(
                old_feature_utils.cross_sectional_zscore(df, "return", group_col, wins),
                new_feature_utils.cross_sectional_zscore(df, "return", group_col, wins),
            )
    with pytest.raises(ValueError) as old_exc:
        old_feature_utils.cross_sectional_zscore(df, "missing")
    with pytest.raises(ValueError) as new_exc:
        new_feature_utils.cross_sectional_zscore(df, "missing")
    assert str(old_exc.value) == str(new_exc.value)


def test_drop_high_correlation_equivalence():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [1.1, 2.1, 2.9, 4.2, 4.8],
        "c": [5.0, 1.0, 4.0, 2.0, 3.0],
    })
    assert old_feature_utils.drop_high_correlation_features(
        df, threshold=0.9
    ) == new_feature_utils.drop_high_correlation_features(df, threshold=0.9)


# ══════════════════ common/trading_config ══════════════════


def test_trading_config_fields_and_defaults():
    _assert_dataclass_fields_equal(old_trading_config.TradingConfig,
                                   new_trading_config.TradingConfig)
    assert asdict(old_trading_config.TradingConfig()) == asdict(
        new_trading_config.TradingConfig()
    )


def test_trading_config_validation_equivalence():
    bad_kwargs = [
        {"top_n": 0},
        {"top_n": 1.5},
        {"rebalance_freq": 0},
        {"stagger_tranches": 0},
        {"top_n": 5, "stagger_tranches": 6},
        {"rebalance_freq": None, "stagger_tranches": 2},
        {"rebalance_freq": 5, "stagger_tranches": 6},
        {"max_weight_per_stock": 0.0},
        {"top_n": 30, "max_weight_per_stock": 0.01},
        {"downside_penalty": 1.0},
        {"downside_penalty": -0.1},
    ]
    for kwargs in bad_kwargs:
        with pytest.raises((ValueError, TypeError)) as old_exc:
            old_trading_config.TradingConfig(**kwargs)
        with pytest.raises((ValueError, TypeError)) as new_exc:
            new_trading_config.TradingConfig(**kwargs)
        assert type(old_exc.value) is type(new_exc.value)
        assert str(old_exc.value) == str(new_exc.value)
    # downside_penalty=None 归一化分支
    assert old_trading_config.TradingConfig(
        downside_penalty=None
    ).downside_penalty == new_trading_config.TradingConfig(downside_penalty=None).downside_penalty


def test_trading_config_from_dict_and_args():
    d = {
        "top_n": 20,
        "rebalance_freq": 10,
        "weight_method": "kelly",  # 旧键别名归一化分支
        "unknown_key": "ignored",
        "downside_penalty": 0.0,
    }
    assert asdict(old_trading_config.TradingConfig.from_dict(d)) == asdict(
        new_trading_config.TradingConfig.from_dict(d)
    )
    args = argparse.Namespace(top_n=15, model_version=3, unrelated="x")
    assert asdict(old_trading_config.TradingConfig.from_args(args)) == asdict(
        new_trading_config.TradingConfig.from_args(args)
    )
    # to_dict 持久化等价
    assert old_trading_config.TradingConfig(top_n=7).to_dict() == new_trading_config.TradingConfig(
        top_n=7
    ).to_dict()


def test_trading_config_create_stop_loss_config():
    enabled = {"stop_loss_enabled": True, "stop_loss_drawdown_pct": 25.0}
    old_cfg = old_trading_config.TradingConfig(**enabled).create_stop_loss_config()
    new_cfg = new_trading_config.TradingConfig(**enabled).create_stop_loss_config()
    assert asdict(old_cfg) == asdict(new_cfg)
    assert old_trading_config.TradingConfig().create_stop_loss_config() is None
    assert new_trading_config.TradingConfig().create_stop_loss_config() is None


def _parse_with(module, argv: List[str], include_price: bool, include_exposure: bool) -> dict:
    parser = argparse.ArgumentParser()
    module.add_trading_args(parser, include_price=include_price, include_exposure=include_exposure)
    return vars(parser.parse_args(argv))


_COMMON_ARGV = [
    "--model-version", "12", "--model-version-b", "34", "--ensemble-weight-a", "0.7",
    "--top-n", "20", "--rebalance-freq", "10", "--stagger-tranches", "2",
    "--max-per-industry", "3", "--max-weight-per-stock", "0.1",
    "--min-list-days", "250", "--position-sizing", "kelly",
    "--kelly-vol-window", "30", "--kelly-max-leverage", "0.2",
    "--downside-penalty", "0.25", "--downside-penalty-column", "cvar_95_20",
    "--no-exclude-st", "--no-early-rebalance-on-empty",
]
_PRICE_ARGV = [
    "--buy-price", "open", "--sell-price", "close", "--initial-capital", "888888",
    "--min-buy-value-ratio", "0.3", "--horizon", "10", "--universe", "all",
]
_EXPOSURE_ARGV = [
    "--exposure-policy", "arm=combined", "--policy-model-root", "some/root",
    "--policy-arm-suffix", "_v6m", "--policy-fold", "OOS13_202506",
    "--policy-warmup-file", "state.json", "--policy-coverage-start", "20250101",
    "--exposure-replenish", "--exposure-trim-tolerance", "0.05",
]


def test_add_trading_args_parsed_namespace_equivalence():
    for include_price in (False, True):
        for include_exposure in (False, True):
            argv = list(_COMMON_ARGV)
            if include_price:
                argv += _PRICE_ARGV
            if include_exposure:
                argv += _EXPOSURE_ARGV
            old_ns = _parse_with(old_trading_config, argv, include_price, include_exposure)
            new_ns = _parse_with(new_trading_config, argv, include_price, include_exposure)
            assert old_ns == new_ns
            # 纯默认值分支
            assert _parse_with(
                old_trading_config, [], include_price, include_exposure
            ) == _parse_with(new_trading_config, [], include_price, include_exposure)
            # help 文本 / 注册顺序锁定（D8 提取式拆分的等价验证）
            old_parser = argparse.ArgumentParser()
            new_parser = argparse.ArgumentParser()
            old_trading_config.add_trading_args(
                old_parser, include_price=include_price, include_exposure=include_exposure
            )
            new_trading_config.add_trading_args(
                new_parser, include_price=include_price, include_exposure=include_exposure
            )
            assert old_parser.format_help() == new_parser.format_help()


# ══════════════════ v2/common/table_schema（F10R-2 拆分） ══════════════════


def test_table_schema_snapshot_constants_equal():
    assert old_sidecar.SNAPSHOT_KEYS == new_table_schema.SNAPSHOT_KEYS
    assert old_sidecar.SNAPSHOT_COLUMNS_ZH == new_table_schema.SNAPSHOT_COLUMNS_ZH


def test_table_schema_policy_keys_not_migrated():
    """F10R-2 边界：政策专用键不迁（随 F8 政策层退役，仍留旧模块）。"""
    for policy_symbol in (
        "LEDGER_COLUMNS_ZH",
        "TRIGGER_COLUMNS_ZH",
        "SCAN_COLUMNS_ZH",
        "GATE_CALIBRATION_KEYS",
        "GATE_CALIBRATION_COLUMNS_ZH",
        "GATE_DAILY_KEYS",
        "GATE_DAILY_COLUMNS_ZH",
        "GATE_EVAL_KEYS",
        "GATE_EVAL_COLUMNS_ZH",
    ):
        assert not hasattr(new_table_schema, policy_symbol), f"政策专用键 {policy_symbol} 不应迁移"


def test_table_schema_to_chinese_equivalence():
    frame = pd.DataFrame({key: range(3) for key in old_sidecar.SNAPSHOT_KEYS})
    old_out = old_sidecar.to_chinese(frame, old_sidecar.SNAPSHOT_COLUMNS_ZH)
    new_out = new_table_schema.to_chinese(frame, new_table_schema.SNAPSHOT_COLUMNS_ZH)
    pd.testing.assert_frame_equal(old_out, new_out)
    assert list(new_out.columns) == list(old_sidecar.SNAPSHOT_COLUMNS_ZH.values())
    # 缺列报错分支（消息一致）
    broken = frame.drop(columns=["shares"])
    with pytest.raises(ValueError) as old_exc:
        old_sidecar.to_chinese(broken, old_sidecar.SNAPSHOT_COLUMNS_ZH)
    with pytest.raises(ValueError) as new_exc:
        new_table_schema.to_chinese(broken, new_table_schema.SNAPSHOT_COLUMNS_ZH)
    assert str(old_exc.value) == str(new_exc.value)
    # select_chinese 列序控制（其实现经 to_chinese 全映射校验，keys 须覆盖映射全键）
    keys = list(old_sidecar.SNAPSHOT_KEYS)
    old_sel = old_sidecar.select_chinese(frame, old_sidecar.SNAPSHOT_COLUMNS_ZH, keys)
    new_sel = new_table_schema.select_chinese(frame, new_table_schema.SNAPSHOT_COLUMNS_ZH, keys)
    pd.testing.assert_frame_equal(old_sel, new_sel)
    # keys 含缺失列 → select_chinese 前置校验报错（消息一致）
    with pytest.raises(ValueError) as old_exc2:
        old_sidecar.select_chinese(broken, old_sidecar.SNAPSHOT_COLUMNS_ZH, ["date", "shares"])
    with pytest.raises(ValueError) as new_exc2:
        new_table_schema.select_chinese(broken, new_table_schema.SNAPSHOT_COLUMNS_ZH,
                                        ["date", "shares"])
    assert str(old_exc2.value) == str(new_exc2.value)

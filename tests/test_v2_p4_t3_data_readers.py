# -*- coding: utf-8 -*-
"""P4-T3 E4：行业、停牌与行情读取的宿主注入及新旧通路等价。"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.common import suspend_calendar as old_suspend
from src.lazybull.portfolio import industry_constraint as old_industry
from src.lazybull.universe.base import Universe
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.suspend_calendar import SuspendCalendar, get_suspend_calendar
from src.lazybull.v2.common.trading_config import TradingConfig
from src.lazybull.v2.core.execution.ml_signal_feed import BacktestEngineML
from src.lazybull.v2.core.signal.base import Signal
from src.lazybull.v2.hosts.backtest.replay import _load_replay_price_data
from src.lazybull.v2.hosts.backtest.runtime import create_backtest_engine_from_config
from src.lazybull.v2.hosts.paper.daily_assembly import _load_price_window

_COST = {
    "commission_rate": 0.0003,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.001,
}


class _Universe(Universe):
    def get_stocks(self, date, quote_data=None):
        return ["A", "B"]


class _Signal(Signal):
    def __init__(self):
        super().__init__("t3_reader")
        self.top_n = 2

    def generate(self, date, universe, data):
        return {stock: 1.0 for stock in universe}


class _Reader:
    def __init__(self, frames=None):
        self.frames = frames or {}
        self.calls = []

    def load_raw_by_date(self, dataset, trade_date):
        self.calls.append((dataset, trade_date))
        return self.frames.get(trade_date)


class _Loader:
    def __init__(self, daily, industry):
        self.daily = daily
        self.industry = industry
        self.daily_calls = []
        self.industry_calls = 0

    def load_clean_daily(self, start_date, end_date):
        self.daily_calls.append((start_date, end_date))
        return self.daily.copy(deep=True)

    def load_shenwan_industry(self):
        self.industry_calls += 1
        return self.industry


def _industry_frame():
    return pd.DataFrame(
        {
            "ts_code": ["A", "B", "C"],
            "sw_l1": ["一级甲", "一级乙", "一级甲"],
            "sw_l2": ["二级甲", "二级乙", "二级丙"],
        }
    )


def _daily_frame():
    return pd.DataFrame(
        {
            "ts_code": ["A", "A", "B", "B"],
            "trade_date": ["20260105", "20260106", "20260105", "20260106"],
            "close": [10.0, 10.1, 20.0, 20.2],
            "close_adj": [10.0, 10.1, 20.0, 20.2],
            "open": [9.9, 10.0, 19.9, 20.1],
            "open_adj": [9.9, 10.0, 19.9, 20.1],
            "is_suspended": [0, 0, 0, 0],
            "is_limit_up": [0, 0, 0, 0],
            "is_limit_down": [0, 0, 0, 0],
            "extra_not_consumed": [1, 2, 3, 4],
        }
    )


def _engine_kwargs(loader):
    return {
        "universe": _Universe("t3"),
        "signal": _Signal(),
        "features_by_date": {},
        "cost_model": CostModel(**_COST),
        "stock_basic": pd.DataFrame({"ts_code": ["A", "B"]}),
        "data_storage": _Reader(),
        "max_per_industry": 1,
        "industry_reader": loader.load_shenwan_industry,
    }


def test_industry_reader_injection_matches_legacy_mapping(monkeypatch):
    monkeypatch.setattr(old_industry, "get_shenwan_level", lambda: "l2")
    monkeypatch.setattr(
        "src.lazybull.v2.core.decision.industry_constraint.get_shenwan_level", lambda: "l2"
    )
    loader = _Loader(_daily_frame(), _industry_frame())
    engine = BacktestEngineML(**_engine_kwargs(loader))
    expected = old_industry.load_industry_mapping(loader.industry)
    assert engine.industry_mapping == expected
    assert loader.industry_calls == 1


def test_backtest_factory_wires_same_industry_reader(monkeypatch):
    monkeypatch.setattr(
        "src.lazybull.v2.core.decision.industry_constraint.get_shenwan_level", lambda: "l2"
    )
    loader = _Loader(_daily_frame(), _industry_frame())
    kwargs = _engine_kwargs(loader)
    engine = create_backtest_engine_from_config(
        trading_config=TradingConfig(model_version=1, top_n=2, max_per_industry=1),
        universe=kwargs["universe"],
        signal=kwargs["signal"],
        features_by_date={},
        stock_basic=kwargs["stock_basic"],
        data_storage=kwargs["data_storage"],
        industry_reader=loader.load_shenwan_industry,
        cost_model=kwargs["cost_model"],
    )
    reference = BacktestEngineML(**kwargs)
    assert engine.industry_mapping == reference.industry_mapping
    assert loader.industry_calls == 2


def test_industry_constraint_without_reader_fails_closed():
    loader = _Loader(_daily_frame(), _industry_frame())
    kwargs = _engine_kwargs(loader)
    kwargs["industry_reader"] = None
    with pytest.raises(ValueError, match="industry_reader"):
        BacktestEngineML(**kwargs)


def test_suspend_calendar_new_old_values_and_cache():
    frames = {
        "20260105": pd.DataFrame({"ts_code": ["A", "B", "C"], "suspend_type": ["S", "R", "X"]}),
        "20260106": pd.DataFrame({"ts_code": [], "suspend_type": []}),
    }
    old = old_suspend.SuspendCalendar(_Reader(frames))
    reader = _Reader(frames)
    new = SuspendCalendar(reader)
    for stock in ("A", "B", "C", "NO_RECORD"):
        assert new.is_suspended(stock, "20260105") == old.is_suspended(stock, "20260105")
        assert new.get_status_reason(stock, "20260105") == old.get_status_reason(stock, "20260105")
    assert new.batch_is_suspended(["A", "B"], "20260105") == old.batch_is_suspended(
        ["A", "B"], "20260105"
    )
    assert new.is_suspended("A", "20260106") is False
    assert reader.calls.count(("suspend", "20260105")) == 1
    with pytest.raises(FileNotFoundError):
        new.is_suspended("A", "20260107")


def test_suspend_calendar_requires_host_reader():
    with pytest.raises(ValueError, match="宿主注入"):
        get_suspend_calendar(None)


def test_market_reads_match_legacy_loader_and_between_hosts():
    loader = _Loader(_daily_frame(), _industry_frame())
    legacy = loader.load_clean_daily("20260105", "20260106")
    consumed_columns = [column for column in legacy.columns if column != "extra_not_consumed"]
    expected = legacy[consumed_columns].copy()
    backtest = _load_replay_price_data(loader, "20260105", "20260106")
    paper = _load_price_window(loader, ["20260105", "20260106"], "20260106", 2)
    pd.testing.assert_frame_equal(backtest, expected, check_exact=True)
    pd.testing.assert_frame_equal(paper, expected, check_exact=True)


def test_core_common_dependency_direction_has_no_old_data_or_store_imports():
    root = Path(__file__).resolve().parents[1] / "src" / "lazybull" / "v2"
    violations = []
    for package in ("core", "common"):
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    if module.startswith("src.lazybull.data") or module.startswith(
                        "src.lazybull.v2.store"
                    ):
                        violations.append(f"{path.relative_to(root)}:{node.lineno}:{module}")
    assert violations == []

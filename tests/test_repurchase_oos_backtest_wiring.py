#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OOS 回测侧股票回购运行时派生接线测试（cs_train 不含本族列）。"""

import pandas as pd
import pytest

from src.lazybull.factors.repurchase import build_repurchase_lookup_by_date
from src.lazybull.ml.walk_forward import backtest as backtest_module
from src.lazybull.ml.walk_forward.backtest import run_oos_backtest

DAYS = ["20240110", "20240111", "20240112"]


class _StubStorage:
    """只提供 load_cs_train_day：返回逐日特征（不含 rp_* 列）。"""

    def __init__(self, days):
        self._days = days
        self.calls = []

    def load_cs_train_day(self, trade_date, subdir="cs_train"):
        self.calls.append(trade_date)
        if trade_date not in self._days:
            return None
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [trade_date, trade_date],
                "close": [1.0, 2.0],
                "circ_mv": [100000.0, 200000.0],  # 万元
                "amount": [10000.0, 20000.0],  # 千元
                "vol": [10000.0, 20000.0],  # 手 ⇒ VWAP 均为 10 元
            }
        )


class _StubLoader:
    def load_clean_daily(self, start, end):
        rows = []
        for day in DAYS:
            rows.append(
                {
                    "ts_code": "000001.SZ",
                    "trade_date": day,
                    "close": 1.0,
                    "close_adj": 1.0,
                    "open": 1.0,
                    "open_adj": 1.0,
                    "is_suspended": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "vol": 1.0,
                    "pct_chg": 0.0,
                    "is_st": False,
                    "list_days": 500,
                    "tradable": True,
                }
            )
        return pd.DataFrame(rows)


class _StubEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.record_holdings_snapshot = False

    def set_exposure_table(self, table, verbose=False):
        self.exposure_table = table

    def run(self, **kwargs):
        return pd.DataFrame({"nav": [1.0, 1.1], "return": [0.0, 0.1]})

    def get_trades(self):
        return pd.DataFrame()

    def get_execution_attribution(self):
        return pd.DataFrame()

    def get_holdings_snapshot(self):
        return pd.DataFrame()


def _trade_cal() -> pd.DataFrame:
    return pd.DataFrame({"cal_date": DAYS, "is_open": [1, 1, 1]})


def _raw_repurchase() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("000001.SZ", "20240110", "实施", 1.0e8, 1.0e6, 12.0),
            ("000001.SZ", "20240110", "实施", 1.4e8, 1.0e6, 12.0),  # 同日累计两行 ⇒ 增量 1.4e8
        ],
        columns=["ts_code", "ann_date", "proc", "amount", "vol", "high_limit"],
    )


def test_oos_backtest_derives_repurchase_columns(monkeypatch):
    lookup = build_repurchase_lookup_by_date(_raw_repurchase(), DAYS)
    captured = {}

    def _fake_engine_factory(**kwargs):
        captured.update(kwargs)
        return _StubEngine(**kwargs)

    monkeypatch.setattr(backtest_module, "create_or_reuse_signal", lambda *a, **k: object())
    monkeypatch.setattr(backtest_module, "create_backtest_engine_from_config", _fake_engine_factory)

    metrics = run_oos_backtest(
        model_version=1,
        bt_start="20240110",
        bt_end="20240112",
        storage=_StubStorage(set(DAYS)),
        loader=_StubLoader(),
        trade_cal=_trade_cal(),
        stock_basic=pd.DataFrame({"ts_code": ["000001.SZ"], "market": ["主板"]}),
        label_column="neu_y_ret_20",
        repurchase_lookup=lookup,
    )

    assert metrics, "回测应返回指标"
    features_by_date = captured["features_by_date"]
    assert set(features_by_date) == set(DAYS)
    for day, frame in features_by_date.items():
        assert "rp_amount_to_mv_90d" in frame.columns
        assert "repurchase_schema_v1" in frame.columns
    # 000001.SZ：增量 1.4e8 元 ÷ 流通市值 100000 万元(=1e9 元)
    assert features_by_date["20240110"]["rp_amount_to_mv_90d"].iloc[0] == pytest.approx(0.14)
    assert features_by_date["20240112"]["rp_amount_to_mv_90d"].iloc[0] == pytest.approx(0.14)
    # VWAP = 10000 千元 × 10 ÷ 10000 手 = 10 元 ⇒ headroom = 12/10 − 1
    assert features_by_date["20240110"]["rp_price_headroom"].iloc[0] == pytest.approx(0.2)
    # 无事件的 000002.SZ 显式填 0（不得 NaN）
    assert features_by_date["20240111"]["rp_amount_to_mv_90d"].iloc[1] == pytest.approx(0.0)


def test_oos_backtest_skips_derivation_without_lookup(monkeypatch):
    captured = {}

    def _fake_engine_factory(**kwargs):
        captured.update(kwargs)
        return _StubEngine(**kwargs)

    monkeypatch.setattr(backtest_module, "create_or_reuse_signal", lambda *a, **k: object())
    monkeypatch.setattr(backtest_module, "create_backtest_engine_from_config", _fake_engine_factory)

    run_oos_backtest(
        model_version=1,
        bt_start="20240110",
        bt_end="20240112",
        storage=_StubStorage(set(DAYS)),
        loader=_StubLoader(),
        trade_cal=_trade_cal(),
        stock_basic=pd.DataFrame({"ts_code": ["000001.SZ"], "market": ["主板"]}),
        label_column="neu_y_ret_20",
    )
    frame = captured["features_by_date"]["20240110"]
    assert not any(col.startswith("rp_") for col in frame.columns)

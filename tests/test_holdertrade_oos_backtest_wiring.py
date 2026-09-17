#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OOS 回测侧股东增减持运行时派生接线测试（cs_train 不含本族列）。"""

import pandas as pd
import pytest

from src.lazybull.ml.walk_forward import backtest as backtest_module
from src.lazybull.ml.walk_forward.backtest import run_oos_backtest


class _StubStorage:
    """只提供 load_cs_train_day：返回逐日特征（不含 ht_* 列）。"""

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
            }
        )


class _StubLoader:
    def load_clean_daily(self, start, end):
        rows = []
        for day in ("20240110", "20240111", "20240112"):
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
    return pd.DataFrame(
        {
            "cal_date": ["20240110", "20240111", "20240112"],
            "is_open": [1, 1, 1],
        }
    )


def test_oos_backtest_derives_holdertrade_columns(monkeypatch):
    from src.lazybull.factors.holdertrade import build_holdertrade_lookup_by_date

    raw = pd.DataFrame(
        [
            ("000001.SZ", "20240110", "G", "IN", 0.5),
            ("000001.SZ", "20240110", "G", "IN", 0.3),
        ],
        columns=["ts_code", "ann_date", "holder_type", "in_de", "change_ratio"],
    )
    days = ["20240110", "20240111", "20240112"]
    lookup = build_holdertrade_lookup_by_date(raw, days)

    captured = {}

    def _fake_engine_factory(**kwargs):
        captured.update(kwargs)
        return _StubEngine(**kwargs)

    monkeypatch.setattr(backtest_module, "create_or_reuse_signal", lambda *a, **k: object())
    monkeypatch.setattr(backtest_module, "create_backtest_engine_from_config", _fake_engine_factory)

    storage = _StubStorage(set(days))
    metrics = run_oos_backtest(
        model_version=1,
        bt_start="20240110",
        bt_end="20240112",
        storage=storage,
        loader=_StubLoader(),
        trade_cal=_trade_cal(),
        stock_basic=pd.DataFrame({"ts_code": ["000001.SZ"], "market": ["主板"]}),
        label_column="neu_y_ret_20",
        holdertrade_lookup=lookup,
    )

    assert metrics, "回测应返回指标"
    features_by_date = captured["features_by_date"]
    assert set(features_by_date) == set(days)
    for day, frame in features_by_date.items():
        assert "ht_net_ratio_30d" in frame.columns
        assert "holdertrade_schema_v1" in frame.columns
    # 1/10 当日有 0.8 增持；1/11、1/12 仍在 30 日窗口内
    assert features_by_date["20240110"]["ht_net_ratio_30d"].iloc[0] == pytest.approx(0.8)
    assert features_by_date["20240112"]["ht_net_ratio_30d"].iloc[0] == pytest.approx(0.8)
    # 无事件的 000002.SZ 显式填 0
    assert features_by_date["20240111"]["ht_net_ratio_30d"].iloc[1] == pytest.approx(0.0)


def test_oos_backtest_skips_derivation_without_lookup(monkeypatch):
    captured = {}

    def _fake_engine_factory(**kwargs):
        captured.update(kwargs)
        return _StubEngine(**kwargs)

    monkeypatch.setattr(backtest_module, "create_or_reuse_signal", lambda *a, **k: object())
    monkeypatch.setattr(backtest_module, "create_backtest_engine_from_config", _fake_engine_factory)

    days = ["20240110", "20240111", "20240112"]
    run_oos_backtest(
        model_version=1,
        bt_start="20240110",
        bt_end="20240112",
        storage=_StubStorage(set(days)),
        loader=_StubLoader(),
        trade_cal=_trade_cal(),
        stock_basic=pd.DataFrame({"ts_code": ["000001.SZ"], "market": ["主板"]}),
        label_column="neu_y_ret_20",
    )
    frame = captured["features_by_date"]["20240110"]
    assert not any(col.startswith("ht_") for col in frame.columns)

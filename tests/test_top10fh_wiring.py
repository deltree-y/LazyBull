#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""十大流通股东运行时派生接线测试（训练入口 + OOS 回测；cs_train 不含本族列）。"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.top10_floatholders import (
    TOP10FH_SCHEMA_VERSION,
    TOP10FH_TOP10_RATIO_COL,
    TOP10FH_VERSION_COL,
    build_top10fh_panel,
)
from src.lazybull.ml.train_core import prepare_training_data
from src.lazybull.ml.walk_forward import backtest as backtest_module
from src.lazybull.ml.walk_forward.backtest import run_oos_backtest

RAW_COLUMNS = [
    "ts_code",
    "end_date",
    "ann_date",
    "holder_name",
    "hold_amount",
    "hold_ratio",
    "hold_float_ratio",
    "hold_change",
    "holder_type",
]

DAYS = ["20240122", "20240123", "20240124"]

#: prepare_training_data 需要的基础特征列（缺失会 KeyError；镜像生产 schema 风格，
#: 与 tests/test_availability_markers.py 同一清单）
_PREPARE_BASE_FEATURE_COLUMNS = [
    "neu_ret_1",
    "neu_ret_20",
    "neu_ret_5",
    "alpha_industry_20",
    "alpha_industry_5",
    "ind_ret_avg",
    "ind_momentum_rank",
    "zscore_ma_deviation_20",
    "zscore_acceleration",
    "zscore_macd_hist",
    "bb_pct",
    "zscore_turnover_rate",
    "vol_ratio_20",
    "vol_burst_20",
    "zscore_amount_ma20",
    "zscore_net_mf_amount",
    "zscore_elg_net_amount_sum_20",
    "lg_net_amount_sum_5",
    "zscore_volatility_20",
    "zscore_volatility_5",
    "amplitude",
    "zscore_bb_width",
    "upper_shadow",
    "lower_shadow",
    "spec_score",
    "rsi_14",
    "kdj_j",
    "zscore_size",
    "zscore_bp",
    "zscore_dv_ttm",
    "zscore_pe_ttm",
    "is_loss",
    "dv_ttm_missing",
    "pe_ttm_missing",
    "list_days",
    "mkt_adv_dec_ratio",
    "mkt_ret_avg_20",
    "mkt_turnover_std",
    "mkt_vol_20",
]


def _raw_top10fh() -> pd.DataFrame:
    """两只股票各两个报告期（第三只永不披露，用于验证 NaN 语义）。

    - 000001.SZ：20230930（ann 20231025）前 10 合计 60 ⇒ 20231231（ann 20240120）66，环比 +6；
    - 000002.SZ：20230930（ann 20231020）合计 50 ⇒ 20231231（ann 20240110）40，环比 −10；
    - 999999.SZ：**无任何报告期** ⇒ 值列必须为 NaN。
    """
    return pd.DataFrame(
        [
            ("000001.SZ", "20230930", "20231025", "甲A", 40.0, 4.0, 40.0, np.nan, "自然人"),
            ("000001.SZ", "20230930", "20231025", "乙B", 20.0, 2.0, 20.0, np.nan, "自然人"),
            (
                "000001.SZ",
                "20231231",
                "20240120",
                "社保A",
                50.0,
                5.0,
                30.0,
                np.nan,
                "社保基金、社保机构",
            ),
            ("000001.SZ", "20231231", "20240120", "保险B", 30.0, 3.0, 18.0, np.nan, "保险投资组合"),
            (
                "000001.SZ",
                "20231231",
                "20240120",
                "券商C",
                20.0,
                2.0,
                12.0,
                np.nan,
                "金融机构—证券公司",
            ),
            ("000001.SZ", "20231231", "20240120", "自然人D", 10.0, 1.0, 6.0, np.nan, "自然人"),
            ("000002.SZ", "20230930", "20231020", "丙C", 30.0, 3.0, 30.0, np.nan, "自然人"),
            ("000002.SZ", "20230930", "20231020", "丁D", 20.0, 2.0, 20.0, np.nan, "自然人"),
            ("000002.SZ", "20231231", "20240110", "戊E", 25.0, 2.5, 25.0, np.nan, "自然人"),
            ("000002.SZ", "20231231", "20240110", "己F", 15.0, 1.5, 15.0, np.nan, "自然人"),
        ],
        columns=RAW_COLUMNS,
    )


def _prepare_frame(n_dates: int = 6, stocks_per_date: int = 3) -> pd.DataFrame:
    codes = ["000001.SZ", "000002.SZ", "999999.SZ"]
    rows = []
    for date_idx in range(n_dates):
        trade_date = f"202401{22 + date_idx:02d}"
        for stock_idx in range(stocks_per_date):
            row = {
                "trade_date": trade_date,
                "ts_code": codes[stock_idx % len(codes)],
                "y_ret_5": 0.01,
            }
            for feat_idx, feat_col in enumerate(_PREPARE_BASE_FEATURE_COLUMNS):
                row[feat_col] = float(date_idx * 0.01 + stock_idx * 0.001 + feat_idx * 1e-6)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 训练入口


def test_top10fh_feature_flag_requires_panel():
    """本族列运行时派生：未提供面板必须明确失败，不得静默降级。"""
    with pytest.raises(ValueError, match="top10fh_panel"):
        prepare_training_data(
            _prepare_frame(),
            label_column="y_ret_5",
            val_ratio=0.4,
            enable_top10fh_features=True,
        )


def test_top10fh_feature_flag_rejects_stale_sentinel_in_partition():
    """特征分区自带本族列（旧语义）时必须失败：运行时派生不覆盖已有列。"""
    from src.lazybull.ml.train_core.constants import TOP10FH_FEATURE_COLUMNS

    df = _prepare_frame()
    for col in TOP10FH_FEATURE_COLUMNS:
        df[col] = 0.0
    df[TOP10FH_VERSION_COL] = TOP10FH_SCHEMA_VERSION - 1

    with pytest.raises(ValueError, match="哨兵列"):
        prepare_training_data(
            df,
            label_column="y_ret_5",
            val_ratio=0.4,
            enable_top10fh_features=True,
            top10fh_panel=build_top10fh_panel(_raw_top10fh()),
        )


def test_top10fh_feature_flag_derives_columns_from_panel():
    """开启开关后本族列进入 feature_columns；未披露股票为 NaN（**不得填 0**）。"""
    result = prepare_training_data(
        _prepare_frame(),
        label_column="y_ret_5",
        val_ratio=0.4,
        enable_top10fh_features=True,
        top10fh_panel=build_top10fh_panel(_raw_top10fh()),
    )
    feature_columns = result[4]
    for col in (
        TOP10FH_TOP10_RATIO_COL,
        "tfh_inst_ratio",
        "tfh_concentration_chg",
        "tfh_freshness_days",
    ):
        assert col in feature_columns

    both = pd.concat([result[5], result[6]])
    assert set(result[0].columns) == set(feature_columns)
    by_code = both.groupby("ts_code")[TOP10FH_TOP10_RATIO_COL].first()
    assert by_code["000001.SZ"] == 66.0
    assert by_code["000002.SZ"] == 40.0
    assert np.isnan(by_code["999999.SZ"])  # 未披露 ⇒ NaN（事件族的 0 填充语义不适用于本族）
    # 集中度环比（对齐上一已存报告期）：66−60=+6 / 40−50=−10
    chg = both.groupby("ts_code")["tfh_concentration_chg"].first()
    assert chg["000001.SZ"] == pytest.approx(6.0)
    assert chg["000002.SZ"] == pytest.approx(-10.0)
    # 长线机构：000001.SZ 社保 30 + 保险 18 = 48；000002.SZ 无
    inst = both.groupby("ts_code")["tfh_inst_ratio"].first()
    assert inst["000001.SZ"] == pytest.approx(48.0)
    assert inst["000002.SZ"] == pytest.approx(0.0)


# ---------------------------------------------------------------- OOS 回测


class _StubStorage:
    def __init__(self, days):
        self._days = days

    def load_cs_train_day(self, trade_date, subdir="cs_train"):
        if trade_date not in self._days:
            return None
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "999999.SZ"],
                "trade_date": [trade_date] * 3,
                "close": [1.0, 2.0, 3.0],
            }
        )


class _StubLoader:
    def load_clean_daily(self, start, end):
        return pd.DataFrame(
            [
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
                for day in DAYS
            ]
        )


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


def _patch_engine(monkeypatch, captured):
    def _fake_engine_factory(**kwargs):
        captured.update(kwargs)
        return _StubEngine(**kwargs)

    monkeypatch.setattr(backtest_module, "create_or_reuse_signal", lambda *a, **k: object())
    monkeypatch.setattr(backtest_module, "create_backtest_engine_from_config", _fake_engine_factory)


def test_oos_backtest_derives_top10fh_columns(monkeypatch):
    captured = {}
    _patch_engine(monkeypatch, captured)

    metrics = run_oos_backtest(
        model_version=1,
        bt_start=DAYS[0],
        bt_end=DAYS[-1],
        storage=_StubStorage(set(DAYS)),
        loader=_StubLoader(),
        trade_cal=_trade_cal(),
        stock_basic=pd.DataFrame({"ts_code": ["000001.SZ"], "market": ["主板"]}),
        label_column="neu_y_ret_20",
        top10fh_panel=build_top10fh_panel(_raw_top10fh()),
    )

    assert metrics, "回测应返回指标"
    features_by_date = captured["features_by_date"]
    assert set(features_by_date) == set(DAYS)
    for _, frame in features_by_date.items():
        assert TOP10FH_TOP10_RATIO_COL in frame.columns
        assert TOP10FH_VERSION_COL in frame.columns
    first = features_by_date[DAYS[0]]
    assert first.loc[first["ts_code"] == "000001.SZ", TOP10FH_TOP10_RATIO_COL].iloc[0] == 66.0
    # 未披露股票必须为 NaN（事件族的 0 填充语义不适用于本族）
    assert first.loc[first["ts_code"] == "999999.SZ", TOP10FH_TOP10_RATIO_COL].isna().all()


def test_oos_backtest_skips_derivation_without_panel(monkeypatch):
    captured = {}
    _patch_engine(monkeypatch, captured)

    run_oos_backtest(
        model_version=1,
        bt_start=DAYS[0],
        bt_end=DAYS[-1],
        storage=_StubStorage(set(DAYS)),
        loader=_StubLoader(),
        trade_cal=_trade_cal(),
        stock_basic=pd.DataFrame({"ts_code": ["000001.SZ"], "market": ["主板"]}),
        label_column="neu_y_ret_20",
    )
    frame = captured["features_by_date"][DAYS[0]]
    assert not any(col.startswith("tfh_") for col in frame.columns)

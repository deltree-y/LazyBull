#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MA250 可观测性相关测试。"""

import tempfile

import pandas as pd

from scripts.compare_walk_forward import build_comparison_table
from scripts.walk_forward import summarize_ma250_signal_coverage
from src.lazybull.backtest.engine_ml import _format_ma250_decision_log
from src.lazybull.data import Storage
from src.lazybull.features.ensure import _REQUIRED_FACTOR_COLS, _check_features_schema


def _make_features(ratio: float) -> pd.DataFrame:
    return pd.DataFrame({"ts_code": ["000001.SZ"], "mkt_ma250_ratio": [ratio]})


class TestMA250SignalCoverage:
    """测试 MA250 阈值命中统计。"""

    def test_distinguishes_trade_day_hits_and_signal_day_hits(self):
        trade_dates = [
            "20240101",
            "20240102",
            "20240103",
            "20240104",
            "20240105",
            "20240106",
            "20240107",
        ]
        features_by_date = {
            "20240101": _make_features(1.02),
            "20240102": _make_features(0.79),
            "20240103": _make_features(0.95),
            "20240104": _make_features(0.78),
            "20240105": _make_features(1.01),
            "20240106": _make_features(0.92),
            "20240107": _make_features(1.03),
        }

        stats = summarize_ma250_signal_coverage(
            trade_dates=trade_dates,
            features_by_date=features_by_date,
            threshold=0.8,
            rebalance_freq=3,
        )

        assert stats["trade_days"] == 7
        assert stats["signal_days"] == 3
        assert stats["hit_trade_days"] == 2
        assert stats["hit_signal_days"] == 1
        assert stats["first_hit_trade_date"] == "20240102"
        assert stats["first_hit_signal_date"] == "20240104"
        assert stats["first_hit_signal_ratio"] == 0.78


class TestCompareWalkForwardMA250Columns:
    """测试 compare 脚本保留 MA250 关键参数列。"""

    def test_build_comparison_table_keeps_ma250_columns(self):
        all_df = pd.DataFrame(
            [
                {
                    "wf_run_id": "wf_test_001",
                    "split_index": 0,
                    "market_regime_ma250_hard_stop": True,
                    "market_regime_ma250_threshold": 0.8,
                    "market_regime_ma250_exposure": 0.2,
                    "market_regime_ma250_atr_scaling": True,
                }
            ]
        )

        result = build_comparison_table(all_df)

        assert "MA250硬条件" in result.columns
        assert "MA250阈值" in result.columns
        assert "MA250仓位" in result.columns
        assert "MA250 ATR缩放" in result.columns
        assert result.loc[0, "MA250硬条件"] == True
        assert result.loc[0, "MA250阈值"] == 0.8
        assert result.loc[0, "MA250仓位"] == 0.2
        assert result.loc[0, "MA250 ATR缩放"] == True


class TestMA250LogFormatting:
    """测试 MA250 日志文案是否足够清晰。"""

    def test_format_log_with_trigger_and_atr_scaling(self):
        message = _format_ma250_decision_log(
            date=pd.Timestamp("2024-08-12"),
            ma250_ratio=0.874,
            threshold=0.95,
            hard_stop_exposure=0.2,
            base_exposure=0.2,
            final_exposure=0.2,
            ma250_triggered=True,
            atr_scaling_enabled=True,
            atr_ratio=0.996,
            mkt_atr=0.0214,
            mkt_atr_ma250=0.0213,
        )

        assert "ratio=0.874 < threshold=0.950" in message
        assert "触发硬条件" in message
        assert "hard_stop_exposure=20.0%" in message
        assert "base_after_ma250=20.0%" in message
        assert "final_after_atr=20.0%" in message
        assert "ATR缩放=开启(scale=0.996" in message
        assert "atr_ma250=2.13%" in message
        assert "atr_now=2.14%" in message

    def test_format_log_with_missing_atr_data(self):
        message = _format_ma250_decision_log(
            date=pd.Timestamp("2024-08-12"),
            ma250_ratio=1.012,
            threshold=0.95,
            hard_stop_exposure=0.2,
            base_exposure=1.0,
            final_exposure=1.0,
            ma250_triggered=False,
            atr_scaling_enabled=True,
        )

        assert "ratio=1.012 >= threshold=0.950" in message
        assert "未触发硬条件" in message
        assert "base_after_ma250=100.0%" in message
        assert "缺少有效ATR数据" in message


class TestMA250FeatureCacheSchema:
    """测试市场级 ATR 特征被纳入缓存完整性校验。"""

    def test_schema_check_rejects_cache_missing_market_atr_cols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(root_path=tmpdir, verbose=False)
            cols = [
                col
                for col in _REQUIRED_FACTOR_COLS
                if col not in {"mkt_atr_pct", "mkt_atr_pct_ma250"}
            ]
            df = pd.DataFrame({col: [0.1] for col in cols})
            storage.save_cs_train_day(df, "20240812")

            assert _check_features_schema(storage, "20240812") is False

    def test_schema_check_accepts_cache_with_market_atr_cols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(root_path=tmpdir, verbose=False)
            df = pd.DataFrame({col: [0.1] for col in _REQUIRED_FACTOR_COLS})
            storage.save_cs_train_day(df, "20240813")

            assert _check_features_schema(storage, "20240813") is True
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MA250 可观测性相关测试。"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_walk_forward import (
    build_auto_compare_jobs,
    build_comparison_table,
    build_period_stability_table,
    compute_composite_score,
    load_all_summaries_from_raw_dirs,
    run_auto_compare_jobs,
)
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
                        "batch_run_id": "wf_batch_001",
                        "batch_period_label": "0101",
                    "split_index": 0,
                    "market_regime_ma250_hard_stop": True,
                    "market_regime_ma250_threshold": 0.8,
                    "market_regime_ma250_exposure": 0.2,
                    "market_regime_ma250_atr_scaling": True,
                    "signal_confidence_gate_enabled": True,
                    "signal_confidence_gate_top_k": 8,
                    "signal_confidence_gate_thresholds": "[0.1, 0.3]",
                    "signal_confidence_gate_exposure_levels": "[0.4, 1.0]",
                    "bt_total_return": 0.12,
                    "bt_signal_confidence_block_rate": 0.25,
                    "bt_signal_confidence_avg_exposure": 0.7,
                    "bt_signal_confidence_avg_score": 0.18,
                    "bt_sell_timing": "close",
                    "bt_min_list_days": 180,
                    "bt_max_weight_per_stock": 0.15,
                    "bt_stop_loss_enabled": True,
                    "bt_equity_curve_enabled": True,
                    "bt_equity_curve_recovery_mode": "immediate",
                }
            ]
        )

        result = build_comparison_table(all_df)

        assert "批次ID" in result.columns
        assert "批次时间段" in result.columns
        assert "MA250硬条件" in result.columns
        assert "MA250阈值" in result.columns
        assert "MA250仓位" in result.columns
        assert "MA250 ATR缩放" in result.columns
        assert "回测卖出时机" in result.columns
        assert "回测最少上市天数" in result.columns
        assert "回测单股最大权重" in result.columns
        assert "回测止损" in result.columns
        assert "回测ECT" in result.columns
        assert "回测ECT恢复模式" in result.columns
        assert "信号置信度门控" in result.columns
        assert "门控TopK" in result.columns
        assert "门控阈值" in result.columns
        assert "门控仓位系数" in result.columns
        assert "门控持币率均值" in result.columns
        assert "门控平均仓位" in result.columns
        assert "门控平均置信度" in result.columns
        assert result.loc[0, "MA250硬条件"] == True
        assert result.loc[0, "批次ID"] == "wf_batch_001"
        assert result.loc[0, "批次时间段"] == "0101"
        assert result.loc[0, "MA250阈值"] == 0.8
        assert result.loc[0, "MA250仓位"] == 0.2
        assert result.loc[0, "MA250 ATR缩放"] == True
        assert result.loc[0, "信号置信度门控"] == True
        assert result.loc[0, "门控TopK"] == 8
        assert result.loc[0, "门控阈值"] == "[0.1, 0.3]"
        assert result.loc[0, "门控仓位系数"] == "[0.4, 1.0]"
        assert result.loc[0, "门控持币率均值"] == 0.25
        assert result.loc[0, "门控平均仓位"] == 0.7
        assert result.loc[0, "门控平均置信度"] == 0.18
        assert result.loc[0, "回测卖出时机"] == "close"
        assert result.loc[0, "回测最少上市天数"] == 180
        assert result.loc[0, "回测单股最大权重"] == 0.15
        assert result.loc[0, "回测止损"] == True
        assert result.loc[0, "回测ECT"] == True
        assert result.loc[0, "回测ECT恢复模式"] == "immediate"


class TestCompareWalkForwardChainMetrics:
    """测试 compare 脚本输出全周期 chain 指标。"""

    def test_build_comparison_table_adds_chain_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            nav_up = np.linspace(1.0, 1.2, 126)
            nav_down = np.linspace(1.2, 1.08, 21)[1:]
            nav_recover = np.linspace(1.08, 1.2, 107)[1:]
            nav = np.concatenate([nav_up, nav_down, nav_recover])
            pd.DataFrame(
                {
                    "date": list(range(len(nav))),
                    "nav": nav,
                    "split_index": [0] * len(nav),
                }
            ).to_csv(raw_dir / "chain_nav_wf_test_001.csv", index=False, encoding="utf-8-sig")

            all_df = pd.DataFrame(
                [
                    {
                        "wf_run_id": "wf_test_001",
                        "split_index": 0,
                        "bt_total_return": 0.12,
                        "bt_annual_return": 0.10,
                        "bt_max_drawdown": -0.20,
                        "bt_sharpe": 1.10,
                    },
                    {
                        "wf_run_id": "wf_test_001",
                        "split_index": 1,
                        "bt_total_return": 0.08,
                        "bt_annual_return": 0.07,
                        "bt_max_drawdown": -0.12,
                        "bt_sharpe": 0.90,
                    },
                ]
            )

            result = build_comparison_table(all_df, raw_dir=raw_dir)

            assert "全周期CAGR" in result.columns
            assert "全周期总收益" in result.columns
            assert "全周期链式最大回撤" in result.columns
            assert "全周期链式夏普" in result.columns
            assert "全周期链式交易日数" in result.columns
            assert abs(result.loc[0, "全周期总收益"] - 0.2) < 1e-6
            assert abs(result.loc[0, "全周期CAGR"] - 0.2) < 1e-6
            assert abs(result.loc[0, "全周期链式最大回撤"] + 0.1) < 1e-6
            assert result.loc[0, "全周期链式交易日数"] == 252
            assert pd.notna(result.loc[0, "全周期链式夏普"])


class TestCompareWalkForwardSummarySanitize:
    """测试 compare 读取历史 summary 时会清洗失效参数。"""

    def test_load_all_summaries_from_raw_dirs_sanitizes_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "wf_run_id": "wf_old_1",
                        "split_index": 0,
                        "oos_backtest": True,
                        "signal_gate_mode": "disabled",
                        "signal_confidence_gate_enabled": True,
                        "signal_confidence_gate_top_k": 8,
                        "signal_gate_dynamic_topn": True,
                        "signal_gate_topn_high_multiplier": 0.5,
                        "signal_gate_topn_low_multiplier": 1.6,
                        "enable_profit_based_holding": False,
                        "profit_extension_mode": "pnl",
                        "profit_extension_days": 5,
                        "market_regime": True,
                        "market_regime_mode": "binary",
                        "market_regime_drawdown_guard": True,
                        "market_regime_drawdown_threshold": -0.09,
                    },
                    {
                        "wf_run_id": "wf_old_2",
                        "split_index": 0,
                        "oos_backtest": True,
                        "signal_gate_mode": "composite",
                        "signal_confidence_gate_enabled": False,
                        "signal_confidence_gate_top_k": 12,
                        "signal_confidence_gate_thresholds": "[0.1, 0.3]",
                        "signal_confidence_gate_exposure_levels": "[0.4, 1.0]",
                        "signal_gate_cost_multiplier": 1.8,
                        "signal_gate_round_trip_cost": 0.004,
                        "signal_gate_percentile_warmup": 9,
                        "market_regime": True,
                        "market_regime_mode": "combined",
                        "market_regime_drawdown_guard": False,
                        "market_regime_drawdown_threshold": -0.08,
                    },
                ]
            ).to_csv(raw_dir / "walk_forward_summary_legacy.csv", index=False, encoding="utf-8-sig")

            loaded = load_all_summaries_from_raw_dirs([raw_dir])

            row1 = loaded.loc[loaded["wf_run_id"] == "wf_old_1"].iloc[0]
            row2 = loaded.loc[loaded["wf_run_id"] == "wf_old_2"].iloc[0]

            assert pd.isna(row1["signal_confidence_gate_top_k"])
            assert pd.isna(row1["signal_gate_dynamic_topn"])
            assert pd.isna(row1["signal_gate_topn_high_multiplier"])
            assert pd.isna(row1["signal_gate_topn_low_multiplier"])
            assert pd.isna(row1["profit_extension_mode"])
            assert pd.isna(row1["profit_extension_days"])
            assert str(row1["market_regime_drawdown_guard"]).lower() == "true"
            assert row1["market_regime_drawdown_threshold"] == -0.09

            assert row2["signal_confidence_gate_top_k"] == 12
            assert pd.isna(row2["signal_confidence_gate_enabled"])
            assert pd.isna(row2["signal_confidence_gate_thresholds"])
            assert pd.isna(row2["signal_confidence_gate_exposure_levels"])
            assert pd.isna(row2["market_regime_drawdown_threshold"])

    def test_load_all_summaries_from_raw_dirs_avoids_concat_futurewarning(self):
        """多个 summary 在局部列全 NA 时，拼接不应再触发 pandas FutureWarning。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "wf_run_id": "wf_a",
                        "split_index": 0,
                        "oos_backtest": True,
                        "enable_profit_based_holding": False,
                        "profit_extension_mode": "pnl",
                        "profit_extension_days": 5,
                        "market_regime": True,
                        "market_regime_mode": "combined",
                        "market_regime_drawdown_guard": False,
                        "market_regime_drawdown_threshold": -0.08,
                    }
                ]
            ).to_csv(raw_dir / "walk_forward_summary_a.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [
                    {
                        "wf_run_id": "wf_b",
                        "split_index": 0,
                        "oos_backtest": True,
                        "enable_profit_based_holding": True,
                        "profit_extension_mode": "strength",
                        "profit_extension_strength_threshold": 0.75,
                        "market_regime": True,
                        "market_regime_mode": "combined",
                        "market_regime_drawdown_guard": True,
                        "market_regime_drawdown_threshold": -0.12,
                    }
                ]
            ).to_csv(raw_dir / "walk_forward_summary_b.csv", index=False, encoding="utf-8-sig")

            warnings.simplefilter("always", FutureWarning)
            with warnings.catch_warnings(record=True) as caught:
                loaded = load_all_summaries_from_raw_dirs([raw_dir])

            future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
            assert not future_warnings
            assert len(loaded) == 2
            row_a = loaded.loc[loaded["wf_run_id"] == "wf_a"].iloc[0]
            row_b = loaded.loc[loaded["wf_run_id"] == "wf_b"].iloc[0]
            assert pd.isna(row_a["profit_extension_mode"])
            assert row_b["profit_extension_mode"] == "strength"
            assert pd.isna(row_a["market_regime_drawdown_threshold"])
            assert row_b["market_regime_drawdown_threshold"] == -0.12

    def test_load_all_summaries_from_raw_dirs_infers_bt_rebalance_freq_from_split_boundaries(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "wf_run_id": "wf_hist_001",
                        "split_index": 0,
                        "split_count": 14,
                        "final_date": "20260324",
                        "step": "semiannual",
                        "train_window_years": 6,
                        "test_window_months": 6,
                        "label_column": "neu_y_ret_20",
                        "oos_backtest": True,
                        "train_start": "20130219",
                        "train_end": "20190219",
                        "test_start": "20190220",
                        "test_end": "20190821",
                    },
                    {
                        "wf_run_id": "wf_hist_001",
                        "split_index": 13,
                        "split_count": 14,
                        "final_date": "20260324",
                        "step": "semiannual",
                        "train_window_years": 6,
                        "test_window_months": 6,
                        "label_column": "neu_y_ret_20",
                        "oos_backtest": True,
                        "train_start": "20190923",
                        "train_end": "20250922",
                        "test_start": "20250923",
                        "test_end": "20260324",
                    },
                ]
            ).to_csv(raw_dir / "walk_forward_summary_hist.csv", index=False, encoding="utf-8-sig")

            loaded = load_all_summaries_from_raw_dirs([raw_dir], data_root=Path("./data"))

            assert "bt_rebalance_freq" in loaded.columns
            assert (loaded["bt_rebalance_freq"] == 3).all()


class TestCompareWalkForwardPeriodStability:
    """测试 compare 脚本的跨时间段稳定性汇总。"""

    def test_build_period_stability_table_groups_same_params_across_periods(self):
        all_df = pd.DataFrame(
            [
                {
                    "wf_run_id": "wf_test_001",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0101",
                    "wf_start_date": "20130101",
                    "wf_end_date": "20201231",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_total_return": 0.12,
                    "daily_rankic_ir": 0.60,
                },
                {
                    "wf_run_id": "wf_test_001",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0101",
                    "wf_start_date": "20130101",
                    "wf_end_date": "20201231",
                    "split_index": 1,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_total_return": 0.10,
                    "daily_rankic_ir": 0.50,
                },
                {
                    "wf_run_id": "wf_test_002",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0209",
                    "wf_start_date": "20130224",
                    "wf_end_date": "20210224",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_total_return": 0.08,
                    "daily_rankic_ir": 0.45,
                },
                {
                    "wf_run_id": "wf_test_002",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0209",
                    "wf_start_date": "20130224",
                    "wf_end_date": "20210224",
                    "split_index": 1,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_total_return": 0.06,
                    "daily_rankic_ir": 0.35,
                },
                {
                    "wf_run_id": "wf_test_003",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0301",
                    "wf_start_date": "20140101",
                    "wf_end_date": "20211231",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 5,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_total_return": 0.10,
                    "daily_rankic_ir": 0.20,
                },
            ]
        )

        comp_df = build_comparison_table(all_df)
        comp_df.insert(1, "综合得分", compute_composite_score(comp_df))

        result = build_period_stability_table(comp_df)

        assert len(result) == 1
        assert result.loc[0, "时间段数"] == 2
        assert result.loc[0, "时间段列表"] == "0101 | 0209"
        assert result.loc[0, "运行ID列表"] == "0101:wf_test_001 | 0209:wf_test_002"
        assert result.loc[0, "最大深度"] == 3
        assert result.loc[0, "批次ID"] == "wf_batch_001"
        assert pd.notna(result.loc[0, "综合得分均值"])
        assert pd.notna(result.loc[0, "跨时间段跨切分IR标准差"])
        assert result.loc[0, "时间段稳定性分"] <= 100

    def test_build_period_stability_table_keeps_batches_separate(self):
        all_df = pd.DataFrame(
            [
                {
                    "wf_run_id": "wf_test_001",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0101",
                    "wf_start_date": "20130101",
                    "wf_end_date": "20201231",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_total_return": 0.12,
                    "daily_rankic_ir": 0.60,
                },
                {
                    "wf_run_id": "wf_test_002",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0209",
                    "wf_start_date": "20130224",
                    "wf_end_date": "20210224",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_total_return": 0.08,
                    "daily_rankic_ir": 0.45,
                },
                {
                    "wf_run_id": "wf_test_003",
                    "batch_run_id": "wf_batch_002",
                    "batch_period_label": "0101",
                    "wf_start_date": "20130101",
                    "wf_end_date": "20201231",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_total_return": 0.11,
                    "daily_rankic_ir": 0.58,
                },
                {
                    "wf_run_id": "wf_test_004",
                    "batch_run_id": "wf_batch_002",
                    "batch_period_label": "0209",
                    "wf_start_date": "20130224",
                    "wf_end_date": "20210224",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_total_return": 0.07,
                    "daily_rankic_ir": 0.40,
                },
            ]
        )

        comp_df = build_comparison_table(all_df)
        comp_df.insert(1, "综合得分", compute_composite_score(comp_df))

        result = build_period_stability_table(comp_df)

        assert len(result) == 2
        assert sorted(result["批次ID"].tolist()) == ["wf_batch_001", "wf_batch_002"]
        assert (result["时间段数"] == 2).all()
        expected_run_id_lists = {
            "wf_batch_001": "0101:wf_test_001 | 0209:wf_test_002",
            "wf_batch_002": "0101:wf_test_003 | 0209:wf_test_004",
        }
        for _, row in result.iterrows():
            assert row["运行ID列表"] == expected_run_id_lists[row["批次ID"]]

    def test_build_period_stability_table_separates_bt_top_n(self):
        all_df = pd.DataFrame(
            [
                {
                    "wf_run_id": "wf_test_001",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0101",
                    "wf_start_date": "20130101",
                    "wf_end_date": "20201231",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_top_n": 20,
                    "position_sizing": "equal",
                    "bt_total_return": 0.12,
                    "daily_rankic_ir": 0.60,
                },
                {
                    "wf_run_id": "wf_test_002",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0209",
                    "wf_start_date": "20130224",
                    "wf_end_date": "20210224",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_top_n": 20,
                    "position_sizing": "equal",
                    "bt_total_return": 0.08,
                    "daily_rankic_ir": 0.45,
                },
                {
                    "wf_run_id": "wf_test_003",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0101",
                    "wf_start_date": "20130101",
                    "wf_end_date": "20201231",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_top_n": 30,
                    "position_sizing": "equal",
                    "bt_total_return": 0.10,
                    "daily_rankic_ir": 0.58,
                },
                {
                    "wf_run_id": "wf_test_004",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0209",
                    "wf_start_date": "20130224",
                    "wf_end_date": "20210224",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_top_n": 30,
                    "position_sizing": "equal",
                    "bt_total_return": 0.07,
                    "daily_rankic_ir": 0.40,
                },
            ]
        )

        comp_df = build_comparison_table(all_df)
        comp_df.insert(1, "综合得分", compute_composite_score(comp_df))

        result = build_period_stability_table(comp_df)

        assert len(result) == 2
        assert sorted(result["回测TopN"].tolist()) == [20, 30]
        assert (result["时间段数"] == 2).all()
        expected_run_id_lists = {
            20: "0101:wf_test_001 | 0209:wf_test_002",
            30: "0101:wf_test_003 | 0209:wf_test_004",
        }
        for _, row in result.iterrows():
            assert row["运行ID列表"] == expected_run_id_lists[row["回测TopN"]]

    def test_build_period_stability_table_dedup_same_period_keep_latest_run(self):
        all_df = pd.DataFrame(
            [
                {
                    "wf_run_id": "wf_20260430_114905_old001",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0101",
                    "wf_start_date": "20130101",
                    "wf_end_date": "20251231",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_total_return": 0.10,
                    "daily_rankic_ir": 0.40,
                },
                {
                    "wf_run_id": "wf_20260430_115053_new002",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0101",
                    "wf_start_date": "20130101",
                    "wf_end_date": "20251231",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_total_return": 0.11,
                    "daily_rankic_ir": 0.42,
                },
                {
                    "wf_run_id": "wf_20260430_114958_run003",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0209",
                    "wf_start_date": "20130209",
                    "wf_end_date": "20260209",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_total_return": 0.08,
                    "daily_rankic_ir": 0.35,
                },
            ]
        )

        comp_df = build_comparison_table(all_df)
        comp_df.insert(1, "综合得分", compute_composite_score(comp_df))

        result = build_period_stability_table(comp_df)

        assert len(result) == 1
        assert result.loc[0, "时间段数"] == 2
        assert result.loc[0, "时间段列表"] == "0101 | 0209"
        assert result.loc[0, "运行ID列表"] == (
            "0101:wf_20260430_115053_new002 | 0209:wf_20260430_114958_run003"
        )

    def test_build_period_stability_table_ignores_split_count_and_final_date(self):
        all_df = pd.DataFrame(
            [
                {
                    "wf_run_id": "wf_20260508_120000_run001",
                    "batch_run_id": "wf_batch_20260508_001",
                    "batch_period_label": "0101",
                    "split_count": "14",
                    "final_date": "20251231",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_top_n": 20,
                    "bt_total_return": 0.12,
                    "daily_rankic_ir": 0.50,
                },
                {
                    "wf_run_id": "wf_20260508_121500_run002",
                    "batch_run_id": "wf_batch_20260508_001",
                    "batch_period_label": "0209",
                    "split_count": "13",
                    "final_date": "20260209",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_top_n": 20,
                    "bt_total_return": 0.10,
                    "daily_rankic_ir": 0.45,
                },
            ]
        )

        comp_df = build_comparison_table(all_df)
        comp_df.insert(1, "综合得分", compute_composite_score(comp_df))

        result = build_period_stability_table(comp_df)

        assert len(result) == 1
        assert result.loc[0, "时间段数"] == 2
        assert result.loc[0, "时间段列表"] == "0101 | 0209"
        assert result.loc[0, "运行ID列表"] == (
            "0101:wf_20260508_120000_run001 | 0209:wf_20260508_121500_run002"
        )

    def test_build_period_stability_table_separates_bt_rebalance_freq(self):
        all_df = pd.DataFrame(
            [
                {
                    "wf_run_id": "wf_test_101",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0101",
                    "wf_start_date": "20130101",
                    "wf_end_date": "20201231",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_top_n": 20,
                    "bt_rebalance_freq": 3,
                    "position_sizing": "equal",
                    "bt_total_return": 0.12,
                    "daily_rankic_ir": 0.60,
                },
                {
                    "wf_run_id": "wf_test_102",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0209",
                    "wf_start_date": "20130224",
                    "wf_end_date": "20210224",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_top_n": 20,
                    "bt_rebalance_freq": 3,
                    "position_sizing": "equal",
                    "bt_total_return": 0.08,
                    "daily_rankic_ir": 0.45,
                },
                {
                    "wf_run_id": "wf_test_103",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0101",
                    "wf_start_date": "20130101",
                    "wf_end_date": "20201231",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_top_n": 20,
                    "bt_rebalance_freq": 5,
                    "position_sizing": "equal",
                    "bt_total_return": 0.10,
                    "daily_rankic_ir": 0.58,
                },
                {
                    "wf_run_id": "wf_test_104",
                    "batch_run_id": "wf_batch_001",
                    "batch_period_label": "0209",
                    "wf_start_date": "20130224",
                    "wf_end_date": "20210224",
                    "split_index": 0,
                    "algorithm": "xgboost",
                    "max_depth": 3,
                    "learning_rate": 0.01,
                    "train_window_years": 6,
                    "test_window_months": 6,
                    "label_column": "neu_y_ret_20",
                    "task": "regression",
                    "label_transform": "cs_zscore",
                    "bt_top_n": 20,
                    "bt_rebalance_freq": 5,
                    "position_sizing": "equal",
                    "bt_total_return": 0.07,
                    "daily_rankic_ir": 0.40,
                },
            ]
        )

        comp_df = build_comparison_table(all_df)
        comp_df.insert(1, "综合得分", compute_composite_score(comp_df))

        result = build_period_stability_table(comp_df)

        assert len(result) == 2
        assert sorted(result["回测调仓频率"].tolist()) == [3, 5]
        assert (result["时间段数"] == 2).all()


class TestCompareWalkForwardAutoDiscovery:
    """测试 compare 脚本无参时自动扫描 raw 与 batches。"""

    def test_run_auto_compare_jobs_generates_raw_and_batches_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            raw_dir = data_root / "walk_forward" / "raw"
            batch_raw_dir = data_root / "walk_forward" / "batches" / "wf_batch_001" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            batch_raw_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "wf_run_id": "wf_raw_001",
                        "split_index": 0,
                        "bt_total_return": 0.05,
                        "bt_annual_return": 0.08,
                        "bt_max_drawdown": -0.10,
                        "bt_sharpe": 1.0,
                    }
                ]
            ).to_csv(raw_dir / "walk_forward_summary_raw_0001.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                {
                    "date": [1, 2, 3],
                    "nav": [1.0, 1.02, 1.05],
                }
            ).to_csv(raw_dir / "chain_nav_wf_raw_001.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [
                    {
                        "wf_run_id": "wf_batch_001",
                        "batch_run_id": "wf_batch_001",
                        "batch_period_label": "0101",
                        "split_index": 0,
                        "bt_total_return": 0.06,
                        "bt_annual_return": 0.09,
                        "bt_max_drawdown": -0.09,
                        "bt_sharpe": 1.1,
                    }
                ]
            ).to_csv(
                batch_raw_dir / "walk_forward_summary_0101_0001.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(
                {
                    "date": [1, 2, 3],
                    "nav": [1.0, 1.03, 1.06],
                }
            ).to_csv(batch_raw_dir / "chain_nav_wf_batch_001.csv", index=False, encoding="utf-8-sig")

            jobs = build_auto_compare_jobs(data_root)

            assert len(jobs) == 2
            assert jobs[0]["output_path"].name == "wf_comparison_raw.xlsx"
            assert jobs[1]["output_path"].name == "wf_comparison_batches.xlsx"

            output_paths = run_auto_compare_jobs(data_root)
            raw_output = data_root / "walk_forward" / "wf_comparison_raw.xlsx"
            batch_output = data_root / "walk_forward" / "wf_comparison_batches.xlsx"

            assert raw_output in output_paths
            assert batch_output in output_paths
            assert raw_output.exists()
            assert batch_output.exists()

            raw_report = pd.read_excel(raw_output, sheet_name="实验对比")
            batch_report = pd.read_excel(batch_output, sheet_name="实验对比")

            assert len(raw_report) == 1
            assert len(batch_report) == 1
            assert raw_report.loc[0, "运行ID"] == "wf_raw_001"
            assert batch_report.loc[0, "运行ID"] == "wf_batch_001"

    def test_run_auto_compare_jobs_writes_placeholder_when_raw_is_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            raw_dir = data_root / "walk_forward" / "raw"
            batch_raw_dir = data_root / "walk_forward" / "batches" / "wf_batch_001" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            batch_raw_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "wf_run_id": "wf_batch_001",
                        "batch_run_id": "wf_batch_001",
                        "batch_period_label": "0101",
                        "split_index": 0,
                        "bt_total_return": 0.06,
                        "bt_annual_return": 0.09,
                        "bt_max_drawdown": -0.09,
                        "bt_sharpe": 1.1,
                    }
                ]
            ).to_csv(
                batch_raw_dir / "walk_forward_summary_0101_0001.csv",
                index=False,
                encoding="utf-8-sig",
            )

            output_paths = run_auto_compare_jobs(data_root)
            raw_output = data_root / "walk_forward" / "wf_comparison_raw.xlsx"
            batch_output = data_root / "walk_forward" / "wf_comparison_batches.xlsx"

            assert raw_output in output_paths
            assert batch_output in output_paths
            assert raw_output.exists()
            assert batch_output.exists()

            raw_report = pd.read_excel(raw_output, sheet_name="实验对比")
            assert raw_report.loc[0, "状态"] == "无可用数据"
            assert raw_report.loc[0, "来源"] == "raw"


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

        assert "MA250: 2024-08-12" in message
        assert "ratio=0.874:触发控仓" in message
        assert "ATR缩放=开启(scale=atr_ma250/atr_now=2.13%/2.14%=99.6%)" in message
        assert "base_after_ma250=20.0%" in message
        assert "final_after_atr=20.0%." in message

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

        assert "ratio=1.012:未触发控仓" in message
        assert "ATR缩放=开启(缺少有效ATR数据)" in message
        assert "base_after_ma250=100.0%" in message
        assert "final_after_atr=100.0%." in message


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


class TestAnnouncementFreshnessCacheSchema:
    """测试 freshness 特征被纳入缓存完整性校验。"""

    def test_schema_check_rejects_cache_missing_freshness_cols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(root_path=tmpdir, verbose=False)
            cols = [
                col
                for col in _REQUIRED_FACTOR_COLS
                if col not in {
                    "fundamental_freshness_days",
                    "holder_freshness_days",
                    "forecast_freshness_days",
                    "fund_portfolio_freshness_days",
                    "express_freshness_days",
                }
            ]
            df = pd.DataFrame({col: [0.1] for col in cols})
            storage.save_cs_train_day(df, "20240814")

            assert _check_features_schema(storage, "20240814") is False

    def test_schema_check_accepts_cache_with_freshness_cols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(root_path=tmpdir, verbose=False)
            df = pd.DataFrame({col: [0.1] for col in _REQUIRED_FACTOR_COLS})
            storage.save_cs_train_day(df, "20240815")

            assert _check_features_schema(storage, "20240815") is True
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Walk-forward 功能单元测试

测试内容：
- split 生成逻辑（边界、数量、日期推进）
- 汇总 CSV 写入与追加
- 与 run_logger 的集成
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from src.lazybull.ml.walk_forward_utils import (
    generate_walk_forward_splits,
    generate_walk_forward_splits_by_count,
    print_splits_summary,
    resolve_deploy_train_window,
    WalkForwardSplit
)
from src.lazybull.ml.run_logger import TrainingRunRecord, write_training_run_to_csv


class TestWalkForwardSplits:
    """测试 walk-forward 切分生成逻辑"""
    
    @pytest.fixture
    def trade_cal(self):
        """创建测试用交易日历"""
        # 生成2018-2023年的交易日（简化：每月22个交易日）
        dates = []
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        current = start_date
        while current <= end_date:
            # 跳过周末
            if current.weekday() < 5:
                dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        
        df = pd.DataFrame({
            'cal_date': dates,
            'is_open': [1] * len(dates)
        })
        
        return df
    
    def test_quarterly_splits_generation(self, trade_cal):
        """测试季度滚动切分生成"""
        splits = generate_walk_forward_splits(
            trade_cal=trade_cal,
            wf_start_date="20180101",
            wf_end_date="20231231",
            step_frequency="quarterly",
            train_window_years=3,  # 使用3年窗口便于测试
            test_window_months=3
        )
        
        # 验证生成了切分
        assert len(splits) > 0, "应该生成至少一个切分"
        
        # 验证切分结构
        for split in splits:
            assert isinstance(split, WalkForwardSplit)
            assert split.train_start < split.train_end
            assert split.test_start > split.train_end
            assert split.test_start < split.test_end
            
        # 验证切分索引连续
        for i, split in enumerate(splits):
            assert split.split_index == i
        
        # 验证第一个切分的训练窗口大约是3年
        first_split = splits[0]
        train_start_dt = datetime.strptime(first_split.train_start, "%Y%m%d")
        train_end_dt = datetime.strptime(first_split.train_end, "%Y%m%d")
        train_duration_days = (train_end_dt - train_start_dt).days
        
        # 3年大约是1095天（允许误差）
        assert 900 < train_duration_days < 1200, f"训练窗口应约为3年，实际为 {train_duration_days} 天"
    
    def test_monthly_splits_generation(self, trade_cal):
        """测试月度滚动切分生成"""
        splits = generate_walk_forward_splits(
            trade_cal=trade_cal,
            wf_start_date="20190101",  # 使用更长的日期范围
            wf_end_date="20221231",
            step_frequency="monthly",
            train_window_years=2,
            test_window_months=1
        )
        
        # 月度滚动应该生成更多切分
        assert len(splits) > 0
        
        # 验证切分间隔（大约1个月）
        if len(splits) >= 2:
            split1 = splits[0]
            split2 = splits[1]
            
            date1 = datetime.strptime(split1.train_end, "%Y%m%d")
            date2 = datetime.strptime(split2.train_end, "%Y%m%d")
            
            days_diff = (date2 - date1).days
            # 1个月大约20-45天
            assert 20 <= days_diff <= 45, f"月度滚动间隔应约为1个月，实际为 {days_diff} 天"
    
    def test_semiannual_splits_generation(self, trade_cal):
        """测试半年度滚动切分生成"""
        splits = generate_walk_forward_splits(
            trade_cal=trade_cal,
            wf_start_date="20200101",
            wf_end_date="20221231",
            step_frequency="semiannual",
            train_window_years=2,
            test_window_months=6
        )
        
        assert len(splits) > 0
        
        # 验证切分间隔（大约6个月）
        if len(splits) >= 2:
            split1 = splits[0]
            split2 = splits[1]
            
            date1 = datetime.strptime(split1.train_end, "%Y%m%d")
            date2 = datetime.strptime(split2.train_end, "%Y%m%d")
            
            days_diff = (date2 - date1).days
            # 6个月大约150-200天
            assert 150 <= days_diff <= 200, f"半年度滚动间隔应约为6个月，实际为 {days_diff} 天"
    
    def test_splits_no_overlap(self, trade_cal):
        """测试切分之间测试集不重叠"""
        splits = generate_walk_forward_splits(
            trade_cal=trade_cal,
            wf_start_date="20200101",
            wf_end_date="20221231",
            step_frequency="quarterly",
            train_window_years=2,
            test_window_months=3
        )
        
        if len(splits) >= 2:
            for i in range(len(splits) - 1):
                split1 = splits[i]
                split2 = splits[i + 1]
                
                # 下一个切分的测试开始应该晚于当前切分的测试开始
                assert split2.test_start >= split1.test_start
    
    def test_splits_boundary_conditions(self, trade_cal):
        """测试边界条件"""
        # 测试窗口过大，无法生成切分
        splits = generate_walk_forward_splits(
            trade_cal=trade_cal,
            wf_start_date="20230101",
            wf_end_date="20231231",
            step_frequency="quarterly",
            train_window_years=10,  # 10年窗口，数据不足
            test_window_months=3
        )
        
        # 应该生成很少或没有切分
        assert len(splits) >= 0
    
    def test_invalid_step_frequency(self, trade_cal):
        """测试无效的 step 频率"""
        with pytest.raises(ValueError, match="不支持的 step_frequency"):
            generate_walk_forward_splits(
                trade_cal=trade_cal,
                wf_start_date="20200101",
                wf_end_date="20221231",
                step_frequency="invalid",
                train_window_years=2,
                test_window_months=3
            )
    
    def test_print_splits_summary(self, trade_cal):
        """测试打印切分汇总"""
        splits = generate_walk_forward_splits(
            trade_cal=trade_cal,
            wf_start_date="20190101",  # 使用更长的日期范围
            wf_end_date="20221231",
            step_frequency="quarterly",
            train_window_years=2,
            test_window_months=3
        )
        
        # 只测试不抛异常
        print_splits_summary(splits)
        deploy_train_start, deploy_train_end = resolve_deploy_train_window(
            trade_cal=trade_cal,
            deploy_train_end=splits[-1].test_end,
            train_window_years=2,
        )
        print_splits_summary(
            splits,
            deploy_train_start=deploy_train_start,
            deploy_train_end=deploy_train_end,
        )
        assert len(splits) > 0
        
        # 测试空列表
        print_splits_summary([])  # 不应抛异常

    def test_resolve_deploy_train_window(self, trade_cal):
        """测试部署训练区间解析与交易日对齐"""
        train_start, train_end = resolve_deploy_train_window(
            trade_cal=trade_cal,
            deploy_train_end="20221231",  # 非交易日（周六）
            train_window_years=2,
        )

        assert train_start == "20201231"
        assert train_end == "20221230"

    def test_rebalance_freq_alignment(self, trade_cal):
        """测试 rebalance_freq 对齐：test_end 应延迟到调仓日边界"""
        rebalance_freq = 5
        splits = generate_walk_forward_splits(
            trade_cal=trade_cal,
            wf_start_date="20190101",
            wf_end_date="20221231",
            step_frequency="quarterly",
            train_window_years=2,
            test_window_months=3,
            rebalance_freq=rebalance_freq
        )
        assert len(splits) > 0

        all_trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()

        # 验证非最后一个 split 的 test_end 与 test_start 的交易日间隔是 rebalance_freq 的整数倍
        for split in splits[:-1]:
            test_start_idx = all_trade_dates.index(split.test_start)
            test_end_idx = all_trade_dates.index(split.test_end)
            interval = test_end_idx - test_start_idx + 1
            assert interval % rebalance_freq == 0, (
                f"split {split.split_index}: test 区间交易日数 {interval} 不是 rebalance_freq={rebalance_freq} 的整数倍"
            )

    def test_last_split_capped_by_wf_end_date(self, trade_cal):
        """测试最后一个 split 的 test_end 不超过 wf_end_date"""
        wf_end_date = "20221231"
        splits = generate_walk_forward_splits(
            trade_cal=trade_cal,
            wf_start_date="20190101",
            wf_end_date=wf_end_date,
            step_frequency="quarterly",
            train_window_years=2,
            test_window_months=6,
            rebalance_freq=5
        )
        assert len(splits) > 0
        assert splits[-1].test_end <= wf_end_date, (
            f"最后一个 split 的 test_end {splits[-1].test_end} 超出了 wf_end_date {wf_end_date}"
        )

    def test_generate_splits_by_count(self, trade_cal):
        """测试按 split 数量 + final_date 反推切分"""
        splits = generate_walk_forward_splits_by_count(
            trade_cal=trade_cal,
            split_count=6,
            final_date="20221231",
            step_frequency="quarterly",
            train_window_years=2,
            test_window_months=3,
            rebalance_freq=5,
        )

        assert len(splits) == 6
        assert [s.split_index for s in splits] == list(range(6))
        assert splits[-1].test_end <= "20221231"

        for split in splits:
            assert split.train_start < split.train_end
            assert split.test_start > split.train_end
            assert split.test_start <= split.test_end

        # 验证测试区间严格连续：split[i].test_end 与 split[i+1].test_start 为连续交易日
        all_trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
        for i in range(len(splits) - 1):
            end_idx = all_trade_dates.index(splits[i].test_end)
            assert end_idx + 1 < len(all_trade_dates), (
                f"split {i} test_end={splits[i].test_end} 已是最后一个交易日"
            )
            assert all_trade_dates[end_idx + 1] == splits[i + 1].test_start, (
                f"split {i} 与 split {i+1} 测试区间不连续: "
                f"test_end={splits[i].test_end}, "
                f"下一个交易日={all_trade_dates[end_idx + 1]}, "
                f"next.test_start={splits[i + 1].test_start}"
            )

    def test_generate_splits_by_count_contiguous_when_window_gt_step(self, trade_cal):
        """测试 test_window_months > step_months 时测试区间仍严格连续（无缺口）"""
        splits = generate_walk_forward_splits_by_count(
            trade_cal=trade_cal,
            split_count=5,
            final_date="20221231",
            step_frequency="quarterly",   # 3 个月
            train_window_years=3,
            test_window_months=6,          # 6 个月 > 3 个月步长
        )

        assert len(splits) == 5

        for split in splits:
            assert split.train_start < split.train_end
            assert split.test_start > split.train_end
            assert split.test_start <= split.test_end

        # 验证测试区间严格连续
        all_trade_dates = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
        for i in range(len(splits) - 1):
            end_idx = all_trade_dates.index(splits[i].test_end)
            assert end_idx + 1 < len(all_trade_dates)
            assert all_trade_dates[end_idx + 1] == splits[i + 1].test_start, (
                f"split {i} 与 split {i+1} 测试区间不连续（存在缺口）: "
                f"test_end={splits[i].test_end}, "
                f"下一个交易日={all_trade_dates[end_idx + 1]}, "
                f"next.test_start={splits[i + 1].test_start}"
            )

    def test_generate_splits_by_count_invalid_split_count(self, trade_cal):
        """测试按数量反推时 split_count 参数校验"""
        with pytest.raises(ValueError, match="split_count"):
            generate_walk_forward_splits_by_count(
                trade_cal=trade_cal,
                split_count=0,
                final_date="20221231",
                step_frequency="quarterly",
                train_window_years=2,
                test_window_months=3,
            )

    def test_filter_splits_by_selected_indices(self):
        """测试按 split 下标筛选：空列表=全部，非空=指定子集。"""
        from scripts.walk_forward import _filter_splits_by_selected_indices

        splits = [
            WalkForwardSplit(
                split_index=i,
                train_start="20200101",
                train_end="20201231",
                test_start="20210101",
                test_end="20210331",
            )
            for i in range(6)
        ]

        selected = _filter_splits_by_selected_indices(splits, [0, 4, 5])
        assert [s.split_index for s in selected] == [0, 4, 5]

        selected_all = _filter_splits_by_selected_indices(splits, [])
        assert [s.split_index for s in selected_all] == [0, 1, 2, 3, 4, 5]

    def test_filter_splits_by_selected_indices_invalid(self):
        """测试按 split 下标筛选时，不存在的下标会报错。"""
        from scripts.walk_forward import _filter_splits_by_selected_indices

        splits = [
            WalkForwardSplit(
                split_index=i,
                train_start="20200101",
                train_end="20201231",
                test_start="20210101",
                test_end="20210331",
            )
            for i in range(3)
        ]

        with pytest.raises(ValueError, match="selected_split_indices"):
            _filter_splits_by_selected_indices(splits, [0, 4])


class TestWalkForwardCSV:
    """测试 walk-forward 汇总CSV生成"""
    
    def test_write_walk_forward_summary(self):
        """测试写入汇总文件"""
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_summary.csv")
            
            # 准备测试数据
            results = [
                {
                    "split_index": 0,
                    "train_start": "20200101",
                    "train_end": "20221231",
                    "test_start": "20230101",
                    "test_end": "20230331",
                    "model_version": 1,
                    "train_samples": 10000,
                    "val_samples": 2000,
                    "test_samples": 3000,
                    "test_daily_metrics": {
                        "daily_rankic_mean": 0.05,
                        "daily_rankic_std": 0.02,
                        "top30_return_mean": 0.03,
                        "top100_return_mean": 0.025
                    }
                },
                {
                    "split_index": 1,
                    "train_start": "20200401",
                    "train_end": "20230331",
                    "test_start": "20230401",
                    "test_end": "20230630",
                    "model_version": 2,
                    "train_samples": 10500,
                    "val_samples": 2100,
                    "test_samples": 3100,
                    "test_daily_metrics": {
                        "daily_rankic_mean": 0.06,
                        "daily_rankic_std": 0.025,
                        "top30_return_mean": 0.035,
                        "top100_return_mean": 0.028
                    }
                }
            ]
            
            # 导入函数
            from scripts.walk_forward import write_walk_forward_summary
            import types

            # 构造 mock args（包含函数内部需要的所有属性）
            mock_args = types.SimpleNamespace(
                wf_start_date="20200101", wf_end_date="20230630",
                batch_run_id="wf_batch_test_001",
                batch_period_label="0101",
                algorithm="xgboost", step=3, train_window_years=3,
                test_window_months=3, val_ratio=0.2,
                label_column="y_ret_5", task="regression",
                label_transform="cs_zscore",
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=1,
                early_stopping_rounds=50, early_stopping_metric="rmse",
                rank_weight_enabled=False, rank_weight_topk=100, rank_weight=2.0,
                time_decay_half_life=0,
                enable_fundamental_features=False, enable_alt_features=False,
                enable_margin_features=False,
                enable_cyq_features=False, enable_fund_features=False,
                enable_express_features=False,
                feature_stability_filter=False,
                oos_backtest=True,
                bt_sell_timing="close",
                bt_exclude_st=False,
                bt_min_list_days=180,
                bt_max_weight_per_stock=0.15,
                bt_max_per_industry=2,
                bt_stop_loss_enabled=True,
                bt_stop_loss_drawdown_pct=18.0,
                bt_stop_loss_trailing_enabled=True,
                bt_stop_loss_trailing_pct=10.0,
                bt_stop_loss_consecutive_limit_down=3,
                bt_equity_curve_enabled=True,
                bt_equity_curve_drawdown_thresholds=[6.0, 12.0],
                bt_equity_curve_exposure_levels=[0.8, 0.5],
                bt_equity_curve_ma_short=7,
                bt_equity_curve_ma_long=21,
                bt_equity_curve_recovery_mode="immediate",
                bt_equity_curve_recovery_step=0.5,
                bt_equity_curve_recovery_delay_periods=2,
                signal_confidence_gate_enabled=True,
                signal_confidence_gate_top_k=8,
                signal_confidence_gate_thresholds=[0.1, 0.3],
                signal_confidence_gate_exposure_levels=[0.4, 1.0],
            )

            # 写入文件
            write_walk_forward_summary(results, output_path, mock_args, "wf_test_001")
            
            # 验证文件存在
            assert os.path.exists(output_path)
            
            # 读取并验证内容
            df = pd.read_csv(output_path)
            
            assert len(df) == 2
            assert "split_index" in df.columns
            assert "train_start" in df.columns
            assert "test_start" in df.columns
            assert "daily_rankic_mean" in df.columns
            assert "top30_return_mean" in df.columns
            assert "bt_sell_timing" in df.columns
            assert "bt_max_weight_per_stock" in df.columns
            assert "bt_stop_loss_enabled" in df.columns
            assert "bt_equity_curve_enabled" in df.columns
            assert "signal_confidence_gate_enabled" in df.columns
            assert "signal_confidence_gate_top_k" in df.columns
            assert "signal_confidence_gate_thresholds" in df.columns
            assert "signal_confidence_gate_exposure_levels" in df.columns
            assert "batch_run_id" in df.columns
            assert "batch_period_label" in df.columns
            
            # 验证数据正确
            assert df.loc[0, "split_index"] == 0
            assert df.loc[0, "daily_rankic_mean"] == 0.05
            assert df.loc[0, "bt_sell_timing"] == "close"
            assert str(df.loc[0, "bt_exclude_st"]).lower() == "false"
            assert df.loc[0, "bt_min_list_days"] == 180
            assert df.loc[0, "bt_max_weight_per_stock"] == 0.15
            assert df.loc[0, "bt_max_per_industry"] == 2
            assert str(df.loc[0, "bt_stop_loss_enabled"]).lower() == "true"
            assert df.loc[0, "bt_stop_loss_drawdown_pct"] == 18.0
            assert str(df.loc[0, "bt_equity_curve_enabled"]).lower() == "true"
            assert "6.0" in str(df.loc[0, "bt_equity_curve_drawdown_thresholds"])
            assert "0.8" in str(df.loc[0, "bt_equity_curve_exposure_levels"])
            assert str(df.loc[0, "signal_confidence_gate_enabled"]).lower() == "true"
            assert df.loc[0, "signal_confidence_gate_top_k"] == 8
            assert "0.1" in str(df.loc[0, "signal_confidence_gate_thresholds"])
            assert "0.4" in str(df.loc[0, "signal_confidence_gate_exposure_levels"])
            assert df.loc[0, "batch_run_id"] == "wf_batch_test_001"
            assert str(df.loc[0, "batch_period_label"]).zfill(4) == "0101"
            assert df.loc[1, "split_index"] == 1
            assert df.loc[1, "daily_rankic_mean"] == 0.06

    def test_write_walk_forward_summary_clears_inactive_conditional_params(self):
        """未启用父开关时，summary 不应写入误导性的默认子参数。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_summary_inactive.csv")

            results = [
                {
                    "split_index": 0,
                    "train_start": "20200101",
                    "train_end": "20221231",
                    "test_start": "20230101",
                    "test_end": "20230331",
                    "model_version": 1,
                    "test_daily_metrics": {},
                }
            ]

            from scripts.walk_forward import write_walk_forward_summary
            import types

            mock_args = types.SimpleNamespace(
                wf_start_date="20200101", wf_end_date="20230630",
                batch_run_id="wf_batch_test_002",
                batch_period_label="0209",
                algorithm="xgboost", step=3, train_window_years=3,
                test_window_months=3, val_ratio=0.2,
                label_column="y_ret_5", task="regression",
                label_transform="cs_zscore",
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=1,
                early_stopping_rounds=50, early_stopping_metric="rmse",
                rank_weight_enabled=False, rank_weight_topk=100, rank_weight=2.0,
                time_decay_half_life=0,
                enable_fundamental_features=False, enable_alt_features=False,
                enable_margin_features=False,
                enable_cyq_features=False, enable_fund_features=False,
                enable_express_features=False,
                feature_stability_filter=False,
                oos_backtest=True,
                signal_gate_mode="disabled",
                signal_gate_cost_multiplier=1.8,
                signal_gate_round_trip_cost=0.004,
                signal_gate_percentile_warmup=9,
                signal_confidence_gate_enabled=True,
                signal_confidence_gate_top_k=8,
                signal_confidence_gate_thresholds=[0.1, 0.3],
                signal_confidence_gate_exposure_levels=[0.4, 1.0],
                signal_gate_quality_enabled=False,
                signal_gate_quality_window=7,
                signal_gate_quality_threshold=0.55,
                signal_gate_quality_halflife=5,
                signal_gate_dynamic_topn=True,
                signal_gate_topn_high_multiplier=0.5,
                signal_gate_topn_low_multiplier=1.6,
                holding_bonus_enabled=False,
                holding_bonus_sigma=0.8,
                bt_sell_timing="open",
                bt_exclude_st=True,
                bt_min_list_days=365,
                bt_stop_loss_enabled=False,
                bt_stop_loss_drawdown_pct=18.0,
                bt_stop_loss_trailing_enabled=True,
                bt_stop_loss_trailing_pct=10.0,
                bt_stop_loss_consecutive_limit_down=3,
                bt_equity_curve_enabled=False,
                bt_equity_curve_drawdown_thresholds=[6.0, 12.0],
                bt_equity_curve_exposure_levels=[0.8, 0.5],
                bt_equity_curve_ma_short=7,
                bt_equity_curve_ma_long=21,
                bt_equity_curve_recovery_mode="immediate",
                bt_equity_curve_recovery_step=0.5,
                bt_equity_curve_recovery_delay_periods=2,
                industry_momentum_filter=False,
                industry_momentum_bottom_pct=0.4,
                industry_rotation_enhanced=False,
                industry_rotation_alpha=0.6,
                position_sizing="equal",
                kelly_vol_window=90,
                kelly_max_leverage=0.4,
                market_regime=False,
                market_regime_mode="combined",
                market_regime_bear_threshold=-0.03,
                market_regime_bear_exposure=0.2,
                market_regime_vol_target=0.18,
                market_regime_trend_threshold=1.1,
                market_regime_min_exposure=0.3,
                market_regime_combine_method="multiply",
                market_regime_trend_guard=False,
                market_regime_drawdown_guard=True,
                market_regime_drawdown_threshold=-0.09,
                market_regime_ma250_hard_stop=False,
                market_regime_ma250_threshold=1.02,
                market_regime_ma250_exposure=0.5,
                market_regime_ma250_atr_scaling=True,
                enable_profit_based_holding=False,
                early_exit_loss_threshold=-0.09,
                early_exit_holding_ratio=1.0,
                profit_extension_threshold=0.1,
                profit_extension_days=5,
                profit_extension_mode="pnl",
                profit_extension_strength_threshold=0.75,
                use_atr_for_early_exit=True,
                atr_multiplier=2.8,
                early_exit_mode="strength_veto",
                early_exit_strength_protect_threshold=0.55,
                early_exit_max_reprieves=2,
                take_profit_threshold=None,
                take_profit_refill=True,
                enable_early_rebalance_on_empty=True,
            )

            write_walk_forward_summary(results, output_path, mock_args, "wf_test_002")

            df = pd.read_csv(output_path)
            row = df.loc[0]

            assert pd.isna(row["signal_gate_cost_multiplier"])
            assert pd.isna(row["signal_gate_round_trip_cost"])
            assert pd.isna(row["signal_gate_percentile_warmup"])
            assert pd.isna(row["signal_confidence_gate_enabled"])
            assert pd.isna(row["signal_confidence_gate_top_k"])
            assert pd.isna(row["signal_confidence_gate_thresholds"])
            assert pd.isna(row["signal_confidence_gate_exposure_levels"])
            assert pd.isna(row["signal_gate_quality_window"])
            assert pd.isna(row["signal_gate_quality_threshold"])
            assert pd.isna(row["signal_gate_quality_halflife"])
            assert pd.isna(row["signal_gate_dynamic_topn"])
            assert pd.isna(row["signal_gate_topn_high_multiplier"])
            assert pd.isna(row["signal_gate_topn_low_multiplier"])
            assert pd.isna(row["holding_bonus_sigma"])
            assert pd.isna(row["bt_stop_loss_drawdown_pct"])
            assert pd.isna(row["bt_stop_loss_trailing_enabled"])
            assert pd.isna(row["bt_stop_loss_trailing_pct"])
            assert pd.isna(row["bt_stop_loss_consecutive_limit_down"])
            assert pd.isna(row["bt_equity_curve_drawdown_thresholds"])
            assert pd.isna(row["bt_equity_curve_exposure_levels"])
            assert pd.isna(row["bt_equity_curve_ma_short"])
            assert pd.isna(row["bt_equity_curve_ma_long"])
            assert pd.isna(row["industry_momentum_bottom_pct"])
            assert pd.isna(row["industry_rotation_alpha"])
            assert pd.isna(row["kelly_vol_window"])
            assert pd.isna(row["kelly_max_leverage"])
            assert pd.isna(row["market_regime_mode"])
            assert pd.isna(row["market_regime_bear_threshold"])
            assert pd.isna(row["market_regime_vol_target"])
            assert pd.isna(row["market_regime_trend_guard"])
            assert pd.isna(row["market_regime_drawdown_threshold"])
            assert pd.isna(row["market_regime_ma250_threshold"])
            assert pd.isna(row["market_regime_ma250_exposure"])
            assert pd.isna(row["market_regime_ma250_atr_scaling"])
            assert pd.isna(row["early_exit_loss_threshold"])
            assert pd.isna(row["early_exit_holding_ratio"])
            assert pd.isna(row["profit_extension_threshold"])
            assert pd.isna(row["profit_extension_days"])
            assert pd.isna(row["profit_extension_mode"])
            assert pd.isna(row["profit_extension_strength_threshold"])
            assert pd.isna(row["use_atr_for_early_exit"])
            assert pd.isna(row["atr_multiplier"])
            assert pd.isna(row["early_exit_mode"])
            assert pd.isna(row["early_exit_strength_protect_threshold"])
            assert pd.isna(row["early_exit_max_reprieves"])
            assert pd.isna(row["take_profit_refill"])

    def test_write_walk_forward_summary_keeps_composite_top_k_and_clears_drawdown_threshold(self):
        """composite 模式保留 top_k，关闭回撤保护时清空其阈值。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_summary_composite.csv")

            results = [
                {
                    "split_index": 0,
                    "train_start": "20200101",
                    "train_end": "20221231",
                    "test_start": "20230101",
                    "test_end": "20230331",
                    "model_version": 1,
                    "test_daily_metrics": {},
                }
            ]

            from scripts.walk_forward import write_walk_forward_summary
            import types

            mock_args = types.SimpleNamespace(
                wf_start_date="20200101", wf_end_date="20230630",
                batch_run_id="wf_batch_test_003",
                batch_period_label="0315",
                algorithm="xgboost", step=3, train_window_years=3,
                test_window_months=3, val_ratio=0.2,
                label_column="y_ret_5", task="regression",
                label_transform="cs_zscore",
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=1,
                early_stopping_rounds=50, early_stopping_metric="rmse",
                rank_weight_enabled=False, rank_weight_topk=100, rank_weight=2.0,
                time_decay_half_life=0,
                enable_fundamental_features=False, enable_alt_features=False,
                enable_margin_features=False,
                enable_cyq_features=False, enable_fund_features=False,
                enable_express_features=False,
                feature_stability_filter=False,
                oos_backtest=True,
                oos_backtest_months=3,
                bt_top_n=20,
                signal_gate_mode="composite",
                signal_gate_cost_multiplier=1.8,
                signal_gate_round_trip_cost=0.004,
                signal_gate_percentile_warmup=9,
                signal_confidence_gate_enabled=False,
                signal_confidence_gate_top_k=12,
                signal_confidence_gate_thresholds=[0.1, 0.3],
                signal_confidence_gate_exposure_levels=[0.4, 1.0],
                signal_gate_quality_enabled=False,
                signal_gate_quality_window=7,
                signal_gate_quality_threshold=0.55,
                signal_gate_quality_halflife=5,
                signal_gate_dynamic_topn=True,
                signal_gate_topn_high_multiplier=0.5,
                signal_gate_topn_low_multiplier=1.6,
                holding_bonus_enabled=False,
                holding_bonus_sigma=0.8,
                bt_sell_timing="open",
                bt_exclude_st=True,
                bt_min_list_days=365,
                bt_max_weight_per_stock=0.1,
                bt_max_per_industry=2,
                bt_stop_loss_enabled=False,
                bt_stop_loss_drawdown_pct=18.0,
                bt_stop_loss_trailing_enabled=False,
                bt_stop_loss_trailing_pct=10.0,
                bt_stop_loss_consecutive_limit_down=3,
                bt_equity_curve_enabled=False,
                bt_equity_curve_drawdown_thresholds=[6.0, 12.0],
                bt_equity_curve_exposure_levels=[0.8, 0.5],
                bt_equity_curve_ma_short=7,
                bt_equity_curve_ma_long=21,
                bt_equity_curve_recovery_mode="immediate",
                bt_equity_curve_recovery_step=0.5,
                bt_equity_curve_recovery_delay_periods=2,
                industry_momentum_filter=False,
                industry_momentum_bottom_pct=0.4,
                industry_rotation_enhanced=False,
                industry_rotation_alpha=0.6,
                position_sizing="equal",
                kelly_vol_window=90,
                kelly_max_leverage=0.4,
                market_regime=True,
                market_regime_mode="combined",
                market_regime_bear_threshold=-0.03,
                market_regime_bear_exposure=0.2,
                market_regime_vol_target=0.18,
                market_regime_trend_threshold=1.1,
                market_regime_min_exposure=0.3,
                market_regime_combine_method="multiply",
                market_regime_trend_guard=True,
                market_regime_drawdown_guard=False,
                market_regime_drawdown_threshold=-0.09,
                market_regime_ma250_hard_stop=False,
                market_regime_ma250_threshold=1.02,
                market_regime_ma250_exposure=0.5,
                market_regime_ma250_atr_scaling=True,
                enable_profit_based_holding=False,
                early_exit_loss_threshold=-0.09,
                early_exit_holding_ratio=1.0,
                profit_extension_threshold=0.1,
                profit_extension_days=5,
                profit_extension_mode="pnl",
                profit_extension_strength_threshold=0.75,
                use_atr_for_early_exit=True,
                atr_multiplier=2.8,
                early_exit_mode="strength_veto",
                early_exit_strength_protect_threshold=0.55,
                early_exit_max_reprieves=2,
                take_profit_threshold=0.15,
                take_profit_refill=True,
                enable_early_rebalance_on_empty=True,
            )

            write_walk_forward_summary(results, output_path, mock_args, "wf_test_003")

            df = pd.read_csv(output_path)
            row = df.loc[0]

            assert row["signal_gate_mode"] == "composite"
            assert row["signal_confidence_gate_top_k"] == 12
            assert pd.isna(row["signal_confidence_gate_enabled"])
            assert pd.isna(row["signal_confidence_gate_thresholds"])
            assert pd.isna(row["signal_confidence_gate_exposure_levels"])
            assert str(row["signal_gate_dynamic_topn"]).lower() == "true"
            assert row["signal_gate_topn_high_multiplier"] == 0.5
            assert row["signal_gate_topn_low_multiplier"] == 1.6
            assert str(row["market_regime_drawdown_guard"]).lower() == "false"
            assert pd.isna(row["market_regime_drawdown_threshold"])

    def test_write_walk_forward_summary_keeps_binary_drawdown_guard(self):
        """binary 模式下回撤保护仍然生效，不应被误清空。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_summary_binary_guard.csv")

            results = [
                {
                    "split_index": 0,
                    "train_start": "20200101",
                    "train_end": "20221231",
                    "test_start": "20230101",
                    "test_end": "20230331",
                    "model_version": 1,
                    "test_daily_metrics": {},
                }
            ]

            from scripts.walk_forward import write_walk_forward_summary
            import types

            mock_args = types.SimpleNamespace(
                wf_start_date="20200101", wf_end_date="20230630",
                batch_run_id="wf_batch_test_004",
                batch_period_label="0412",
                algorithm="xgboost", step=3, train_window_years=3,
                test_window_months=3, val_ratio=0.2,
                label_column="y_ret_5", task="regression",
                label_transform="cs_zscore",
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=1, gamma=0, reg_alpha=0, reg_lambda=1,
                early_stopping_rounds=50, early_stopping_metric="rmse",
                rank_weight_enabled=False, rank_weight_topk=100, rank_weight=2.0,
                time_decay_half_life=0,
                enable_fundamental_features=False, enable_alt_features=False,
                enable_margin_features=False,
                enable_cyq_features=False, enable_fund_features=False,
                enable_express_features=False,
                feature_stability_filter=False,
                oos_backtest=True,
                oos_backtest_months=3,
                bt_top_n=20,
                signal_gate_mode="disabled",
                signal_gate_cost_multiplier=1.8,
                signal_gate_round_trip_cost=0.004,
                signal_gate_percentile_warmup=9,
                signal_confidence_gate_enabled=False,
                signal_confidence_gate_top_k=8,
                signal_confidence_gate_thresholds=[0.1, 0.3],
                signal_confidence_gate_exposure_levels=[0.4, 1.0],
                signal_gate_quality_enabled=False,
                signal_gate_quality_window=7,
                signal_gate_quality_threshold=0.55,
                signal_gate_quality_halflife=5,
                signal_gate_dynamic_topn=False,
                signal_gate_topn_high_multiplier=0.5,
                signal_gate_topn_low_multiplier=1.6,
                holding_bonus_enabled=False,
                holding_bonus_sigma=0.8,
                bt_sell_timing="open",
                bt_exclude_st=True,
                bt_min_list_days=365,
                bt_max_weight_per_stock=0.1,
                bt_max_per_industry=2,
                bt_stop_loss_enabled=False,
                bt_stop_loss_drawdown_pct=18.0,
                bt_stop_loss_trailing_enabled=False,
                bt_stop_loss_trailing_pct=10.0,
                bt_stop_loss_consecutive_limit_down=3,
                bt_equity_curve_enabled=False,
                bt_equity_curve_drawdown_thresholds=[6.0, 12.0],
                bt_equity_curve_exposure_levels=[0.8, 0.5],
                bt_equity_curve_ma_short=7,
                bt_equity_curve_ma_long=21,
                bt_equity_curve_recovery_mode="immediate",
                bt_equity_curve_recovery_step=0.5,
                bt_equity_curve_recovery_delay_periods=2,
                industry_momentum_filter=False,
                industry_momentum_bottom_pct=0.4,
                industry_rotation_enhanced=False,
                industry_rotation_alpha=0.6,
                position_sizing="equal",
                kelly_vol_window=90,
                kelly_max_leverage=0.4,
                market_regime=True,
                market_regime_mode="binary",
                market_regime_bear_threshold=-0.03,
                market_regime_bear_exposure=0.2,
                market_regime_vol_target=0.18,
                market_regime_trend_threshold=1.1,
                market_regime_min_exposure=0.3,
                market_regime_combine_method="multiply",
                market_regime_trend_guard=True,
                market_regime_drawdown_guard=True,
                market_regime_drawdown_threshold=-0.09,
                market_regime_ma250_hard_stop=False,
                market_regime_ma250_threshold=1.02,
                market_regime_ma250_exposure=0.5,
                market_regime_ma250_atr_scaling=True,
                enable_profit_based_holding=False,
                early_exit_loss_threshold=-0.09,
                early_exit_holding_ratio=1.0,
                profit_extension_threshold=0.1,
                profit_extension_days=5,
                profit_extension_mode="pnl",
                profit_extension_strength_threshold=0.75,
                use_atr_for_early_exit=True,
                atr_multiplier=2.8,
                early_exit_mode="strength_veto",
                early_exit_strength_protect_threshold=0.55,
                early_exit_max_reprieves=2,
                take_profit_threshold=0.15,
                take_profit_refill=True,
                enable_early_rebalance_on_empty=True,
            )

            write_walk_forward_summary(results, output_path, mock_args, "wf_test_004")

            df = pd.read_csv(output_path)
            row = df.loc[0]

            assert row["market_regime_mode"] == "binary"
            assert str(row["market_regime_drawdown_guard"]).lower() == "true"
            assert row["market_regime_drawdown_threshold"] == -0.09


class TestRunLoggerIntegration:
    """测试与 run_logger 的集成"""
    
    def test_training_run_record_with_wf_fields(self):
        """测试 TrainingRunRecord 包含 walk-forward 字段"""
        record = TrainingRunRecord(
            timestamp="2024-01-01 10:00:00",
            model_version=1,
            start_date="20200101",
            end_date="20221231",
            label_column="y_ret_5",
            task="regression",
            wf_run_id="wf_test_123",
            split_index=0,
            step_frequency="quarterly",
            test_start_date="20230101",
            test_end_date="20230331"
        )
        
        # 验证字段存在
        assert record.wf_run_id == "wf_test_123"
        assert record.split_index == 0
        assert record.step_frequency == "quarterly"
        assert record.test_start_date == "20230101"
        assert record.test_end_date == "20230331"
        
        # 验证转换为字典
        record_dict = record.to_dict()
        assert "wf_run_id" in record_dict
        assert "split_index" in record_dict
        assert "step_frequency" in record_dict
        assert "test_start_date" in record_dict
        assert "test_end_date" in record_dict
    
    def test_write_training_run_with_wf_fields(self):
        """测试写入包含 walk-forward 字段的训练记录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_ml_runs.csv")
            
            # 创建记录
            record1 = TrainingRunRecord(
                timestamp="2024-01-01 10:00:00",
                model_version=1,
                start_date="20200101",
                end_date="20221231",
                label_column="y_ret_5",
                task="regression",
                wf_run_id="wf_test_123",
                split_index=0,
                step_frequency="quarterly",
                test_start_date="20230101",
                test_end_date="20230331",
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05
            )
            
            record2 = TrainingRunRecord(
                timestamp="2024-01-01 11:00:00",
                model_version=2,
                start_date="20200401",
                end_date="20230331",
                label_column="y_ret_5",
                task="regression",
                wf_run_id="wf_test_123",
                split_index=1,
                step_frequency="quarterly",
                test_start_date="20230401",
                test_end_date="20230630",
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05
            )
            
            # 写入两条记录
            write_training_run_to_csv(record1, csv_path)
            write_training_run_to_csv(record2, csv_path)
            
            # 读取并验证
            df = pd.read_csv(csv_path)
            
            assert len(df) == 2
            assert "wf_run_id" in df.columns
            assert "split_index" in df.columns
            assert "step_frequency" in df.columns
            assert "test_start_date" in df.columns
            assert "test_end_date" in df.columns
            
            # 验证数据正确
            assert df.loc[0, "wf_run_id"] == "wf_test_123"
            assert df.loc[0, "split_index"] == 0
            assert df.loc[1, "split_index"] == 1
            assert df.loc[1, "step_frequency"] == "quarterly"
    
    def test_csv_dynamic_column_expansion(self):
        """测试 CSV 动态列扩展（新增 wf 字段时，旧记录自动留空）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_ml_runs.csv")
            
            # 写入一条旧记录（无 wf 字段）
            old_record = TrainingRunRecord(
                timestamp="2023-12-01 10:00:00",
                model_version=1,
                start_date="20200101",
                end_date="20221231",
                label_column="y_ret_5",
                task="regression",
                n_estimators=200,
                max_depth=8
            )
            
            write_training_run_to_csv(old_record, csv_path)
            
            # 写入一条新记录（有 wf 字段）
            new_record = TrainingRunRecord(
                timestamp="2024-01-01 10:00:00",
                model_version=2,
                start_date="20200401",
                end_date="20230331",
                label_column="y_ret_5",
                task="regression",
                wf_run_id="wf_test_123",
                split_index=0,
                n_estimators=200,
                max_depth=8
            )
            
            write_training_run_to_csv(new_record, csv_path)
            
            # 读取并验证
            df = pd.read_csv(csv_path)
            
            assert len(df) == 2
            assert "wf_run_id" in df.columns
            
            # 旧记录的 wf_run_id 应该是 NaN
            assert pd.isna(df.loc[0, "wf_run_id"])
            
            # 新记录的 wf_run_id 应该正确
            assert df.loc[1, "wf_run_id"] == "wf_test_123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

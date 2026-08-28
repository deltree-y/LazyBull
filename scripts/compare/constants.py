# -*- coding: utf-8 -*-
"""walk-forward 对比脚本常量与列名配置。"""

from typing import Optional

import pandas as pd

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))



SUMMARY_CSV_DTYPE = {
    "wf_run_id": str,
    "batch_run_id": str,
    "batch_period_label": str,
    "split_count": str,
    "final_date": str,
    "wf_start_date": str,
    "wf_end_date": str,
}


_TRADE_CAL_CACHE: dict[str, Optional[pd.DataFrame]] = {}
_TRADE_CAL_OPEN_DATES_CACHE: dict[int, tuple[list[str], dict[str, int]]] = {}
_BT_REBALANCE_FREQ_MAX = 60


# ---------------------------------------------------------------------------
# 输出列名：英文内部键 → 中文列名（用于最终 CSV 输出）
# ---------------------------------------------------------------------------
COL_NAMES = {
    # 标识
    "wf_run_id": "运行ID",
    "KEY_说明": "重点说明",
    "KEY_Top20_list": "重点Top20最新名单",
    "KEY_Top30_list": "重点Top30最新名单",
    "key_top20_hit_rate_mean": "重点Top20命中率均值",
    "key_top20_avg_return_median_mean": "重点Top20收益中位数均值",
    "key_top30_hit_rate_mean": "重点Top30命中率均值",
    "key_top30_avg_return_median_mean": "重点Top30收益中位数均值",
    "batch_run_id": "批次ID",
    "batch_period_label": "批次时间段",
    "split_count": "切分数量",
    "final_date": "最终日期",
    # OOS 性能
    "n_splits": "切分数",
    "model_version_range": "模型版本范围",
    "oos_rankic_ir_mean": "OOS_RankIC_IR均值",
    "oos_rankic_ir_std": "OOS_RankIC_IR标准差",
    "oos_cross_split_ir": "跨切分IR",
    "daily_rankic_mean": "RankIC均值",
    "icir": "ICIR",
    "selection_monotonicity": "分层单调性(近似)",
    "oos_rankic_ir_trend": "RankIC_IR趋势(近-早)",
    "oos_top30_median_mean": "Top30中位收益均值",
    "oos_top30_win_rate": "Top30胜率",
    "oos_top30_worst_median": "Top30最差中位收益",
    "oos_top30_skew_score_mean": "Top30偏斜度均值",
    "oos_top30_lift_mean": "Top30超额均值",
    "oos_top100_median_mean": "Top100中位收益均值",
    "oos_top100_win_rate": "Top100胜率",
    "oos_top300_median_mean": "Top300中位收益均值",
    "oos_top300_win_rate": "Top300胜率",
    # OOS 回测
    "chain_cagr": "全周期CAGR",
    "chain_total_return": "全周期总收益",
    "chain_max_drawdown": "全周期链式最大回撤",
    "chain_sharpe": "全周期链式夏普",
    "chain_trading_days": "全周期链式交易日数",
    "bt_annual_return_mean": "回测年化收益均值",
    "bt_sharpe_mean": "回测夏普均值",
    "bt_max_drawdown_worst": "回测最大回撤(最差)",
    "bt_calmar_mean": "回测Calmar均值",
    "bt_win_rate": "回测胜率",
    "bt_total_return_mean": "回测总收益均值",
    "bt_volatility_mean": "回测波动率均值",
    "bt_rebalance_freq": "回测调仓频率",
    "bt_initial_capital": "回测初始资金",
    "bt_sell_timing": "回测卖出时机",
    "bt_exclude_st": "回测排除ST",
    "bt_min_list_days": "回测最少上市天数",
    "bt_max_weight_per_stock": "回测单股最大权重",
    "bt_max_per_industry": "回测单行业最大持仓数",
    "bt_stop_loss_enabled": "回测止损",
    "bt_stop_loss_drawdown_pct": "回测回撤止损%",
    "bt_stop_loss_consecutive_limit_down": "回测连续跌停止损",
    # 训练质量
    "val_rankic_ir_mean": "验证集RankIC_IR均值",
    "train_val_ir_gap": "验证_OOS_IR差距",
    "best_iter_mean": "最佳迭代均值",
    "best_iter_min": "最佳迭代最小值",
    "best_iter_max": "最佳迭代最大值",
    "best_iter_std": "最佳迭代标准差",
    # 跨时间段稳定性
    "period_count": "时间段数",
    "period_labels": "时间段列表",
    "run_id_list": "运行ID列表",
    "score_mean": "综合得分均值",
    "score_std": "综合得分标准差",
    "score_min": "综合得分最差",
    "score_max": "综合得分最佳",
    "chain_cagr_mean": "跨时间段CAGR均值",
    "chain_cagr_std": "跨时间段CAGR标准差",
    "chain_cagr_min": "跨时间段CAGR最差",
    "chain_max_drawdown_mean": "跨时间段回撤均值",
    "chain_max_drawdown_worst": "跨时间段回撤最差",
    "oos_cross_split_ir_mean": "跨时间段跨切分IR均值",
    "oos_cross_split_ir_std": "跨时间段跨切分IR标准差",
    "bt_win_rate_mean": "跨时间段回测胜率均值",
    "bt_win_rate_min": "跨时间段回测胜率最差",
    "chain_sharpe_mean": "跨时间段夏普均值",
    "chain_sharpe_std": "跨时间段夏普标准差",
    "stability_score": "时间段稳定性分",
    "seed_alpha_median": "模型Alpha分中位数",
    # 训练参数
    "split_count": "切分数量",
    "final_date": "最终日期",
    "wf_start_date": "WF起始日期",
    "wf_end_date": "WF结束日期",
    "train_window_years": "训练窗口年数",
    "test_window_months": "测试窗口月数",
    "val_ratio": "验证集比例",
    "label_column": "标签列",
    "neutral_label_blend_weight": "中性标签混合权重",
    "task": "任务类型",
    "label_transform": "标签变换",
    "n_estimators": "树数量",
    "max_depth": "最大深度",
    "num_leaves": "LGB叶子数",
    "learning_rate": "学习率",
    "subsample": "样本采样比",
    "colsample_bytree": "特征采样比",
    "min_child_weight": "最小叶节点权重",
    "gamma": "gamma",
    "reg_alpha": "L1正则",
    "reg_lambda": "L2正则",
    "early_stopping_rounds": "早停轮数",
    "early_stopping_metric": "早停指标",
    "ensemble_seeds": "多种子bagging种子",
    "ensemble_seed_keep_top_ratio": "多种子保留比例",
    "ensemble_seed_keep_min_models": "多种子最少保留模型数",
    "rank_weight_enabled": "rank权重启用",
    "rank_weight_topk": "rank权重TopK",
    "rank_weight": "rank权重值",
    "rank_weight_topk_weight_mode": "rank权重TopK模式",
    "time_decay_half_life": "时间衰减半衰期",
    "freshness_strategy": "freshness策略",
    "event_freshness_half_life_days": "事件freshness半衰期天数",
    "objective": "目标函数",
    "algorithm": "算法",
    "enable_fundamental": "基本面因子",
    "enable_alt": "另类因子",
    "enable_margin": "融资融券因子",
    "enable_cyq": "筹码胜率因子",
    "enable_fund": "基金持仓因子",
    "enable_express": "业绩快报因子",
    "feature_stability_filter": "特征稳定性筛选",
    "factor_prune": "因子精简",
    "factor_exclude_file": "因子精简清单",
    "ensemble_offsets": "多偏移集成",
    "enable_enhanced_features": "增强因子",
    "enable_north_features": "北向资金因子",
    "enable_lhb_features": "龙虎榜因子",
    "enable_consensus_features": "一致预期因子",
    "enable_cashflow_quality_features": "现金流质量因子",
    "cashflow_quality_cols_live": "现金流质量实际入模列",
    "enable_consensus_revision_features": "一致预期修正因子",
    "oos_backtest": "OOS回测",
    "oos_backtest_months": "OOS回测月数",
    "bt_top_n": "回测TopN",
    # 盈亏动态持仓
    # ATR 动态阈值与仓位缩放
    # 亏损提前换出二次确认
    # 仓位管理
    "position_sizing": "仓位模式",
    "kelly_vol_window": "Kelly波动窗口",
    "kelly_max_leverage": "Kelly仓位上限",
    "stagger_tranches": "分批调仓批数",
    # 整体持仓止盈
    "enable_early_rebalance_on_empty": "空仓提前调仓",
    "skip_training": "跳过训练",
    "start_model_version": "起始模型版本",
    "no_deploy_train": "禁用部署训练",
}


BATCH_EXPERIMENT_CORE_COLS = [
    "运行ID",
    "最新运行时间",
    "批次ID",
    "批次时间段",
    "最终日期",
    "综合得分",
    "选股综合得分",
    "全周期CAGR",
    "全周期链式最大回撤",
    "全周期链式夏普",
    "跨切分IR",
    "回测胜率",
    "RankIC均值",
    "ICIR",
    "Top30超额均值",
    "Top30最差中位收益",
    "分层单调性(近似)",
    "验证_OOS_IR差距",
    "惩罚覆盖率",
    "替换收益贡献",
    "重点Top20命中率均值",
    "重点Top20收益中位数均值",
    "重点Top30命中率均值",
    "重点Top30收益中位数均值",
]


BATCH_EXPERIMENT_PARAM_CANDIDATES = [
    "标签列",
    "任务类型",
    "滚动频率",
    "训练窗口年数",
    "测试窗口月数",
    "树数量",
    "最大深度",
    "学习率",
    "rank权重TopK",
    "rank权重值",
    "多种子bagging种子",
    "多种子保留比例",
    "多种子最少保留模型数",
    "最佳迭代均值",
    "回测TopN",
    "回测调仓频率",
    "回测卖出时机",
    "市场择时",
    "盈亏动态持仓",
]

# ---------------------------------------------------------------------------
# 训练参数列（来自 write_walk_forward_summary 写入的列名，取每组第一行即可）
# ---------------------------------------------------------------------------
PARAM_COLS = [
    "wf_run_id",
    "batch_run_id",
    "batch_period_label",
    "algorithm",
    "split_count",
    "final_date",
    "wf_start_date",
    "wf_end_date",
    "train_window_years",
    "test_window_months",
    "val_ratio",
    "label_column",
    "neutral_label_blend_weight",
    "task",
    "label_transform",
    "n_estimators",
    "max_depth",
    "num_leaves",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "reg_alpha",
    "reg_lambda",
    "early_stopping_rounds",
    "early_stopping_metric",
    "ensemble_seeds",
    "ensemble_seed_keep_top_ratio",
    "ensemble_seed_keep_min_models",
    "rank_weight_enabled",
    "rank_weight_topk",
    "rank_weight",
    "time_decay_half_life",
    "freshness_strategy",
    "event_freshness_half_life_days",
    "objective",
    "enable_fundamental",
    "enable_alt",
    "enable_margin",
    "enable_cyq",
    "enable_fund",
    "enable_express",
    "feature_stability_filter",
    "factor_prune",
    "factor_exclude_file",
    "ensemble_offsets",
    "enable_enhanced_features",
    "enable_north_features",
    "enable_lhb_features",
    "enable_consensus_features",
    "enable_cashflow_quality_features",
    "enable_consensus_revision_features",
    "oos_backtest",
    "oos_backtest_months",
    "bt_top_n",
    "bt_rebalance_freq",
    "bt_initial_capital",
    "bt_sell_timing",
    "bt_exclude_st",
    "bt_min_list_days",
    "bt_max_weight_per_stock",
    "bt_max_per_industry",
    "bt_stop_loss_enabled",
    "bt_stop_loss_drawdown_pct",
    "bt_stop_loss_trailing_enabled",
    "bt_stop_loss_trailing_pct",
    "bt_stop_loss_consecutive_limit_down",
    # 行业轮动加权
    "industry_momentum_filter",
    "industry_momentum_bottom_pct",
    "industry_rotation_enhanced",
    "industry_rotation_alpha",
    # 仓位管理
    "position_sizing",
    "kelly_vol_window",
    "kelly_max_leverage",
    "market_regime",
    "market_regime_bear_threshold",
    "market_regime_bear_exposure",
    "market_regime_mode",
    "market_regime_vol_target",
    "market_regime_trend_threshold",
    "market_regime_min_exposure",
    "market_regime_combine_method",
    "market_regime_trend_guard",
    "market_regime_drawdown_guard",
    "market_regime_drawdown_threshold",
    "market_regime_ma250_hard_stop",
    "market_regime_ma250_threshold",
    "market_regime_ma250_exposure",
    "market_regime_ma250_atr_scaling",
    "stagger_tranches",
    "enable_early_rebalance_on_empty",
    "skip_training",
    "start_model_version",
    "no_deploy_train",
]


MODEL_PARAM_KEYS = [
    "algorithm",
    "split_count",
    "final_date",
    "wf_start_date",
    "wf_end_date",
    "train_window_years",
    "test_window_months",
    "val_ratio",
    "label_column",
    "neutral_label_blend_weight",
    "task",
    "label_transform",
    "n_estimators",
    "max_depth",
    "num_leaves",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "reg_alpha",
    "reg_lambda",
    "early_stopping_rounds",
    "early_stopping_metric",
    "ensemble_seeds",
    "ensemble_seed_keep_top_ratio",
    "ensemble_seed_keep_min_models",
    "rank_weight_enabled",
    "rank_weight_topk",
    "rank_weight",
    "rank_weight_topk_weight_mode",
    "time_decay_half_life",
    "freshness_strategy",
    "event_freshness_half_life_days",
    "objective",
    "enable_fundamental",
    "enable_alt",
    "enable_margin",
    "enable_cyq",
    "enable_fund",
    "enable_express",
    "feature_stability_filter",
    "factor_prune",
    "ensemble_offsets",
    "enable_enhanced_features",
    "enable_north_features",
    "enable_lhb_features",
    "enable_consensus_features",
    "enable_cashflow_quality_features",
    "enable_consensus_revision_features",
    "skip_training",
    "start_model_version",
    "no_deploy_train",
]


SEED_STABILITY_EXCLUDED_MODEL_KEYS = [
    "ensemble_seeds",
    "ensemble_seed_keep_top_ratio",
    "ensemble_seed_keep_min_models",
]


BATCH_EXPERIMENT_EXCLUDED_PARAM_COLS = {
    COL_NAMES["ensemble_seeds"],
    COL_NAMES["ensemble_seed_keep_top_ratio"],
    COL_NAMES["ensemble_seed_keep_min_models"],
}


TRADE_PARAM_KEYS = [
    "oos_backtest",
    "oos_backtest_months",
    "bt_top_n",
    "bt_rebalance_freq",
    "bt_initial_capital",
    "bt_sell_timing",
    "bt_exclude_st",
    "bt_min_list_days",
    "bt_max_weight_per_stock",
    "bt_max_per_industry",
    "bt_stop_loss_enabled",
    "bt_stop_loss_drawdown_pct",
    "bt_stop_loss_trailing_enabled",
    "bt_stop_loss_trailing_pct",
    "bt_stop_loss_consecutive_limit_down",
    "industry_momentum_filter",
    "industry_momentum_bottom_pct",
    "industry_rotation_enhanced",
    "industry_rotation_alpha",
    "position_sizing",
    "kelly_vol_window",
    "kelly_max_leverage",
    "market_regime",
    "market_regime_bear_threshold",
    "market_regime_bear_exposure",
    "market_regime_mode",
    "market_regime_vol_target",
    "market_regime_trend_threshold",
    "market_regime_min_exposure",
    "market_regime_combine_method",
    "market_regime_trend_guard",
    "market_regime_drawdown_guard",
    "market_regime_drawdown_threshold",
    "market_regime_ma250_hard_stop",
    "market_regime_ma250_threshold",
    "market_regime_ma250_exposure",
    "market_regime_ma250_atr_scaling",
    "stagger_tranches",
    "enable_early_rebalance_on_empty",
]


PAIR_CONTEXT_KEYS = ["batch_period_label", "final_date", "split_count"]


MODEL_ALPHA_SCORE_CONFIG = [
    ("选股综合得分均值", 0.30, "high"),
    ("ICIR均值", 0.20, "high"),
    ("Top30超额均值", 0.15, "high"),
    ("Top30最差中位收益", 0.15, "high"),
    ("分层单调性均值", 0.10, "high"),
    ("验证_OOS_IR差距", 0.10, "low"),
]


TRADE_YIELD_SCORE_CONFIG = [
    ("CAGR配对百分位均值", 0.40, "high"),
    ("总收益配对百分位均值", 0.25, "high"),
    ("Calmar配对百分位均值", 0.15, "high"),
    ("夏普配对百分位均值", 0.10, "high"),
    ("胜率配对百分位均值", 0.10, "high"),
]


TRADE_ROBUST_SCORE_CONFIG = [
    ("最大回撤配对百分位均值", 0.35, "high"),
    ("Calmar配对百分位均值", 0.25, "high"),
    ("夏普配对百分位均值", 0.20, "high"),
    ("胜率配对百分位均值", 0.10, "high"),
    ("CAGR最差配对百分位", 0.10, "high"),
]


CANDIDATE_SCORE_CONFIG = [
    ("模型Alpha分", 0.45, "high"),
    ("交易收益分", 0.30, "high"),
    ("交易稳健分", 0.15, "high"),
    ("最差场景防守分", 0.10, "high"),
]


CANDIDATE_MIN_MODEL_ALPHA = 60.0
CANDIDATE_MIN_EFFECTIVE_PAIR_CONTEXTS = 2
CANDIDATE_MIN_CHAIN_MAX_DRAWDOWN = -0.35
CANDIDATE_MIN_CHAIN_CAGR_WORST = -0.05




















        # 末段可能被 final_date 截断；此时无法由边界稳定反推调仓频率，跳过该 probe。





















    # 同一批边界约束下，满足条件的最小频率就是实际调仓频率；更大的倍数只是在个别窗口上“碰巧也对齐”。
























        # 显式排除单个 frame 内全 NA 列，保持 pandas 旧版 concat 的 dtype 推断语义。




# ---------------------------------------------------------------------------
# 综合得分配置：(英文列键, 权重, 方向)
#   "high"    → 值越大越好
#   "low"     → 值越小越好
#   "abs_low" → 绝对值越小越好
# 权重之和应为 1.0
# ---------------------------------------------------------------------------
SCORE_CONFIG = [
    # ── 回测指标（60%）：真实组合模拟，最直接反映参数优劣 ──────────
    ("chain_cagr", 0.20, "high"),  # 全周期串联 CAGR：最终最该看的盈利能力
    ("bt_win_rate", 0.15, "high"),  # 回测胜率：各切分正收益占比
    ("bt_sharpe_mean", 0.15, "high"),  # 回测夏普均值：风险收益比
    ("chain_max_drawdown", 0.10, "high"),  # 全周期链式最大回撤：真实风险下限
    # ── 统计指标（32%）：辅助验证，防止回测过拟合 ─────────────────
    ("oos_cross_split_ir", 0.10, "high"),  # 跨切分IR：核心稳健性
    ("oos_top30_win_rate", 0.05, "high"),  # Top30胜率：策略持续性
    ("oos_top30_worst_median", 0.05, "high"),  # Top30最差切分：抗压能力
    ("oos_rankic_ir_trend", 0.05, "high"),  # IC趋势：alpha是否在衰减
    ("oos_top30_median_mean", 0.03, "high"),  # Top30中位收益：核心盈利能力
    ("oos_top30_lift_mean", 0.02, "high"),  # Top30超额：纯选股能力
    ("oos_top30_skew_score_mean", 0.02, "abs_low"),  # 偏斜度：极端日干扰风险（绝对值越小越好）
    # ── 训练质量（8%）：过拟合检测 ────────────────────────────────
    ("train_val_ir_gap", 0.08, "low"),  # 验证_OOS差距：过拟合风险（越小越好）
]

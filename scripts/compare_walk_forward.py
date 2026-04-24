#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Walk-forward 实验对比脚本

功能：
- 读取 data/walk_forward/raw/ 下所有汇总CSV（每次 walk_forward 运行生成一个）
- 按 wf_run_id 分组，跨 split 聚合各项指标
- 生成对比表格（行=实验，列=聚合指标+训练参数）
- 输出到 data/walk_forward/wf_comparison.csv

使用示例：
    python scripts/compare_walk_forward.py
    python scripts/compare_walk_forward.py --data-root ./data
    python scripts/compare_walk_forward.py --raw-dir ./data/walk_forward/raw --output ./data/walk_forward/wf_comparison.csv
"""

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.hyperlink import Hyperlink

from src.lazybull.common.config import get_data_root
from src.lazybull.common.logger import setup_logger


# ---------------------------------------------------------------------------
# 输出列名：英文内部键 → 中文列名（用于最终 CSV 输出）
# ---------------------------------------------------------------------------
COL_NAMES = {
    # 标识
    "wf_run_id":                  "运行ID",
    "batch_run_id":               "批次ID",
    "batch_period_label":         "批次时间段",
    # OOS 性能
    "n_splits":                   "切分数",
    "model_version_range":        "模型版本范围",
    "oos_rankic_ir_mean":         "OOS_RankIC_IR均值",
    "oos_rankic_ir_std":          "OOS_RankIC_IR标准差",
    "oos_cross_split_ir":         "跨切分IR",
    "oos_rankic_ir_trend":        "RankIC_IR趋势(近-早)",
    "oos_top30_median_mean":      "Top30中位收益均值",
    "oos_top30_win_rate":         "Top30胜率",
    "oos_top30_worst_median":     "Top30最差中位收益",
    "oos_top30_skew_score_mean":  "Top30偏斜度均值",
    "oos_top30_lift_mean":        "Top30超额均值",
    "oos_top100_median_mean":     "Top100中位收益均值",
    "oos_top100_win_rate":        "Top100胜率",
    "oos_top300_median_mean":     "Top300中位收益均值",
    "oos_top300_win_rate":        "Top300胜率",
    # OOS 回测
    "chain_cagr":                 "全周期CAGR",
    "chain_total_return":         "全周期总收益",
    "chain_max_drawdown":         "全周期链式最大回撤",
    "chain_sharpe":               "全周期链式夏普",
    "chain_trading_days":         "全周期链式交易日数",
    "bt_annual_return_mean":      "回测年化收益均值",
    "bt_sharpe_mean":             "回测夏普均值",
    "bt_max_drawdown_worst":      "回测最大回撤(最差)",
    "bt_calmar_mean":             "回测Calmar均值",
    "bt_win_rate":                "回测胜率",
    "bt_total_return_mean":       "回测总收益均值",
    "bt_volatility_mean":         "回测波动率均值",
    "bt_signal_confidence_block_rate_mean": "门控持币率均值",
    "bt_signal_confidence_avg_exposure_mean": "门控平均仓位",
    "bt_signal_confidence_avg_score_mean": "门控平均置信度",
    "bt_sell_timing":             "回测卖出时机",
    "signal_confidence_gate_enabled": "信号置信度门控",
    "signal_confidence_gate_top_k": "门控TopK",
    "signal_confidence_gate_thresholds": "门控阈值",
    "signal_confidence_gate_exposure_levels": "门控仓位系数",
    "bt_exclude_st":              "回测排除ST",
    "bt_min_list_days":           "回测最少上市天数",
    "bt_max_weight_per_stock":    "回测单股最大权重",
    "bt_max_per_industry":        "回测单行业最大持仓数",
    "bt_stop_loss_enabled":       "回测止损",
    "bt_stop_loss_drawdown_pct":  "回测回撤止损%",
    "bt_stop_loss_trailing_enabled": "回测移动止损",
    "bt_stop_loss_trailing_pct":  "回测移动止损%",
    "bt_stop_loss_consecutive_limit_down": "回测连续跌停止损",
    "bt_equity_curve_enabled":    "回测ECT",
    "bt_equity_curve_drawdown_thresholds": "回测ECT回撤阈值",
    "bt_equity_curve_exposure_levels": "回测ECT仓位系数",
    "bt_equity_curve_ma_short":   "回测ECT短均线",
    "bt_equity_curve_ma_long":    "回测ECT长均线",
    "bt_equity_curve_recovery_mode": "回测ECT恢复模式",
    "bt_equity_curve_recovery_step": "回测ECT恢复步长",
    "bt_equity_curve_recovery_delay_periods": "回测ECT恢复等待",
    # 训练质量
    "val_rankic_ir_mean":         "验证集RankIC_IR均值",
    "train_val_ir_gap":           "验证_OOS_IR差距",
    "best_iter_mean":             "最佳迭代均值",
    "best_iter_min":              "最佳迭代最小值",
    "best_iter_max":              "最佳迭代最大值",
    "best_iter_std":              "最佳迭代标准差",
    # 跨时间段稳定性
    "period_count":               "时间段数",
    "period_labels":              "时间段列表",
    "score_mean":                 "综合得分均值",
    "score_std":                  "综合得分标准差",
    "score_min":                  "综合得分最差",
    "score_max":                  "综合得分最佳",
    "chain_cagr_mean":            "跨时间段CAGR均值",
    "chain_cagr_std":             "跨时间段CAGR标准差",
    "chain_cagr_min":             "跨时间段CAGR最差",
    "chain_max_drawdown_mean":    "跨时间段回撤均值",
    "chain_max_drawdown_worst":   "跨时间段回撤最差",
    "oos_cross_split_ir_mean":    "跨时间段跨切分IR均值",
    "oos_cross_split_ir_std":     "跨时间段跨切分IR标准差",
    "bt_win_rate_mean":           "跨时间段回测胜率均值",
    "bt_win_rate_min":            "跨时间段回测胜率最差",
    "stability_score":            "时间段稳定性分",
    # 训练参数
    "wf_start_date":              "WF起始日期",
    "wf_end_date":                "WF结束日期",
    "step":                       "滚动频率",
    "train_window_years":         "训练窗口年数",
    "test_window_months":         "测试窗口月数",
    "val_ratio":                  "验证集比例",
    "label_column":               "标签列",
    "task":                       "任务类型",
    "label_transform":            "标签变换",
    "n_estimators":               "树数量",
    "max_depth":                  "最大深度",
    "learning_rate":              "学习率",
    "subsample":                  "样本采样比",
    "colsample_bytree":           "特征采样比",
    "min_child_weight":           "最小叶节点权重",
    "gamma":                      "gamma",
    "reg_alpha":                  "L1正则",
    "reg_lambda":                 "L2正则",
    "rank_weight_enabled":        "rank权重启用",
    "rank_weight_topk":           "rank权重TopK",
    "rank_weight":                "rank权重值",
    "algorithm":                  "算法",
    # 盈亏动态持仓
    "enable_profit_based_holding":  "盈亏动态持仓",
    "early_exit_loss_threshold":    "早退亏损阈值",
    "early_exit_holding_ratio":     "早退持仓比例",
    "profit_extension_threshold":   "盈利延持阈值",
    "profit_extension_days":        "盈利延持天数",
    "profit_extension_mode":        "盈利延持模式",
    "profit_extension_strength_threshold": "强势度阈值",
    # ATR 动态阈值与仓位缩放
    "use_atr_for_early_exit":       "ATR动态阈值",
    "atr_multiplier":               "ATR倍数",
    # 亏损提前换出二次确认
    "early_exit_mode":                        "早退模式",
    "early_exit_strength_protect_threshold":   "早退保护阈值",
    "early_exit_max_reprieves":               "早退最大缓刑",
    # 行业轮动加权
    "industry_rotation_enhanced":    "行业轮动加权",
    "industry_rotation_alpha":       "轮动Alpha",
    # 仓位管理
    "position_sizing":               "仓位模式",
    "kelly_vol_window":              "Kelly波动窗口",
    "kelly_max_leverage":            "Kelly仓位上限",
    "market_regime_ma250_hard_stop": "MA250硬条件",
    "market_regime_ma250_threshold": "MA250阈值",
    "market_regime_ma250_exposure":  "MA250仓位",
    "market_regime_ma250_atr_scaling": "MA250 ATR缩放",
    # 整体持仓止盈
    "take_profit_threshold":        "整体止盈阈值",
    "take_profit_refill":           "止盈后补仓",
}

# ---------------------------------------------------------------------------
# 训练参数列（来自 write_walk_forward_summary 写入的列名，取每组第一行即可）
# ---------------------------------------------------------------------------
PARAM_COLS = [
    "wf_run_id",
    "batch_run_id",
    "batch_period_label",
    "algorithm",
    "wf_start_date", "wf_end_date", "step",
    "train_window_years", "test_window_months", "val_ratio",
    "label_column", "task", "label_transform",
    "n_estimators", "max_depth", "learning_rate",
    "subsample", "colsample_bytree", "min_child_weight",
    "gamma", "reg_alpha", "reg_lambda",
    "rank_weight_enabled", "rank_weight_topk", "rank_weight",
    "signal_confidence_gate_enabled",
    "signal_confidence_gate_top_k",
    "signal_confidence_gate_thresholds",
    "signal_confidence_gate_exposure_levels",
    "bt_sell_timing", "bt_exclude_st", "bt_min_list_days",
    "bt_max_weight_per_stock", "bt_max_per_industry",
    "bt_stop_loss_enabled", "bt_stop_loss_drawdown_pct",
    "bt_stop_loss_trailing_enabled", "bt_stop_loss_trailing_pct",
    "bt_stop_loss_consecutive_limit_down",
    "bt_equity_curve_enabled", "bt_equity_curve_drawdown_thresholds",
    "bt_equity_curve_exposure_levels", "bt_equity_curve_ma_short",
    "bt_equity_curve_ma_long", "bt_equity_curve_recovery_mode",
    "bt_equity_curve_recovery_step", "bt_equity_curve_recovery_delay_periods",
    # 盈亏动态持仓
    "enable_profit_based_holding",
    "early_exit_loss_threshold", "early_exit_holding_ratio",
    "profit_extension_threshold", "profit_extension_days",
    "profit_extension_mode", "profit_extension_strength_threshold",
    # ATR 动态阈值与仓位缩放
    "use_atr_for_early_exit", "atr_multiplier",
    # 亏损提前换出二次确认
    "early_exit_mode", "early_exit_strength_protect_threshold", "early_exit_max_reprieves",
    # 行业轮动加权
    "industry_rotation_enhanced", "industry_rotation_alpha",
    # 仓位管理
    "position_sizing", "kelly_vol_window", "kelly_max_leverage",
    "market_regime_ma250_hard_stop", "market_regime_ma250_threshold",
    "market_regime_ma250_exposure", "market_regime_ma250_atr_scaling",
    # 整体持仓止盈
    "take_profit_threshold", "take_profit_refill",
]

# ---------------------------------------------------------------------------
# 综合得分配置：(英文列键, 权重, 方向)
#   "high"    → 值越大越好
#   "low"     → 值越小越好
#   "abs_low" → 绝对值越小越好
# 权重之和应为 1.0
# ---------------------------------------------------------------------------
SCORE_CONFIG = [
    # ── 回测指标（60%）：真实组合模拟，最直接反映参数优劣 ──────────
    ("chain_cagr",               0.20, "high"),     # 全周期串联 CAGR：最终最该看的盈利能力
    ("bt_win_rate",               0.15, "high"),     # 回测胜率：各切分正收益占比
    ("bt_sharpe_mean",            0.15, "high"),     # 回测夏普均值：风险收益比
    ("chain_max_drawdown",        0.10, "high"),     # 全周期链式最大回撤：真实风险下限
    # ── 统计指标（32%）：辅助验证，防止回测过拟合 ─────────────────
    ("oos_cross_split_ir",        0.10, "high"),     # 跨切分IR：核心稳健性
    ("oos_top30_win_rate",        0.05, "high"),     # Top30胜率：策略持续性
    ("oos_top30_worst_median",    0.05, "high"),     # Top30最差切分：抗压能力
    ("oos_rankic_ir_trend",       0.05, "high"),     # IC趋势：alpha是否在衰减
    ("oos_top30_median_mean",     0.03, "high"),     # Top30中位收益：核心盈利能力
    ("oos_top30_lift_mean",       0.02, "high"),     # Top30超额：纯选股能力
    ("oos_top30_skew_score_mean", 0.02, "abs_low"),  # 偏斜度：极端日干扰风险（绝对值越小越好）
    # ── 训练质量（8%）：过拟合检测 ────────────────────────────────
    ("train_val_ir_gap",          0.08, "low"),      # 验证_OOS差距：过拟合风险（越小越好）
]


def load_chain_metrics(raw_dir: Optional[Path], wf_run_id: str) -> dict:
    """读取 chain_nav 并计算全周期指标。"""
    empty = {
        "chain_total_return": None,
        "chain_cagr": None,
        "chain_max_drawdown": None,
        "chain_sharpe": None,
        "chain_trading_days": None,
    }
    if raw_dir is None:
        return empty

    chain_path = raw_dir / f"chain_nav_{wf_run_id}.csv"
    if not chain_path.exists():
        return empty

    try:
        chain_df = pd.read_csv(chain_path, encoding="utf-8-sig")
    except Exception as exc:
        logger.warning(f"读取 chain_nav 失败: {chain_path.name} — {exc}")
        return empty

    if chain_df.empty or "nav" not in chain_df.columns:
        return empty

    nav = pd.to_numeric(chain_df["nav"], errors="coerce").dropna()
    if nav.empty or nav.iloc[0] == 0:
        return empty

    trading_days = len(nav)
    years = trading_days / 252 if trading_days > 0 else 0
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else None
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_drawdown = drawdown.min() if not drawdown.empty else None
    daily_ret = nav.pct_change().dropna()
    volatility = daily_ret.std() * (252 ** 0.5) if len(daily_ret) > 1 else None
    sharpe = None
    if cagr is not None and volatility is not None and volatility > 0:
        sharpe = (cagr - 0.03) / volatility

    return {
        "chain_total_return": round(total_return, 6),
        "chain_cagr": round(cagr, 6) if cagr is not None else None,
        "chain_max_drawdown": round(max_drawdown, 6) if max_drawdown is not None else None,
        "chain_sharpe": round(sharpe, 4) if sharpe is not None else None,
        "chain_trading_days": trading_days,
    }


def load_all_summaries(raw_dir: Path) -> pd.DataFrame:
    """加载 raw/ 目录下所有 walk_forward 汇总 CSV"""
    csv_files = sorted(raw_dir.glob("walk_forward_summary_*.csv"))
    if len(csv_files) == 0:
        logger.warning(f"未找到任何汇总CSV: {raw_dir}")
        return pd.DataFrame()

    logger.info(f"找到 {len(csv_files)} 个汇总CSV文件")
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(
                f,
                encoding="utf-8-sig",
                dtype={
                    "wf_run_id": str,
                    "batch_run_id": str,
                    "batch_period_label": str,
                    "wf_start_date": str,
                    "wf_end_date": str,
                },
            )
            df["_source_file"] = f.name
            frames.append(df)
            logger.debug(f"  已加载: {f.name}（{len(df)} 行）")
        except Exception as e:
            logger.warning(f"  跳过（读取失败）: {f.name} — {e}")

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    logger.info(f"合并后总行数: {len(all_df)}，unique wf_run_id: {all_df['wf_run_id'].nunique() if 'wf_run_id' in all_df.columns else '?'}")
    return all_df


def aggregate_run(group: pd.DataFrame) -> dict:
    """对单个 wf_run_id 的所有 split 行进行聚合，返回一行对比指标"""
    row = {}
    n = len(group)
    row["n_splits"] = n

    # 模型版本范围（min~max）
    if "model_version" in group.columns:
        mv = group["model_version"].dropna()
        if len(mv):
            row["model_version_range"] = f"{int(mv.min())}~{int(mv.max())}"
        else:
            row["model_version_range"] = None
    else:
        row["model_version_range"] = None

    # -----------------------------------------------------------------------
    # OOS 性能指标（来自 test_daily_metrics 展开列）
    # -----------------------------------------------------------------------
    def safe_mean(col): return group[col].mean() if col in group.columns else None
    def safe_std(col):  return group[col].std()  if col in group.columns else None
    def safe_min(col):  return group[col].min()  if col in group.columns else None
    def safe_max(col):  return group[col].max()  if col in group.columns else None

    # OOS RankIC IR
    oos_ir_series = group["daily_rankic_ir"] if "daily_rankic_ir" in group.columns else pd.Series(dtype=float)
    oos_ir_mean = oos_ir_series.mean() if len(oos_ir_series) else None
    oos_ir_std  = oos_ir_series.std()  if len(oos_ir_series) > 1 else None
    row["oos_rankic_ir_mean"] = round(oos_ir_mean, 4) if oos_ir_mean is not None else None
    row["oos_rankic_ir_std"]  = round(oos_ir_std,  4) if oos_ir_std  is not None else None
    row["oos_cross_split_ir"] = round(oos_ir_mean / oos_ir_std, 3) if (oos_ir_mean and oos_ir_std and oos_ir_std != 0) else None

    # OOS RankIC 衰减检测（最近3个split均值 - 最早3个split均值）
    if len(oos_ir_series) >= 6:
        sorted_ir = group.sort_values("split_index")["daily_rankic_ir"] if "daily_rankic_ir" in group.columns else oos_ir_series
        row["oos_rankic_ir_trend"] = round(sorted_ir.iloc[-3:].mean() - sorted_ir.iloc[:3].mean(), 4)
    else:
        row["oos_rankic_ir_trend"] = None

    # Top30 指标（以中位数为核心，不受极端日干扰）
    med30_col  = "diagnostic_Top30_逐日均值_50分位"
    mean30_col = "diagnostic_Top30_逐日均值的均值"
    std30_col  = "diagnostic_Top30_逐日均值的标准差"
    lift30_col = "diagnostic_Top30_相对全市场提升_均值"

    if med30_col in group.columns:
        med30_series = group[med30_col].dropna()
        row["oos_top30_median_mean"]   = round(med30_series.mean(), 6) if len(med30_series) else None
        row["oos_top30_win_rate"]      = round((med30_series > 0).mean(), 3) if len(med30_series) else None
        row["oos_top30_worst_median"]  = round(med30_series.min(), 6) if len(med30_series) else None
    else:
        row["oos_top30_median_mean"] = row["oos_top30_win_rate"] = row["oos_top30_worst_median"] = None

    # Top30 偏斜度（均值/中位数 gap，衡量是否被极端日驱动）
    if all(c in group.columns for c in [mean30_col, med30_col, std30_col]):
        valid = group[[mean30_col, med30_col, std30_col]].dropna()
        if len(valid):
            skew_scores = (valid[mean30_col] - valid[med30_col]) / valid[std30_col].replace(0, np.nan)
            row["oos_top30_skew_score_mean"] = round(skew_scores.mean(), 3)
        else:
            row["oos_top30_skew_score_mean"] = None
    else:
        row["oos_top30_skew_score_mean"] = None

    row["oos_top30_lift_mean"] = round(safe_mean(lift30_col), 6) if safe_mean(lift30_col) is not None else None

    # Top100 指标
    med100_col = "diagnostic_Top100_逐日均值_50分位"
    if med100_col in group.columns:
        med100_series = group[med100_col].dropna()
        row["oos_top100_median_mean"] = round(med100_series.mean(), 6) if len(med100_series) else None
        row["oos_top100_win_rate"]    = round((med100_series > 0).mean(), 3) if len(med100_series) else None
    else:
        row["oos_top100_median_mean"] = row["oos_top100_win_rate"] = None

    # Top300 指标
    med300_col = "diagnostic_Top300_逐日均值_50分位"
    if med300_col in group.columns:
        med300_series = group[med300_col].dropna()
        row["oos_top300_median_mean"] = round(med300_series.mean(), 6) if len(med300_series) else None
        row["oos_top300_win_rate"]    = round((med300_series > 0).mean(), 3) if len(med300_series) else None
    else:
        row["oos_top300_median_mean"] = row["oos_top300_win_rate"] = None

    # -----------------------------------------------------------------------
    # OOS 回测指标（来自 run_oos_backtest 写入的 bt_* 列）
    # -----------------------------------------------------------------------
    if "bt_total_return" in group.columns:
        bt_ret = group["bt_total_return"].dropna()
        bt_ar  = group["bt_annual_return"].dropna() if "bt_annual_return" in group.columns else pd.Series(dtype=float)
        bt_sh  = group["bt_sharpe"].dropna()        if "bt_sharpe"        in group.columns else pd.Series(dtype=float)
        bt_md  = group["bt_max_drawdown"].dropna()  if "bt_max_drawdown"  in group.columns else pd.Series(dtype=float)
        bt_cal = group["bt_calmar"].dropna()         if "bt_calmar"        in group.columns else pd.Series(dtype=float)
        bt_vol = group["bt_volatility"].dropna()     if "bt_volatility"    in group.columns else pd.Series(dtype=float)

        row["bt_total_return_mean"]  = round(bt_ret.mean(), 6) if len(bt_ret) else None
        row["bt_annual_return_mean"] = round(bt_ar.mean(),  6) if len(bt_ar)  else None
        row["bt_sharpe_mean"]        = round(bt_sh.mean(),  4) if len(bt_sh)  else None
        row["bt_max_drawdown_worst"] = round(bt_md.min(),   6) if len(bt_md)  else None
        row["bt_calmar_mean"]        = round(bt_cal.mean(), 4) if len(bt_cal) else None
        row["bt_volatility_mean"]    = round(bt_vol.mean(), 6) if len(bt_vol) else None
        row["bt_win_rate"]           = round((bt_ret > 0).mean(), 3) if len(bt_ret) else None
        bt_gate_block = (
            group["bt_signal_confidence_block_rate"].dropna()
            if "bt_signal_confidence_block_rate" in group.columns else pd.Series(dtype=float)
        )
        bt_gate_exposure = (
            group["bt_signal_confidence_avg_exposure"].dropna()
            if "bt_signal_confidence_avg_exposure" in group.columns else pd.Series(dtype=float)
        )
        bt_gate_score = (
            group["bt_signal_confidence_avg_score"].dropna()
            if "bt_signal_confidence_avg_score" in group.columns else pd.Series(dtype=float)
        )
        row["bt_signal_confidence_block_rate_mean"] = (
            round(bt_gate_block.mean(), 6) if len(bt_gate_block) else None
        )
        row["bt_signal_confidence_avg_exposure_mean"] = (
            round(bt_gate_exposure.mean(), 6) if len(bt_gate_exposure) else None
        )
        row["bt_signal_confidence_avg_score_mean"] = (
            round(bt_gate_score.mean(), 6) if len(bt_gate_score) else None
        )
    else:
        for k in ["bt_total_return_mean", "bt_annual_return_mean", "bt_sharpe_mean",
                   "bt_max_drawdown_worst", "bt_calmar_mean", "bt_volatility_mean", "bt_win_rate",
                   "bt_signal_confidence_block_rate_mean",
                   "bt_signal_confidence_avg_exposure_mean",
                   "bt_signal_confidence_avg_score_mean"]:
            row[k] = None

    # -----------------------------------------------------------------------
    # 训练质量指标
    # -----------------------------------------------------------------------
    # 验证集 RankIC IR（val_rankic_ir 列，每 split 一个值）
    if "val_rankic_ir" in group.columns:
        val_ir_series = group["val_rankic_ir"].dropna()
        row["val_rankic_ir_mean"] = round(val_ir_series.mean(), 4) if len(val_ir_series) else None
        # 泛化差距（val IR 越接近 oos IR 越好；负值说明 oos 反而更好，通常是正常的）
        if row["val_rankic_ir_mean"] is not None and row["oos_rankic_ir_mean"] is not None:
            row["train_val_ir_gap"] = round(row["val_rankic_ir_mean"] - row["oos_rankic_ir_mean"], 4)
        else:
            row["train_val_ir_gap"] = None
    else:
        row["val_rankic_ir_mean"] = row["train_val_ir_gap"] = None

    # 最佳迭代次数统计
    if "best_iteration" in group.columns:
        bi = group["best_iteration"].dropna()
        row["best_iter_mean"] = round(bi.mean(), 1) if len(bi) else None
        row["best_iter_min"]  = int(bi.min())       if len(bi) else None
        row["best_iter_max"]  = int(bi.max())       if len(bi) else None
        row["best_iter_std"]  = round(bi.std(), 1)  if len(bi) > 1 else None
    else:
        row["best_iter_mean"] = row["best_iter_min"] = row["best_iter_max"] = row["best_iter_std"] = None

    return row


def build_comparison_table(all_df: pd.DataFrame, raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """构建对比表（行=run，列=聚合指标+训练参数）"""
    if "wf_run_id" not in all_df.columns:
        logger.error("汇总CSV中缺少 wf_run_id 列，无法分组")
        return pd.DataFrame()

    rows = []
    for wf_run_id, group in all_df.groupby("wf_run_id", sort=False):
        # 聚合性能指标
        agg = aggregate_run(group)
        agg.update(load_chain_metrics(raw_dir, wf_run_id))
        agg["wf_run_id"] = wf_run_id

        # 追加训练参数（取第一行即可，同一 run 内所有 split 参数相同）
        first = group.iloc[0]
        for col in PARAM_COLS:
            if col in first.index and col != "wf_run_id":
                agg[col] = first[col]
            elif col != "wf_run_id":
                agg[col] = None

        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    # 列顺序：wf_run_id → 参与评分的指标（按权重降序）→ 非评分指标 → 训练参数
    scored_cols = [col for col, _w, _d in sorted(SCORE_CONFIG, key=lambda x: -x[1])]

    non_scored_metric_cols = [
        "n_splits", "model_version_range",
        # 全周期串联补充
        "chain_total_return", "chain_sharpe", "chain_trading_days",
        # 回测补充
        "bt_annual_return_mean", "bt_calmar_mean", "bt_total_return_mean",
        "bt_max_drawdown_worst", "bt_volatility_mean",
        "bt_signal_confidence_block_rate_mean",
        "bt_signal_confidence_avg_exposure_mean",
        "bt_signal_confidence_avg_score_mean",
        # 统计补充
        "oos_rankic_ir_mean", "oos_rankic_ir_std",
        "oos_top100_median_mean", "oos_top100_win_rate",
        "oos_top300_median_mean", "oos_top300_win_rate",
        # 训练质量补充
        "val_rankic_ir_mean",
        "best_iter_mean", "best_iter_min", "best_iter_max", "best_iter_std",
    ]
    param_cols_ordered = [c for c in PARAM_COLS if c != "wf_run_id"]

    all_cols = ["wf_run_id"] + scored_cols + non_scored_metric_cols + param_cols_ordered
    df = pd.DataFrame(rows)
    # 只保留存在的列，避免 KeyError
    final_cols = [c for c in all_cols if c in df.columns]
    df = df[final_cols]

    df = df.reset_index(drop=True)

    # 列名改为中文
    df = df.rename(columns={k: v for k, v in COL_NAMES.items() if k in df.columns})

    return df


def _build_period_label(row: pd.Series) -> str:
    """优先使用批量脚本传入的时间段标签，否则退回到起止日期。"""
    batch_period_label = row.get(COL_NAMES["batch_period_label"])
    if pd.notna(batch_period_label) and str(batch_period_label).strip():
        return str(batch_period_label)

    wf_start = row.get(COL_NAMES["wf_start_date"])
    wf_end = row.get(COL_NAMES["wf_end_date"])
    if pd.notna(wf_start) and pd.notna(wf_end):
        return f"{wf_start}~{wf_end}"
    return "未标注"


def build_period_stability_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    """按参数组合跨时间段聚合，输出稳定性汇总。"""
    if comp_df.empty:
        return pd.DataFrame()

    varying_cols = {
        COL_NAMES["wf_run_id"],
        COL_NAMES["batch_run_id"],
        COL_NAMES["wf_start_date"],
        COL_NAMES["wf_end_date"],
        COL_NAMES["batch_period_label"],
    }
    metric_cols = {
        "综合得分",
        COL_NAMES["chain_cagr"],
        COL_NAMES["chain_max_drawdown"],
        COL_NAMES["oos_cross_split_ir"],
        COL_NAMES["bt_win_rate"],
    }
    group_cols = []
    for key in PARAM_COLS:
        if key == "wf_run_id":
            continue
        col = COL_NAMES.get(key)
        if col and col in comp_df.columns and col not in varying_cols and col not in metric_cols:
            group_cols.append(col)

    if not group_cols:
        return pd.DataFrame()

    working_df = comp_df.copy()
    working_df["__时间段标签"] = working_df.apply(_build_period_label, axis=1)

    rows = []
    for _, group in working_df.groupby(group_cols, dropna=False, sort=False):
        if len(group) <= 1:
            continue

        score_series = pd.to_numeric(group.get("综合得分"), errors="coerce")
        cagr_series = pd.to_numeric(group.get(COL_NAMES["chain_cagr"]), errors="coerce")
        drawdown_series = pd.to_numeric(group.get(COL_NAMES["chain_max_drawdown"]), errors="coerce")
        ir_series = pd.to_numeric(group.get(COL_NAMES["oos_cross_split_ir"]), errors="coerce")
        win_rate_series = pd.to_numeric(group.get(COL_NAMES["bt_win_rate"]), errors="coerce")

        score_std = score_series.std()
        cagr_std = cagr_series.std()
        ir_std = ir_series.std()

        score_penalty = 0.0 if pd.isna(score_std) else min(max(score_std / 20.0, 0.0), 1.0)
        cagr_penalty = 0.0 if pd.isna(cagr_std) else min(max(cagr_std / 0.2, 0.0), 1.0)
        ir_penalty = 0.0 if pd.isna(ir_std) else min(max(ir_std / 1.0, 0.0), 1.0)
        stability_score = round(
            (1 - (0.4 * score_penalty + 0.3 * cagr_penalty + 0.3 * ir_penalty)) * 100,
            1,
        )

        row = {col: group.iloc[0][col] for col in group_cols}
        if COL_NAMES["batch_run_id"] in group.columns:
            row[COL_NAMES["batch_run_id"]] = group.iloc[0][COL_NAMES["batch_run_id"]]
        row.update(
            {
                COL_NAMES["period_count"]: len(group),
                COL_NAMES["period_labels"]: " | ".join(group["__时间段标签"].astype(str).tolist()),
                COL_NAMES["score_mean"]: round(score_series.mean(), 2) if score_series.notna().any() else None,
                COL_NAMES["score_std"]: round(score_std, 2) if pd.notna(score_std) else None,
                COL_NAMES["score_min"]: round(score_series.min(), 2) if score_series.notna().any() else None,
                COL_NAMES["score_max"]: round(score_series.max(), 2) if score_series.notna().any() else None,
                COL_NAMES["chain_cagr_mean"]: round(cagr_series.mean(), 6) if cagr_series.notna().any() else None,
                COL_NAMES["chain_cagr_std"]: round(cagr_std, 6) if pd.notna(cagr_std) else None,
                COL_NAMES["chain_cagr_min"]: round(cagr_series.min(), 6) if cagr_series.notna().any() else None,
                COL_NAMES["chain_max_drawdown_mean"]: round(drawdown_series.mean(), 6) if drawdown_series.notna().any() else None,
                COL_NAMES["chain_max_drawdown_worst"]: round(drawdown_series.min(), 6) if drawdown_series.notna().any() else None,
                COL_NAMES["oos_cross_split_ir_mean"]: round(ir_series.mean(), 4) if ir_series.notna().any() else None,
                COL_NAMES["oos_cross_split_ir_std"]: round(ir_std, 4) if pd.notna(ir_std) else None,
                COL_NAMES["bt_win_rate_mean"]: round(win_rate_series.mean(), 4) if win_rate_series.notna().any() else None,
                COL_NAMES["bt_win_rate_min"]: round(win_rate_series.min(), 4) if win_rate_series.notna().any() else None,
                COL_NAMES["stability_score"]: max(stability_score, 0.0),
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    ordered_cols = [
        COL_NAMES["batch_run_id"],
        COL_NAMES["period_count"],
        COL_NAMES["period_labels"],
        COL_NAMES["stability_score"],
        COL_NAMES["score_mean"],
        COL_NAMES["score_std"],
        COL_NAMES["score_min"],
        COL_NAMES["score_max"],
        COL_NAMES["chain_cagr_mean"],
        COL_NAMES["chain_cagr_std"],
        COL_NAMES["chain_cagr_min"],
        COL_NAMES["chain_max_drawdown_mean"],
        COL_NAMES["chain_max_drawdown_worst"],
        COL_NAMES["oos_cross_split_ir_mean"],
        COL_NAMES["oos_cross_split_ir_std"],
        COL_NAMES["bt_win_rate_mean"],
        COL_NAMES["bt_win_rate_min"],
    ] + group_cols
    result = pd.DataFrame(rows)
    result = result[[col for col in ordered_cols if col in result.columns]]
    result = result.sort_values(
        [COL_NAMES["stability_score"], COL_NAMES["score_mean"]],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    return result


def compute_composite_score(df: pd.DataFrame) -> pd.Series:
    """计算综合得分（0~100，越高越好）

    方法：对 SCORE_CONFIG 中的每个指标在当前实验集内做百分位排名（0~1），
    按权重加权求和后乘以 100。

    设计原则：
    - 百分位排名（percentile rank）完全回避量纲差异，结果仅反映相对优劣
    - NaN 值视为中性（百分位 0.5），不奖励也不惩罚
    - 单个实验时各指标百分位均为 0.5，得分固定为 50.0
    - ascending=True  时：最大值 → 百分位 1.0（高好）
    - ascending=False 时：最小值 → 百分位 1.0（低好）
    """
    n = len(df)
    weighted_pct = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for eng_key, weight, direction in SCORE_CONFIG:
        col = COL_NAMES.get(eng_key)
        if col is None or col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")

        if direction == "abs_low":
            s = s.abs()
            ascending = False   # 最大绝对值 → rank 1 → 百分位低 → 得分低 ✓
        elif direction == "low":
            ascending = False   # 最大值 → rank 1 → 百分位低 → 得分低 ✓
        else:  # "high"
            ascending = True    # 最小值 → rank 1 → 百分位低 → 大值得高分 ✓

        if n > 0:
            # rank(ascending=True): 最小→1, 最大→n → pct=rank/n
            pct = s.rank(ascending=ascending, method="average", na_option="keep") / n
            pct = pct.fillna(0.5)
        else:
            pct = pd.Series(0.5, index=df.index)

        weighted_pct += weight * pct
        total_weight += weight

    if total_weight > 0:
        score = (weighted_pct / total_weight) * 100
    else:
        score = pd.Series(50.0, index=df.index)

    return score.round(1)


def build_metric_descriptions() -> pd.DataFrame:
    """构建指标说明表（第二个 sheet）"""
    rows = [
        # ── 综合评分 ──────────────────────────────────────────────────────────
        ("综合评分", "综合得分",
         "跨实验百分位排名加权综合得分（0~100，越高越好）。"
         "对12个关键指标分别计算当前实验集内的百分位排名（0~1），再按权重求和×100。"
         "权重配置（回测60%）：全周期CAGR 20%、回测胜率 15%、回测夏普 15%、全周期链式最大回撤 10%；"
         "（统计32%）：跨切分IR 10%、Top30胜率 5%、Top30最差中位收益 5%、"
         "RankIC_IR趋势 5%、Top30中位收益 3%、Top30超额 2%、偏斜度 2%（绝对值低好）；"
         "（训练质量8%）：验证_OOS_IR差距 8%（低好）。"
         "NaN指标以中性分（0.5百分位）计入；仅1组实验时固定得50分。",
         "越高越好"),
        # ── OOS 性能指标 ──────────────────────────────────────────────────────
        ("OOS性能", "运行ID",               "walk-forward运行的唯一标识符，格式为wf_YYYYMMDD_HHMMSS_xxxxxxxx",                                                                              "标识符，无优劣"),
        ("OOS性能", "切分数",               "本次实验成功完成的OOS切分数量，越多统计结论越可靠",                                                                                               "越多越好"),
        ("OOS性能", "模型版本范围",         "本次walk-forward生成的模型编号范围（格式：最小编号~最大编号），可在ModelRegistry中定位具体模型文件",                                                      "参考"),
        ("OOS性能", "OOS_RankIC_IR均值",    "各切分OOS期逐日RankIC信息比率（均值/标准差）的跨切分均值，衡量预测对股票排序的整体有效性",                                                            "越高越好"),
        ("OOS性能", "OOS_RankIC_IR标准差",  "各切分OOS RankIC IR的标准差，衡量策略在不同时间段的稳定性",                                                                                      "越低越稳定"),
        ("OOS性能", "跨切分IR",             "OOS_RankIC_IR均值 / 标准差，类似夏普比率，同时衡量收益水平与跨时间段稳定性，是排序各实验的首要指标",                                                   "越高越好（首要排序指标）"),
        ("OOS性能", "RankIC_IR趋势(近-早)", "最近3个切分的IR均值 - 最早3个切分的IR均值；正值说明模型随时间改善，负值说明alpha在衰减",                                                              "接近0或正值为好，持续负值需警惕"),
        ("OOS性能", "Top30中位收益均值",     "各切分中每日Top30持仓20日收益中位数的跨切分均值；用中位数代替均值，减少极端行情日（如大涨停日）的干扰",                                                  "越高越好"),
        ("OOS性能", "Top30胜率",            "各切分中Top30中位收益>0的占比；衡量策略在不同历史时段的正收益稳健性，>70%可认为优秀",                                                               "越高越好（>0.7为优秀）"),
        ("OOS性能", "Top30最差中位收益",     "所有切分中Top30中位收益的最小值，代表策略的最差历史表现，用于压力测试",                                                                               "越高越好（大幅负值需警惕）"),
        ("OOS性能", "Top30偏斜度均值",       "各切分中(Top30均值-中位数)/标准差的均值；偏斜度高说明均值被少数极端行情日拉偏，均值的代表性下降",                                                      "越接近0越好（>0.6需警惕）"),
        ("OOS性能", "Top30超额均值",         "各切分中Top30相对全市场平均收益的超额均值，衡量纯选股能力（剔除市场整体涨跌的影响）",                                                                  "越高越好"),
        ("OOS性能", "Top100中位收益均值",    "各切分每日Top100持仓20日收益中位数的跨切分均值",                                                                                                 "越高越好"),
        ("OOS性能", "Top100胜率",           "各切分中Top100中位收益>0的占比",                                                                                                              "越高越好"),
        ("OOS性能", "Top300中位收益均值",    "各切分每日Top300持仓20日收益中位数的跨切分均值；样本量大，统计更稳定但个股alpha被稀释",                                                               "越高越好"),
        ("OOS性能", "Top300胜率",           "各切分中Top300中位收益>0的占比",                                                                                                              "越高越好"),
        # ── OOS 回测指标 ──────────────────────────────────────────────────────
        ("OOS回测", "全周期CAGR",            "基于 chain_nav 将所有 split 的 OOS 净值顺次串联后得到的全周期复利年化收益，更适合做最终策略筛选",                                                       "越高越好（优先看）"),
        ("OOS回测", "全周期总收益",          "基于 chain_nav 串联后的起止总收益率，反映整轮 walk-forward 结束时的真实累计收益",                                                                      "越高越好"),
        ("OOS回测", "全周期链式最大回撤",    "基于 chain_nav 串联后的全周期最大回撤，避免只看单个 split 的最差值而忽略跨 split 累计损失",                                                              "越接近0越好（优先看）"),
        ("OOS回测", "全周期链式夏普",        "基于 chain_nav 串联后的全周期夏普比率，使用 3% 无风险利率与全周期日收益波动计算",                                                                      "越高越好"),
        ("OOS回测", "全周期链式交易日数",    "chain_nav 串联后的交易日总数，用于判断 CAGR 计算口径和样本长度是否一致",                                                                            "参考"),
        ("OOS回测", "回测年化收益均值",       "各切分 OOS 回测年化收益率的跨切分均值，适合做横向粗筛，但不等同于全周期 CAGR",                                                                      "越高越好（粗筛用）"),
        ("OOS回测", "回测夏普均值",          "各切分OOS回测夏普比率（年化收益-3%无风险利率/年化波动率）的跨切分均值",                                                                                "越高越好（>1.0为优秀）"),
        ("OOS回测", "回测最大回撤(最差)",     "所有切分 OOS 回测中最大回撤的最差值（绝对值最大的单段回撤），用于观察最脆弱 split，但不等同于全周期链式回撤",                                        "越接近0越好（-30%以下需警惕）"),
        ("OOS回测", "回测Calmar均值",        "各切分年化收益/最大回撤的均值，衡量单位风险回报",                                                                                                   "越高越好（>1.0为良好）"),
        ("OOS回测", "回测胜率",              "各切分 OOS 回测总收益>0 的占比，衡量策略在不同历史时段的盈利稳健性",                                                                                 "越高越好（>0.7为优秀）"),
        ("OOS回测", "回测总收益均值",         "各切分 OOS 回测期间总收益率的跨切分均值，适合观察 split 层面的平均水平，不等同于全周期累计收益",                                                      "越高越好"),
        ("OOS回测", "回测波动率均值",         "各切分OOS回测年化波动率的跨切分均值",                                                                                                             "越低越稳定"),
        # ── 训练质量指标 ──────────────────────────────────────────────────────
        ("训练质量", "验证集RankIC_IR均值",  "各切分内部验证集逐日RankIC IR的跨切分均值；验证集来自训练窗口末尾，反映模型在样本内末期的泛化能力",                                                      "越高越好"),
        ("训练质量", "验证_OOS_IR差距",      "验证集IR均值 - OOS IR均值；正值表示验证集优于OOS（轻度过拟合信号），负值表示OOS优于验证集（通常正常）",                                               "接近0为好，负值可接受，大正值（>0.5）需警惕过拟合"),
        ("训练质量", "最佳迭代均值",         "各切分早停触发时的迭代次数均值；反映模型实际使用的树数量，可指导n_estimators上限的设置",                                                               "参考指标（<100可能欠拟合，接近n_estimators上限则建议增大）"),
        ("训练质量", "最佳迭代最小值",       "所有切分中最少的早停迭代次数；如果某个切分迭代极少，说明该时段数据可能有异常",                                                                          "不宜过低（<100需关注）"),
        ("训练质量", "最佳迭代最大值",       "所有切分中最多的早停迭代次数",                                                                                                                   "不宜接近n_estimators（说明需增大树数量上限）"),
        ("训练质量", "最佳迭代标准差",       "各切分迭代次数的标准差；衡量模型在不同时间段需要的学习量是否一致，差异过大说明数据分布在各时段差异明显",                                                    "越低越稳定"),
        # ── 训练参数（仅供参考） ───────────────────────────────────────────────
        ("训练参数", "WF起始日期",           "walk-forward整体时间范围起始日期，训练集不早于此日期",                                                                                             "参考"),
        ("训练参数", "WF结束日期",           "walk-forward整体时间范围结束日期，训练集截止于此，测试集可超出此范围",                                                                               "参考"),
        ("训练参数", "滚动频率",             "每次train_end向前推进的步长（monthly=月度 / quarterly=季度 / semiannual=半年度）",                                                               "参考"),
        ("训练参数", "训练窗口年数",         "每次训练使用的历史数据年数；过短欠拟合，过长可能纳入失效的历史规律",                                                                                   "参考（通常3~7年）"),
        ("训练参数", "测试窗口月数",         "每次OOS评估的时间窗口（月数），建议与标签持仓周期相近",                                                                                             "参考"),
        ("训练参数", "验证集比例",           "训练数据中用于内部早停评估的比例",                                                                                                               "参考"),
        ("训练参数", "标签列",               "预测目标列名，如neu_y_ret_20（中性化20日收益）；neu_前缀表示已剔除市值/行业因子",                                                                    "参考"),
        ("训练参数", "任务类型",             "regression=回归（预测收益率大小）/ classification=分类（预测涨跌方向）",                                                                           "参考"),
        ("训练参数", "标签变换",             "raw=使用原始收益率标签 / cs_zscore=截面z-score标准化（消除截面异方差，让模型聚焦排序）",                                                             "参考"),
        ("训练参数", "树数量",               "XGBoost决策树总数上限；配合早停使用，实际用量为最佳迭代次数",                                                                                       "参考"),
        ("训练参数", "最大深度",             "每棵树的最大层数；越大模型越复杂越容易过拟合",                                                                                                     "参考（建议6~10）"),
        ("训练参数", "学习率",               "梯度下降步长；越小越精细但需要更多树才能收敛",                                                                                                     "参考"),
        ("训练参数", "样本采样比",           "每棵树随机抽取的样本比例（行采样），降低过拟合风险",                                                                                                 "参考"),
        ("训练参数", "特征采样比",           "每棵树随机抽取的特征比例（列采样），降低过拟合风险",                                                                                                 "参考"),
        ("训练参数", "最小叶节点权重",       "叶节点所需的最少样本权重之和；越大越保守，防止模型学习噪声",                                                                                          "参考"),
        ("训练参数", "gamma",               "节点分裂所需的最小损失下降量；越大越保守，减少无效分裂",                                                                                             "参考"),
        ("训练参数", "L1正则",               "L1正则化系数，使特征权重趋向稀疏（部分特征权重归零）",                                                                                              "参考"),
        ("训练参数", "L2正则",               "L2正则化系数，使特征权重趋向平滑（防止某个特征权重过大）",                                                                                          "参考"),
        ("训练参数", "rank权重启用",         "是否对每日Top/Bottom K样本赋予更高训练权重，使模型更关注极端收益样本",                                                                               "参考"),
        ("训练参数", "rank权重TopK",         "每日增强权重覆盖的头部/尾部股票数量",                                                                                                             "参考"),
        ("训练参数", "rank权重值",           "增强样本相对普通样本的权重倍数",                                                                                                                 "参考"),
    ]
    return pd.DataFrame(rows, columns=["分类", "指标名", "说明", "优劣方向"])


def build_split_detail_table(all_df: pd.DataFrame) -> pd.DataFrame:
    """构建逐 split 明细表（每个 split 一行，展示该 split 的回测表现与训练参数）

    用于对比不同实验在各时间段的表现差异，帮助定位哪些时段拖后腿。
    """
    if all_df.empty or "wf_run_id" not in all_df.columns:
        return pd.DataFrame()

    # 选取需要的列
    detail_cols = [
        "wf_run_id", "split_index",
        "train_start", "train_end", "test_start", "test_end",
        # 逐 split 回测指标
        "bt_total_return", "bt_annual_return", "bt_max_drawdown",
        "bt_sharpe", "bt_calmar", "bt_volatility",
        "bt_trading_days", "bt_top_n",
        # 逐 split 模型质量
        "daily_rankic_mean", "daily_rankic_ir", "val_rankic_ir",
        "best_iteration", "train_samples", "test_samples",
        # 关键训练参数（用于区分实验）
        "algorithm", "label_column", "task", "label_transform",
        "max_depth", "learning_rate", "train_window_years", "test_window_months",
        "time_decay_half_life", "market_regime", "enable_fundamental",
    ]
    available_cols = [c for c in detail_cols if c in all_df.columns]
    result = all_df[available_cols].copy()

    # 按运行时间降序、split 升序排列
    result = result.sort_values(["wf_run_id", "split_index"], ascending=[False, True])

    # 计算累计净值（组内 cumprod）
    if "bt_total_return" in result.columns:
        result["chain_nav"] = result.groupby("wf_run_id")["bt_total_return"].transform(
            lambda x: (1 + x).cumprod()
        )

    # 列名翻译
    rename_map = {
        "wf_run_id": "运行ID",
        "split_index": "切分序号",
        "train_start": "训练开始",
        "train_end": "训练结束",
        "test_start": "测试开始",
        "test_end": "测试结束",
        "bt_total_return": "总收益",
        "bt_annual_return": "年化收益",
        "bt_max_drawdown": "最大回撤",
        "bt_sharpe": "夏普",
        "bt_calmar": "Calmar",
        "bt_volatility": "波动率",
        "bt_trading_days": "交易天数",
        "bt_top_n": "TopN",
        "daily_rankic_mean": "OOS_RankIC均值",
        "daily_rankic_ir": "OOS_RankIC_IR",
        "val_rankic_ir": "验证集RankIC_IR",
        "best_iteration": "最佳迭代",
        "train_samples": "训练样本数",
        "test_samples": "测试样本数",
        "chain_nav": "累计净值",
        "algorithm": "算法",
        "label_column": "标签列",
        "task": "任务类型",
        "label_transform": "标签变换",
        "max_depth": "最大深度",
        "learning_rate": "学习率",
        "train_window_years": "训练窗口年数",
        "test_window_months": "测试窗口月数",
        "time_decay_half_life": "时间衰减半衰期",
        "market_regime": "市场择时",
        "enable_fundamental": "基本面因子",
    }
    result = result.rename(columns={k: v for k, v in rename_map.items() if k in result.columns})

    # 调整列序：运行ID → 切分序号 → 时间 → 回测指标 → 累计净值 → 模型质量 → 训练参数
    ordered = [
        "运行ID", "切分序号",
        "训练开始", "训练结束", "测试开始", "测试结束",
        "总收益", "年化收益", "最大回撤", "夏普", "Calmar", "波动率", "交易天数", "TopN",
        "累计净值",
        "OOS_RankIC均值", "OOS_RankIC_IR", "验证集RankIC_IR", "最佳迭代",
        "训练样本数", "测试样本数",
        "算法", "标签列", "任务类型", "标签变换",
        "最大深度", "学习率", "训练窗口年数", "测试窗口月数",
        "时间衰减半衰期", "市场择时", "基本面因子",
    ]
    final_cols = [c for c in ordered if c in result.columns]
    return result[final_cols].reset_index(drop=True)


def sort_by_run_time(df: pd.DataFrame, run_id_col: str = "运行ID") -> pd.DataFrame:
    """按 wf_run_id 中的运行时间戳降序排列（最近的排在前面）

    wf_run_id 格式: wf_YYYYMMDD_HHMMSS_xxxxxxxx
    提取 YYYYMMDD_HHMMSS 作为排序键。
    """
    if run_id_col not in df.columns:
        return df

    def _extract_dt(run_id: str) -> str:
        parts = str(run_id).split("_")
        # parts: ['wf', 'YYYYMMDD', 'HHMMSS', 'xxxxxxxx']
        if len(parts) >= 3:
            return parts[1] + parts[2]  # YYYYMMDDHHMMSS
        return run_id

    df = df.copy()
    df["_sort_key"] = df[run_id_col].map(_extract_dt)
    df = df.sort_values("_sort_key", ascending=False, na_position="last")
    df = df.drop(columns=["_sort_key"]).reset_index(drop=True)
    return df


def _str_display_width(s: str) -> int:
    """计算字符串的显示宽度（CJK 字符计为 2，ASCII 计为 1）"""
    width = 0
    for ch in str(s):
        eaw = unicodedata.east_asian_width(ch)
        width += 2 if eaw in ("W", "F") else 1
    return width


def _weight_to_green_fill(weight: float, max_weight: float) -> PatternFill:
    """将权重映射为浅绿→中绿的填充色（权重越高越深，黑字始终可读）

    颜色范围：
      最小权重 → RGB(220, 245, 220)  极浅绿
      最大权重 → RGB(130, 215, 130)  中等绿（与黑字对比度 ≈ 8:1，远超 WCAG AA 4.5:1）
    """
    t = min(weight / max_weight, 1.0) if max_weight > 0 else 0.0
    r = int(round(220 - t * 90))   # 220 → 130
    g = int(round(245 - t * 30))   # 245 → 215
    b = int(round(220 - t * 90))   # 220 → 130
    return PatternFill(fill_type="solid", fgColor=f"{r:02X}{g:02X}{b:02X}")


def format_excel_output(wb, desc_df: pd.DataFrame) -> None:
    """对 Excel 工作簿应用格式化

    - 全局字体: 微软雅黑 9 号
    - 冻结标题行（两个 sheet 均适用）
    - 实验对比标题行超链接跳转至指标说明对应行
    - 自动列宽（CJK 双倍宽）
    - 参与综合得分的列着浅绿背景，权重越高越深
    """
    # 构建 指标名 → 指标说明 sheet 行号的映射（第 1 行为标题，数据从第 2 行起）
    desc_row_map: dict[str, int] = {}
    for i, row in desc_df.iterrows():
        desc_row_map[str(row["指标名"])] = int(i) + 2  # +2: header占第1行，数据从第2行

    font_normal = Font(name="微软雅黑", size=9)
    font_link   = Font(name="微软雅黑", size=9, color="0563C1", underline="single")

    # ── 构建 中文列名 → 绿色填充 的映射（用于"实验对比"sheet）────────────
    _max_w = max(w for _, w, _ in SCORE_CONFIG) if SCORE_CONFIG else 1.0
    # 综合得分列本身也着色，用最深绿（权重等同最大权重）
    score_cn_fills: dict[str, PatternFill] = {
        "综合得分": _weight_to_green_fill(_max_w, _max_w),
    }
    for eng_key, weight, _ in SCORE_CONFIG:
        col_cn = COL_NAMES.get(eng_key)
        if col_cn:
            score_cn_fills[col_cn] = _weight_to_green_fill(weight, _max_w)

    # 扫描"实验对比"标题行，建立 列字母 → 填充色 的映射
    ws_comp_ref = wb["实验对比"]
    col_letter_fill: dict[str, PatternFill] = {}
    for cell in next(ws_comp_ref.iter_rows(min_row=1, max_row=1)):
        if cell.value and str(cell.value) in score_cn_fills:
            col_letter_fill[cell.column_letter] = score_cn_fills[str(cell.value)]

    # ── 全局字体、冻结、列宽、绿色背景 ──────────────────────────────────
    all_sheets = [s for s in ["实验对比", "跨时间段稳定性", "指标说明", "逐Split明细"] if s in wb.sheetnames]
    for sheet_name in all_sheets:
        ws = wb[sheet_name]
        ws.freeze_panes = "A2"

        col_widths: dict[str, int] = {}
        for row in ws.iter_rows():
            for cell in row:
                cell.font = font_normal
                # 仅对"实验对比"sheet 的参与评分列着绿色背景
                if sheet_name == "实验对比" and cell.column_letter in col_letter_fill:
                    cell.fill = col_letter_fill[cell.column_letter]
                if cell.value is not None:
                    w = _str_display_width(str(cell.value))
                    col_letter = cell.column_letter
                    col_widths[col_letter] = max(col_widths.get(col_letter, 0), w)

        for col_letter, w in col_widths.items():
            ws.column_dimensions[col_letter].width = min(w + 2, 60)  # 最多60宽，留2字符边距

    # ── 实验对比标题行超链接（内部链接须用 Hyperlink(location=...)）───────
    ws_comp = wb["实验对比"]
    for cell in next(ws_comp.iter_rows(min_row=1, max_row=1)):
        metric_name = str(cell.value) if cell.value else ""
        if metric_name in desc_row_map:
            target_row = desc_row_map[metric_name]
            cell.hyperlink = Hyperlink(
                ref=cell.coordinate,
                location=f"'指标说明'!A{target_row}",
            )
            cell.font = font_link


def print_comparison_table(df: pd.DataFrame) -> None:
    """控制台打印可读的对比表（精简版）"""
    if df.empty:
        logger.info("对比表为空")
        return

    display_cols = [
        COL_NAMES["wf_run_id"],
        "综合得分",
        COL_NAMES["n_splits"],
        COL_NAMES["model_version_range"],
        COL_NAMES["chain_cagr"],
        COL_NAMES["chain_max_drawdown"],
        COL_NAMES["chain_total_return"],
        COL_NAMES["oos_cross_split_ir"],
        COL_NAMES["oos_rankic_ir_mean"],
        COL_NAMES["oos_top30_win_rate"],
        COL_NAMES["oos_top30_median_mean"],
        COL_NAMES["oos_top30_worst_median"],
        COL_NAMES["oos_top30_lift_mean"],
        COL_NAMES["bt_annual_return_mean"],
        COL_NAMES["bt_sharpe_mean"],
        COL_NAMES["bt_max_drawdown_worst"],
        COL_NAMES["bt_win_rate"],
        COL_NAMES["val_rankic_ir_mean"],
        COL_NAMES["train_val_ir_gap"],
        COL_NAMES["best_iter_mean"],
        COL_NAMES["label_column"],
        COL_NAMES["task"],
        COL_NAMES["n_estimators"],
        COL_NAMES["max_depth"],
        COL_NAMES["learning_rate"],
    ]
    show_cols = [c for c in display_cols if c in df.columns]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    logger.info("\n" + df[show_cols].to_string(index=True))


def main():
    parser = argparse.ArgumentParser(description="Walk-forward 实验对比分析")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录；未指定时使用 configs/base.yaml 中的 data.root",
    )
    parser.add_argument("--raw-dir",   type=str, default=None,     help="walk_forward 汇总CSV目录，默认 {data_root}/walk_forward/raw")
    parser.add_argument("--output",    type=str, default=None,     help="对比Excel输出路径，默认 {data_root}/walk_forward/wf_comparison.xlsx")
    args = parser.parse_args()

    setup_logger()

    effective_data_root = Path(args.data_root or get_data_root())
    raw_dir     = Path(args.raw_dir) if args.raw_dir else effective_data_root / "walk_forward" / "raw"
    output_path = Path(args.output)  if args.output  else effective_data_root / "walk_forward" / "wf_comparison.xlsx"

    logger.info("=" * 70)
    logger.info("Walk-forward 实验对比分析")
    logger.info("=" * 70)
    logger.info(f"汇总CSV目录: {raw_dir}")
    logger.info(f"输出路径:     {output_path}")

    # 1. 加载所有汇总CSV
    all_df = load_all_summaries(raw_dir)
    if all_df.empty:
        logger.error("没有可用数据，退出")
        return

    # 2. 构建对比表
    comp_df = build_comparison_table(all_df, raw_dir=raw_dir)
    if comp_df.empty:
        logger.error("构建对比表失败，退出")
        return

    # 3. 计算综合得分并插入为第二列（运行ID 之后）
    comp_df.insert(1, "综合得分", compute_composite_score(comp_df))
    logger.info(f"综合得分计算完成（参与评分指标数: {sum(1 for k,_,_ in SCORE_CONFIG if COL_NAMES.get(k) in comp_df.columns)}）")

    # 4. 构建指标说明表
    desc_df = build_metric_descriptions()

    # 5. 构建逐 split 明细表
    split_df = build_split_detail_table(all_df)
    logger.info(f"逐Split明细表: {len(split_df)} 行")

    # 6. 构建跨时间段稳定性表
    period_stability_df = build_period_stability_table(comp_df)
    logger.info(f"跨时间段稳定性表: {len(period_stability_df)} 行")

    # 7. 按运行时间降序排列（最近的排在最前面）
    comp_df = sort_by_run_time(comp_df)

    # 8. 输出 Excel（三到四个 sheet），并应用格式化
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        comp_df.to_excel(writer, sheet_name="实验对比", index=False)
        if not period_stability_df.empty:
            period_stability_df.to_excel(writer, sheet_name="跨时间段稳定性", index=False)
        desc_df.to_excel(writer, sheet_name="指标说明", index=False)
        if not split_df.empty:
            split_df.to_excel(writer, sheet_name="逐Split明细", index=False)
        format_excel_output(writer.book, desc_df)
    logger.info(f"对比表已保存: {output_path}（{len(comp_df)} 个实验，共{len(writer.book.sheetnames)}个sheet）")

    # 7. 控制台打印精简版
    #print_comparison_table(comp_df)

    logger.info("=" * 70)


if __name__ == "__main__":
    main()

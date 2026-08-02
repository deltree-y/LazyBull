# -*- coding: utf-8 -*-
"""逐Split明细、排序、列重排与紧凑展示。"""

import unicodedata

import pandas as pd

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.compare.constants import (
    BATCH_EXPERIMENT_CORE_COLS,
    BATCH_EXPERIMENT_EXCLUDED_PARAM_COLS,
    BATCH_EXPERIMENT_PARAM_CANDIDATES,
)
from scripts.compare.aggregate import sort_by_latest_run_time


def build_split_detail_table(all_df: pd.DataFrame) -> pd.DataFrame:
    """构建逐 split 明细表（每个 split 一行，展示该 split 的回测表现与训练参数）

    用于对比不同实验在各时间段的表现差异，帮助定位哪些时段拖后腿。
    """
    if all_df.empty or "wf_run_id" not in all_df.columns:
        return pd.DataFrame()

    # 选取需要的列
    detail_cols = [
        "wf_run_id",
        "split_index",
        "KEY_Top20_list",
        "KEY_Top30_list",
        "KEY_Top20_hit_rate",
        "KEY_Top20_avg_return_median",
        "KEY_Top30_hit_rate",
        "KEY_Top30_avg_return_median",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        # 逐 split 回测指标
        "bt_total_return",
        "bt_annual_return",
        "bt_max_drawdown",
        "bt_sharpe",
        "bt_calmar",
        "bt_volatility",
        "bt_trading_days",
        "bt_top_n",
        "bt_rebalance_freq",
        # 逐 split 模型质量
        "daily_rankic_mean",
        "daily_rankic_ir",
        "val_rankic_ir",
        "best_iteration",
        "train_samples",
        "test_samples",
        # 关键训练参数（用于区分实验）
        "algorithm",
        "label_column",
        "neutral_label_blend_weight",
        "task",
        "label_transform",
        "max_depth",
        "learning_rate",
        "train_window_years",
        "test_window_months",
        "time_decay_half_life",
        "market_regime",
        "enable_fundamental",
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
        "KEY_Top20_list": "重点Top20名单",
        "KEY_Top30_list": "重点Top30名单",
        "KEY_Top20_hit_rate": "重点Top20命中率",
        "KEY_Top20_avg_return_median": "重点Top20收益中位数",
        "KEY_Top30_hit_rate": "重点Top30命中率",
        "KEY_Top30_avg_return_median": "重点Top30收益中位数",
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
        "bt_rebalance_freq": "调仓频率",
        "daily_rankic_mean": "OOS_RankIC均值",
        "daily_rankic_ir": "OOS_RankIC_IR",
        "val_rankic_ir": "验证集RankIC_IR",
        "best_iteration": "最佳迭代",
        "train_samples": "训练样本数",
        "test_samples": "测试样本数",
        "chain_nav": "累计净值",
        "algorithm": "算法",
        "label_column": "标签列",
        "neutral_label_blend_weight": "中性标签混合权重",
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
        "运行ID",
        "切分序号",
        "重点Top20命中率",
        "重点Top20收益中位数",
        "重点Top30命中率",
        "重点Top30收益中位数",
        "重点Top20名单",
        "重点Top30名单",
        "训练开始",
        "训练结束",
        "测试开始",
        "测试结束",
        "总收益",
        "年化收益",
        "最大回撤",
        "夏普",
        "Calmar",
        "波动率",
        "交易天数",
        "TopN",
        "累计净值",
        "OOS_RankIC均值",
        "OOS_RankIC_IR",
        "验证集RankIC_IR",
        "最佳迭代",
        "训练样本数",
        "测试样本数",
        "算法",
        "标签列",
        "任务类型",
        "标签变换",
        "最大深度",
        "学习率",
        "训练窗口年数",
        "测试窗口月数",
        "时间衰减半衰期",
        "市场择时",
        "基本面因子",
    ]
    final_cols = [c for c in ordered if c in result.columns]
    return result[final_cols].reset_index(drop=True)


def sort_by_run_time(df: pd.DataFrame, run_id_col: str = "运行ID") -> pd.DataFrame:
    """按 wf_run_id 中的运行时间戳降序排列（最近的排在前面）

    wf_run_id 格式: wf_YYYYMMDD_HHMMSS_xxxxxxxx
    提取 YYYYMMDD_HHMMSS 作为排序键。
    """
    return sort_by_latest_run_time(df, run_id_col, ["综合得分"])


def reorder_comparison_columns(df: pd.DataFrame) -> pd.DataFrame:
    """按汇总阅读习惯重排实验对比列顺序。"""
    preferred_order = [
        "运行ID",
        "最新运行时间",
        "综合得分",
        "选股综合得分",
        "RankIC均值",
        "ICIR",
        "Top30超额均值",
        "最大深度",
        "学习率",
        "rank权重TopK",
        "rank权重值",
    ]

    tail_order = [
    ]

    ordered_cols = [col for col in preferred_order if col in df.columns]

    # 其余列保持原顺序，避免非目标列发生意外位移。
    for col in df.columns:
        if col not in ordered_cols and col not in tail_order:
            ordered_cols.append(col)

    ordered_cols.extend([col for col in tail_order if col in df.columns])
    return df.reindex(columns=ordered_cols)


def _series_has_meaningful_variation(series: pd.Series) -> bool:
    """判断列是否存在实际区分度，纯常量或仅空值视为无区分度。"""
    if series is None:
        return False
    non_na = series.dropna()
    if non_na.empty:
        return False
    return non_na.astype(str).nunique() > 1


def compact_experiment_sheet_for_display(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """压缩 batches 实验对比页，只保留核心指标和有区分度的关键参数。"""
    if df.empty or source_label != "batches":
        return df

    ordered: list[str] = []
    for col in BATCH_EXPERIMENT_CORE_COLS:
        if col in df.columns and col not in ordered:
            ordered.append(col)

    for col in BATCH_EXPERIMENT_PARAM_CANDIDATES:
        if (
            col in df.columns
            and col not in ordered
            and col not in BATCH_EXPERIMENT_EXCLUDED_PARAM_COLS
            and _series_has_meaningful_variation(df[col])
        ):
            ordered.append(col)

    return df[[col for col in ordered if col in df.columns]].copy()


def _str_display_width(s: str) -> int:
    """计算字符串的显示宽度（CJK 字符计为 2，ASCII 计为 1）"""
    width = 0
    for ch in str(s):
        eaw = unicodedata.east_asian_width(ch)
        width += 2 if eaw in ("W", "F") else 1
    return width

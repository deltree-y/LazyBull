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
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger

from src.lazybull.common.logger import setup_logger


# ---------------------------------------------------------------------------
# 训练参数列（来自 write_walk_forward_summary 写入的列名，取每组第一行即可）
# ---------------------------------------------------------------------------
PARAM_COLS = [
    "wf_run_id",
    "wf_start_date", "wf_end_date", "step",
    "train_window_years", "test_window_months", "val_ratio",
    "label_column", "task", "label_transform",
    "n_estimators", "max_depth", "learning_rate",
    "subsample", "colsample_bytree", "min_child_weight",
    "gamma", "reg_alpha", "reg_lambda",
    "rank_weight_enabled", "rank_weight_topk", "rank_weight",
]


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
            df = pd.read_csv(f, encoding="utf-8-sig")
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


def build_comparison_table(all_df: pd.DataFrame) -> pd.DataFrame:
    """构建对比表（行=run，列=聚合指标+训练参数）"""
    if "wf_run_id" not in all_df.columns:
        logger.error("汇总CSV中缺少 wf_run_id 列，无法分组")
        return pd.DataFrame()

    rows = []
    for wf_run_id, group in all_df.groupby("wf_run_id", sort=False):
        # 聚合性能指标
        agg = aggregate_run(group)
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

    # 列顺序：wf_run_id → OOS 指标 → 训练质量指标 → 训练参数
    oos_cols = [
        "n_splits",
        "oos_rankic_ir_mean", "oos_rankic_ir_std", "oos_cross_split_ir", "oos_rankic_ir_trend",
        "oos_top30_median_mean", "oos_top30_win_rate", "oos_top30_worst_median",
        "oos_top30_skew_score_mean", "oos_top30_lift_mean",
        "oos_top100_median_mean", "oos_top100_win_rate",
        "oos_top300_median_mean", "oos_top300_win_rate",
    ]
    quality_cols = [
        "val_rankic_ir_mean", "train_val_ir_gap",
        "best_iter_mean", "best_iter_min", "best_iter_max", "best_iter_std",
    ]
    param_cols_ordered = [c for c in PARAM_COLS if c != "wf_run_id"]

    all_cols = ["wf_run_id"] + oos_cols + quality_cols + param_cols_ordered
    df = pd.DataFrame(rows)
    # 只保留存在的列，避免 KeyError
    final_cols = [c for c in all_cols if c in df.columns]
    df = df[final_cols]

    # 按 oos_cross_split_ir 降序排（越高越稳健）
    if "oos_cross_split_ir" in df.columns:
        df = df.sort_values("oos_cross_split_ir", ascending=False, na_position="last")

    return df.reset_index(drop=True)


def print_comparison_table(df: pd.DataFrame) -> None:
    """控制台打印可读的对比表（精简版）"""
    if df.empty:
        logger.info("对比表为空")
        return

    display_cols = [
        "wf_run_id",
        "n_splits",
        "oos_cross_split_ir",
        "oos_rankic_ir_mean",
        "oos_top30_win_rate",
        "oos_top30_median_mean",
        "oos_top30_worst_median",
        "oos_top30_lift_mean",
        "val_rankic_ir_mean",
        "train_val_ir_gap",
        "best_iter_mean",
        "label_column",
        "task",
        "n_estimators",
        "max_depth",
        "learning_rate",
    ]
    show_cols = [c for c in display_cols if c in df.columns]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    logger.info("\n" + df[show_cols].to_string(index=True))


def main():
    parser = argparse.ArgumentParser(description="Walk-forward 实验对比分析")
    parser.add_argument("--data-root", type=str, default="./data", help="数据根目录，默认 ./data")
    parser.add_argument("--raw-dir",   type=str, default=None,     help="walk_forward 汇总CSV目录，默认 {data_root}/walk_forward/raw")
    parser.add_argument("--output",    type=str, default=None,     help="对比CSV输出路径，默认 {data_root}/walk_forward/wf_comparison.csv")
    args = parser.parse_args()

    setup_logger()

    raw_dir    = Path(args.raw_dir)    if args.raw_dir else Path(args.data_root) / "walk_forward" / "raw"
    output_path = Path(args.output)   if args.output  else Path(args.data_root) / "walk_forward" / "wf_comparison.csv"

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
    comp_df = build_comparison_table(all_df)
    if comp_df.empty:
        logger.error("构建对比表失败，退出")
        return

    # 3. 输出CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"对比表已保存: {output_path}（{len(comp_df)} 个实验）")

    # 4. 控制台打印精简版
    print_comparison_table(comp_df)

    logger.info("=" * 70)


if __name__ == "__main__":
    main()

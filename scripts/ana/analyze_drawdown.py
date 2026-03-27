#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回撤归因分析脚本

功能：
- 基于 walk-forward 的 summary + chain_nav 数据，对指定时段的回撤进行多维归因
- 分析维度：split级别对比、信号质量、市场环境、净值回撤详情
- 输出：CSV数据 + matplotlib图表 + 文字归因报告

使用示例：
    python scripts/ana/analyze_drawdown.py --wf-run-id wf_20260325_120126_f641fee9
    python scripts/ana/analyze_drawdown.py --wf-run-id wf_20260325_120126_f641fee9 --focus-splits 9,10
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.common.config import Config
from src.lazybull.common.logger import setup_logger
from src.lazybull.data.storage import Storage

# ── matplotlib 中文配置 ──────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ── 数据加载 ────────────────────────────────────────────────────


def load_wf_data(
    wf_run_id: str, data_root: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载 walk-forward 的 summary 和 chain_nav 数据

    Args:
        wf_run_id: walk-forward 运行ID
        data_root: 数据根目录

    Returns:
        (summary_df, chain_nav_df)
    """
    raw_dir = data_root / "walk_forward" / "raw"
    summary_path = raw_dir / f"walk_forward_summary_{wf_run_id}.csv"
    chain_path = raw_dir / f"chain_nav_{wf_run_id}.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"找不到 summary 文件: {summary_path}")
    if not chain_path.exists():
        raise FileNotFoundError(f"找不到 chain_nav 文件: {chain_path}")

    summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
    chain_nav_df = pd.read_csv(chain_path, encoding="utf-8-sig")

    logger.info(f"已加载 summary: {len(summary_df)} splits, chain_nav: {len(chain_nav_df)} 行")
    return summary_df, chain_nav_df


def build_trade_dates(
    summary_df: pd.DataFrame, chain_nav_df: pd.DataFrame, storage: Storage
) -> pd.Series:
    """将 chain_nav 的行号映射为真实交易日期

    通过交易日历，根据每个split的test_start/test_end推算每行对应的真实日期。

    Args:
        summary_df: walk-forward summary
        chain_nav_df: chain_nav数据
        storage: Storage实例用于读取交易日历

    Returns:
        与 chain_nav_df 等长的日期 Series
    """
    # 读取交易日历
    trade_cal_df = storage.load_raw("trade_cal")
    if trade_cal_df is None:
        logger.warning("无法读取交易日历，降级为按split均匀分配日期")
        return _fallback_dates(summary_df, chain_nav_df)

    # 提取开市日期
    if "is_open" in trade_cal_df.columns:
        open_dates = trade_cal_df[trade_cal_df["is_open"] == 1]["cal_date"].values
    else:
        open_dates = trade_cal_df["cal_date"].values
    open_dates = sorted([str(d) for d in open_dates])

    # 为每个split分配真实日期
    dates = []
    for _, row in summary_df.iterrows():
        split_idx = int(row["split_index"])
        test_start = str(int(row["test_start"]))
        test_end = str(int(row["test_end"]))

        # 获取该split区间内的交易日
        split_dates = [d for d in open_dates if test_start <= d <= test_end]

        # 获取chain_nav中该split的行数
        split_nav = chain_nav_df[chain_nav_df["split_index"] == split_idx]
        n_rows = len(split_nav)

        if len(split_dates) >= n_rows:
            dates.extend(split_dates[:n_rows])
        else:
            # 交易日不够，用已有的补齐
            dates.extend(split_dates)
            dates.extend([split_dates[-1]] * (n_rows - len(split_dates)))

    if len(dates) != len(chain_nav_df):
        logger.warning(f"日期数({len(dates)})与chain_nav行数({len(chain_nav_df)})不匹配，降级处理")
        return _fallback_dates(summary_df, chain_nav_df)

    return pd.Series(dates, name="trade_date")


def _fallback_dates(
    summary_df: pd.DataFrame, chain_nav_df: pd.DataFrame
) -> pd.Series:
    """降级方案：用split的test_start/test_end生成近似日期"""
    dates = []
    for _, row in summary_df.iterrows():
        split_idx = int(row["split_index"])
        test_start = str(int(row["test_start"]))
        test_end = str(int(row["test_end"]))
        n_rows = len(chain_nav_df[chain_nav_df["split_index"] == split_idx])
        # 生成等间隔日期
        start = pd.Timestamp(test_start)
        end = pd.Timestamp(test_end)
        split_dates = pd.date_range(start, end, periods=n_rows).strftime("%Y%m%d").tolist()
        dates.extend(split_dates)
    return pd.Series(dates, name="trade_date")


def detect_worst_splits(
    summary_df: pd.DataFrame, n: int = 2
) -> List[int]:
    """自动检测表现最差的 n 个 splits

    Args:
        summary_df: walk-forward summary
        n: 返回最差的n个split

    Returns:
        最差split的索引列表
    """
    worst = summary_df.nsmallest(n, "bt_total_return")["split_index"].astype(int).tolist()
    return sorted(worst)


# ── 分析模块 ────────────────────────────────────────────────────


def analyze_split_performance(
    summary_df: pd.DataFrame, focus_splits: List[int]
) -> pd.DataFrame:
    """Split级别对比分析

    对比聚焦splits与全部splits的关键指标。
    """
    cols = [
        "split_index", "test_start", "test_end",
        "daily_rankic_mean", "daily_rankic_ir",
        "top30_return_mean", "top100_return_mean", "top300_return_mean",
        "bt_total_return", "bt_annual_return", "bt_max_drawdown",
        "bt_volatility", "bt_sharpe", "bt_calmar",
    ]
    available_cols = [c for c in cols if c in summary_df.columns]
    df = summary_df[available_cols].copy()
    df["is_focus"] = df["split_index"].isin(focus_splits)

    # 添加统计汇总行
    focus_mask = df["is_focus"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "split_index"]

    stats_rows = []
    for label, mask in [("聚焦期均值", focus_mask), ("非聚焦期均值", ~focus_mask), ("全部均值", None)]:
        subset = df[mask] if mask is not None else df
        row = {"split_index": label, "is_focus": ""}
        for c in numeric_cols:
            row[c] = subset[c].mean()
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    result = pd.concat([df, stats_df], ignore_index=True)

    logger.info("Split级别对比分析完成")
    return result


def analyze_signal_quality(
    summary_df: pd.DataFrame, focus_splits: List[int]
) -> pd.DataFrame:
    """信号质量诊断

    对比聚焦期与非聚焦期的IC、TopK收益、Alpha等。
    同时分析信号收益与回测收益之间的转化效率。
    """
    records = []
    for _, row in summary_df.iterrows():
        split_idx = int(row["split_index"])
        mkt_ret = row.get("diagnostic_全市场收益_逐日均值的均值", 0)
        top30_ret = row.get("diagnostic_Top30_逐日均值的均值", 0)
        top30_lift = row.get("diagnostic_Top30_相对全市场提升_均值", 0)
        bt_return = row.get("bt_total_return", 0)
        trading_days = row.get("bt_trading_days", 120)

        # 信号隐含收益（Top30日均 × 交易日数）vs 回测实际收益
        signal_implied_return = top30_ret * trading_days
        conversion_gap = bt_return - signal_implied_return

        records.append({
            "split_index": split_idx,
            "test_start": str(int(row["test_start"])),
            "test_end": str(int(row["test_end"])),
            "is_focus": split_idx in focus_splits,
            "daily_rankic_mean": row.get("daily_rankic_mean", np.nan),
            "daily_rankic_ir": row.get("daily_rankic_ir", np.nan),
            "top30_return_mean": row.get("top30_return_mean", np.nan),
            "top100_return_mean": row.get("top100_return_mean", np.nan),
            "mkt_return_mean": mkt_ret,
            "top30_alpha": top30_ret - mkt_ret if not np.isnan(top30_ret) else np.nan,
            "top30_lift": top30_lift,
            "signal_implied_return": signal_implied_return,
            "bt_total_return": bt_return,
            "conversion_gap": conversion_gap,
        })

    df = pd.DataFrame(records)

    # 汇总
    focus_df = df[df["is_focus"]]
    other_df = df[~df["is_focus"]]
    numeric_cols = ["daily_rankic_mean", "daily_rankic_ir", "top30_return_mean",
                    "top100_return_mean", "mkt_return_mean", "top30_alpha", "top30_lift",
                    "signal_implied_return", "bt_total_return", "conversion_gap"]

    summary_rows = []
    for label, subset in [("聚焦期均值", focus_df), ("非聚焦期均值", other_df)]:
        row = {"split_index": label, "is_focus": ""}
        for c in numeric_cols:
            row[c] = subset[c].mean()
        summary_rows.append(row)

    # 变化率
    delta_row = {"split_index": "变化率(%)", "is_focus": ""}
    for c in numeric_cols:
        base = other_df[c].mean()
        focus_val = focus_df[c].mean()
        if base != 0:
            delta_row[c] = (focus_val - base) / abs(base) * 100
        else:
            delta_row[c] = np.nan
    summary_rows.append(delta_row)

    result = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    logger.info("信号质量诊断完成")
    return result


def analyze_market_environment(
    summary_df: pd.DataFrame, focus_splits: List[int]
) -> pd.DataFrame:
    """市场环境对比分析

    分析beta拖累 vs alpha失效。
    """
    records = []
    for _, row in summary_df.iterrows():
        split_idx = int(row["split_index"])
        mkt_ret = row.get("diagnostic_全市场收益_逐日均值的均值", 0)
        mkt_vol = row.get("diagnostic_全市场收益_逐日均值的标准差", 0)
        mkt_cross_vol = row.get("diagnostic_全市场收益_逐日标准差的均值", 0)
        top30_ret = row.get("diagnostic_Top30_逐日均值的均值", 0)
        bt_return = row.get("bt_total_return", 0)

        # 半年期总市场收益（近似）
        test_start = str(int(row["test_start"]))
        test_end = str(int(row["test_end"]))
        trading_days = row.get("bt_trading_days", 120)

        mkt_period_return = mkt_ret * trading_days
        alpha_period = (top30_ret - mkt_ret) * trading_days

        records.append({
            "split_index": split_idx,
            "test_start": test_start,
            "test_end": test_end,
            "is_focus": split_idx in focus_splits,
            "mkt_daily_return": mkt_ret,
            "mkt_daily_vol": mkt_vol,
            "mkt_cross_vol": mkt_cross_vol,
            "mkt_period_return_approx": mkt_period_return,
            "alpha_daily": top30_ret - mkt_ret,
            "alpha_period_approx": alpha_period,
            "bt_total_return": bt_return,
            "trading_days": trading_days,
        })

    df = pd.DataFrame(records)

    # 判断：聚焦期的回撤主要来自市场beta还是alpha失效
    focus_df = df[df["is_focus"]]
    other_df = df[~df["is_focus"]]

    summary_rows = []
    numeric_cols = ["mkt_daily_return", "mkt_daily_vol", "mkt_cross_vol",
                    "mkt_period_return_approx", "alpha_daily", "alpha_period_approx",
                    "bt_total_return"]
    for label, subset in [("聚焦期均值", focus_df), ("非聚焦期均值", other_df)]:
        row = {"split_index": label, "is_focus": ""}
        for c in numeric_cols:
            row[c] = subset[c].mean()
        summary_rows.append(row)

    result = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    logger.info("市场环境分析完成")
    return result


def analyze_nav_drawdown(
    chain_nav_df: pd.DataFrame, trade_dates: pd.Series, summary_df: pd.DataFrame
) -> pd.DataFrame:
    """净值回撤详细分析

    计算滚动最大回撤，识别所有显著回撤段。
    """
    nav = chain_nav_df["nav"].values
    cummax = np.maximum.accumulate(nav)
    drawdown = (nav - cummax) / cummax

    # 识别所有回撤段（回撤>10%的段）
    threshold = -0.10
    segments = []
    in_dd = False
    start_idx = 0

    for i in range(len(drawdown)):
        if drawdown[i] < threshold and not in_dd:
            # 寻找此次回撤的真正起点（cummax开始下降的位置）
            start_idx = i
            for j in range(i, -1, -1):
                if drawdown[j] == 0:
                    start_idx = j
                    break
            in_dd = True
        elif drawdown[i] == 0 and in_dd:
            # 回撤恢复
            trough_idx = start_idx + np.argmin(drawdown[start_idx:i + 1])
            segments.append({
                "start_idx": start_idx,
                "trough_idx": trough_idx,
                "end_idx": i,
                "start_date": trade_dates.iloc[start_idx] if start_idx < len(trade_dates) else "",
                "trough_date": trade_dates.iloc[trough_idx] if trough_idx < len(trade_dates) else "",
                "end_date": trade_dates.iloc[i] if i < len(trade_dates) else "",
                "max_drawdown": drawdown[trough_idx],
                "peak_nav": cummax[start_idx],
                "trough_nav": nav[trough_idx],
                "drawdown_days": trough_idx - start_idx,
                "recovery_days": i - trough_idx,
                "total_days": i - start_idx,
                "split_at_trough": int(chain_nav_df.iloc[trough_idx]["split_index"]),
            })
            in_dd = False

    # 处理尾部未恢复的回撤
    if in_dd:
        trough_idx = start_idx + np.argmin(drawdown[start_idx:])
        segments.append({
            "start_idx": start_idx,
            "trough_idx": trough_idx,
            "end_idx": len(drawdown) - 1,
            "start_date": trade_dates.iloc[start_idx] if start_idx < len(trade_dates) else "",
            "trough_date": trade_dates.iloc[trough_idx] if trough_idx < len(trade_dates) else "",
            "end_date": "未恢复",
            "max_drawdown": drawdown[trough_idx],
            "peak_nav": cummax[start_idx],
            "trough_nav": nav[trough_idx],
            "drawdown_days": trough_idx - start_idx,
            "recovery_days": -1,
            "total_days": len(drawdown) - 1 - start_idx,
            "split_at_trough": int(chain_nav_df.iloc[trough_idx]["split_index"]),
        })

    df = pd.DataFrame(segments)
    if not df.empty:
        df = df.sort_values("max_drawdown").reset_index(drop=True)

    logger.info(f"净值回撤分析完成，发现 {len(segments)} 段显著回撤（>10%）")
    return df


# ── 图表输出 ────────────────────────────────────────────────────


def plot_chain_nav(
    chain_nav_df: pd.DataFrame,
    trade_dates: pd.Series,
    summary_df: pd.DataFrame,
    focus_splits: List[int],
    output_path: Path,
) -> None:
    """绘制全周期净值曲线 + split分界线 + 聚焦区间高亮"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1], sharex=True)

    nav = chain_nav_df["nav"].values
    x = np.arange(len(nav))

    # 上图：净值曲线
    ax1 = axes[0]
    ax1.plot(x, nav, color="#2196F3", linewidth=1.2, label="组合净值")

    # 高亮聚焦区间
    for split_idx in focus_splits:
        mask = chain_nav_df["split_index"] == split_idx
        if mask.any():
            indices = np.where(mask.values)[0]
            ax1.axvspan(indices[0], indices[-1], alpha=0.15, color="red", label=f"聚焦 Split {split_idx}")

    # split分界线
    split_boundaries = []
    for split_idx in summary_df["split_index"].unique():
        mask = chain_nav_df["split_index"] == split_idx
        if mask.any():
            first_idx = np.where(mask.values)[0][0]
            split_boundaries.append((first_idx, split_idx))

    for pos, split_idx in split_boundaries:
        ax1.axvline(x=pos, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        # 获取该split的test_start日期作为标签
        row = summary_df[summary_df["split_index"] == split_idx]
        if not row.empty:
            label = str(int(row.iloc[0]["test_start"]))
            ax1.text(pos, ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else nav.min() * 0.95,
                     f"S{int(split_idx)}\n{label[:4]}.{label[4:6]}",
                     fontsize=7, ha="left", va="bottom", color="gray")

    ax1.set_ylabel("净值", fontsize=12)
    ax1.set_title("Walk-Forward 全周期净值曲线", fontsize=14)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 下图：回撤曲线
    ax2 = axes[1]
    cummax = np.maximum.accumulate(nav)
    drawdown = (nav - cummax) / cummax * 100
    ax2.fill_between(x, drawdown, 0, color="red", alpha=0.3)
    ax2.plot(x, drawdown, color="red", linewidth=0.8)

    for split_idx in focus_splits:
        mask = chain_nav_df["split_index"] == split_idx
        if mask.any():
            indices = np.where(mask.values)[0]
            ax2.axvspan(indices[0], indices[-1], alpha=0.15, color="red")

    ax2.set_ylabel("回撤 (%)", fontsize=12)
    ax2.set_xlabel("交易日序号", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"净值曲线图已保存: {output_path}")


def plot_split_metrics(
    summary_df: pd.DataFrame, focus_splits: List[int], output_path: Path
) -> None:
    """绘制各split的关键指标柱状图对比"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    splits = summary_df["split_index"].values
    colors = ["red" if s in focus_splits else "#2196F3" for s in splits]
    x = np.arange(len(splits))
    labels = [f"S{int(s)}" for s in splits]

    metrics = [
        ("daily_rankic_ir", "RankIC IR", axes[0, 0]),
        ("bt_total_return", "回测总收益", axes[0, 1]),
        ("bt_max_drawdown", "最大回撤", axes[1, 0]),
        ("bt_sharpe", "夏普比率", axes[1, 1]),
    ]

    for col, title, ax in metrics:
        if col not in summary_df.columns:
            continue
        vals = summary_df[col].values
        ax.bar(x, vals, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        # 均值参考线
        ax.axhline(y=np.mean(vals), color="gray", linestyle="--", linewidth=1, alpha=0.7)

    fig.suptitle("各Split关键指标对比（红色=聚焦期）", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Split指标对比图已保存: {output_path}")


def plot_alpha_vs_market(
    summary_df: pd.DataFrame, focus_splits: List[int], output_path: Path
) -> None:
    """绘制每split的超额收益 vs 市场收益散点图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    mkt_col = "diagnostic_全市场收益_逐日均值的均值"
    top30_col = "diagnostic_Top30_逐日均值的均值"

    if mkt_col not in summary_df.columns or top30_col not in summary_df.columns:
        logger.warning("缺少必要的diagnostic字段，跳过alpha vs market图")
        plt.close(fig)
        return

    for _, row in summary_df.iterrows():
        split_idx = int(row["split_index"])
        mkt = row[mkt_col] * 10000  # 转为bps
        alpha = (row[top30_col] - row[mkt_col]) * 10000
        color = "red" if split_idx in focus_splits else "#2196F3"
        marker = "s" if split_idx in focus_splits else "o"
        size = 120 if split_idx in focus_splits else 80
        ax.scatter(mkt, alpha, c=color, s=size, marker=marker, zorder=5, edgecolors="white")
        ax.annotate(f"S{split_idx}", (mkt, alpha), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")

    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("市场日均收益 (bps)", fontsize=12)
    ax.set_ylabel("日均超额收益 Alpha (bps)", fontsize=12)
    ax.set_title("各Split: Alpha vs 市场收益（红色方块=聚焦期）", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Alpha vs Market图已保存: {output_path}")


# ── 归因报告 ────────────────────────────────────────────────────


def generate_summary_report(
    summary_df: pd.DataFrame,
    focus_splits: List[int],
    split_perf: pd.DataFrame,
    signal_quality: pd.DataFrame,
    market_env: pd.DataFrame,
    nav_dd: pd.DataFrame,
    output_path: Path,
) -> None:
    """生成文字归因报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("回撤归因分析报告")
    lines.append("=" * 70)
    lines.append("")

    # 基本信息
    focus_rows = summary_df[summary_df["split_index"].isin(focus_splits)]
    lines.append(f"聚焦 Splits: {focus_splits}")
    for _, row in focus_rows.iterrows():
        lines.append(
            f"  Split {int(row['split_index'])}: "
            f"{str(int(row['test_start']))} ~ {str(int(row['test_end']))}, "
            f"收益={row['bt_total_return']*100:.1f}%, "
            f"回撤={row['bt_max_drawdown']*100:.1f}%, "
            f"IC_IR={row.get('daily_rankic_ir', 0):.2f}"
        )
    lines.append("")

    # 1. 信号质量分析
    lines.append("-" * 50)
    lines.append("一、信号质量分析")
    lines.append("-" * 50)

    sq = signal_quality
    focus_sq = sq[sq["is_focus"] == True]  # noqa: E712
    other_sq = sq[sq["is_focus"] == False]  # noqa: E712

    if not focus_sq.empty and not other_sq.empty:
        ic_focus = focus_sq["daily_rankic_ir"].mean()
        ic_other = other_sq["daily_rankic_ir"].mean()
        top30_focus = focus_sq["top30_return_mean"].mean()
        top30_other = other_sq["top30_return_mean"].mean()
        alpha_focus = focus_sq["top30_alpha"].mean()
        alpha_other = other_sq["top30_alpha"].mean()

        lines.append(f"  IC_IR: 聚焦期={ic_focus:.3f}, 非聚焦期={ic_other:.3f}, "
                     f"变化={((ic_focus/ic_other)-1)*100:.1f}%")
        lines.append(f"  Top30日均收益: 聚焦期={top30_focus*10000:.1f}bps, "
                     f"非聚焦期={top30_other*10000:.1f}bps")
        lines.append(f"  日均Alpha: 聚焦期={alpha_focus*10000:.1f}bps, "
                     f"非聚焦期={alpha_other*10000:.1f}bps")

        if ic_focus < ic_other * 0.7:
            lines.append("  >> 结论：信号质量显著下降，IC_IR降幅>30%")
        elif ic_focus < ic_other * 0.9:
            lines.append("  >> 结论：信号质量有所下降，但降幅在合理范围")
        else:
            lines.append("  >> 结论：信号质量基本稳定，回撤非IC下降导致")

        # 信号-回测转化效率分析
        lines.append("")
        lines.append("  【信号→回测转化效率】")
        sig_implied_focus = focus_sq["signal_implied_return"].mean()
        bt_focus = focus_sq["bt_total_return"].mean()
        gap_focus = focus_sq["conversion_gap"].mean()
        sig_implied_other = other_sq["signal_implied_return"].mean()
        bt_other = other_sq["bt_total_return"].mean()
        gap_other = other_sq["conversion_gap"].mean()

        lines.append(f"  聚焦期: 信号隐含收益={sig_implied_focus*100:.1f}%, "
                     f"回测实际={bt_focus*100:.1f}%, 转化损耗={gap_focus*100:.1f}%")
        lines.append(f"  非聚焦期: 信号隐含收益={sig_implied_other*100:.1f}%, "
                     f"回测实际={bt_other*100:.1f}%, 转化损耗={gap_other*100:.1f}%")

        if abs(gap_focus) > abs(gap_other) * 1.5:
            lines.append("  >> 结论：聚焦期信号→收益的转化损耗显著增大，"
                         "交易执行层面（调仓频率/成本/持有期/止损）是重要拖累因素")
        elif sig_implied_focus > 0 and bt_focus < 0:
            lines.append("  >> 结论：信号隐含收益为正但回测为负，"
                         "选股信号到实际收益存在严重转化障碍")

    lines.append("")

    # 2. 市场环境分析
    lines.append("-" * 50)
    lines.append("二、市场环境分析")
    lines.append("-" * 50)

    me = market_env
    focus_me = me[me["is_focus"] == True]  # noqa: E712
    other_me = me[me["is_focus"] == False]  # noqa: E712

    if not focus_me.empty and not other_me.empty:
        mkt_focus = focus_me["mkt_daily_return"].mean()
        mkt_other = other_me["mkt_daily_return"].mean()
        vol_focus = focus_me["mkt_daily_vol"].mean()
        vol_other = other_me["mkt_daily_vol"].mean()
        mkt_period_focus = focus_me["mkt_period_return_approx"].mean()
        mkt_period_other = other_me["mkt_period_return_approx"].mean()

        lines.append(f"  市场日均收益: 聚焦期={mkt_focus*10000:.1f}bps, "
                     f"非聚焦期={mkt_other*10000:.1f}bps")
        lines.append(f"  市场日均波动: 聚焦期={vol_focus*10000:.1f}bps, "
                     f"非聚焦期={vol_other*10000:.1f}bps")
        lines.append(f"  半年期市场收益(近似): 聚焦期={mkt_period_focus*100:.1f}%, "
                     f"非聚焦期={mkt_period_other*100:.1f}%")

        if mkt_focus < -0.0002:
            lines.append("  >> 结论：聚焦期市场环境明显偏空，存在系统性beta拖累")
        elif mkt_focus < mkt_other * 0.5:
            lines.append("  >> 结论：聚焦期市场偏弱，对策略产生一定拖累")
        else:
            lines.append("  >> 结论：市场环境无明显恶化，回撤主因非市场beta")

    lines.append("")

    # 3. 综合归因
    lines.append("-" * 50)
    lines.append("三、综合归因结论")
    lines.append("-" * 50)

    # 判断逻辑
    alpha_degraded = False
    market_weak = False
    conversion_problem = False
    if not focus_sq.empty and not other_sq.empty:
        alpha_degraded = focus_sq["top30_alpha"].mean() < other_sq["top30_alpha"].mean() * 0.5
        # 信号隐含正收益但回测为负 → 转化问题
        sig_implied = focus_sq["signal_implied_return"].mean()
        bt_actual = focus_sq["bt_total_return"].mean()
        gap_focus = abs(focus_sq["conversion_gap"].mean())
        gap_other = abs(other_sq["conversion_gap"].mean())
        conversion_problem = (sig_implied > 0 and bt_actual < 0) or (gap_focus > gap_other * 1.5)
    if not focus_me.empty:
        market_weak = focus_me["mkt_daily_return"].mean() < -0.0001

    if conversion_problem:
        conclusion = "信号→收益转化障碍为主因"
        lines.append(f"  主因: {conclusion}")
        lines.append("  选股信号（IC/TopK）本身并未严重恶化，但信号到回测收益的转化出现了严重损耗。")
        lines.append("  可能原因:")
        lines.append("    1) 调仓频率与信号衰减速度不匹配（信号有效期短于持有期）")
        lines.append("    2) 交易成本和滑点在高换手率下侵蚀收益")
        lines.append("    3) 止损频繁触发但效果不佳（止损后反弹踏空）")
        lines.append("    4) 涨跌停/停牌导致无法按信号执行买卖")
        lines.append("  建议:")
        lines.append("    - 分析调仓频率与信号IC衰减的关系，优化调仓节奏")
        lines.append("    - 检查该时段的换手率和交易成本占比")
        lines.append("    - 评估止损触发后个股的后续表现（误止损率）")
        lines.append("    - 考虑动态Top-N和自适应调仓策略")
    elif alpha_degraded and market_weak:
        conclusion = "Alpha失效 + 市场下跌 双重叠加"
        lines.append(f"  主因: {conclusion}")
        lines.append("  聚焦期Alpha显著下降，同时市场整体偏弱，两者叠加导致较大回撤。")
        lines.append("  建议: 优先优化信号稳定性（因子筛选/模型泛化），同时考虑加强择时保护。")
    elif alpha_degraded:
        conclusion = "Alpha失效为主因"
        lines.append(f"  主因: {conclusion}")
        lines.append("  市场环境尚可，但选股Alpha显著下降，信号失效是回撤主要来源。")
        lines.append("  建议: 聚焦因子优化和模型泛化能力提升。")
    elif market_weak:
        conclusion = "市场系统性下跌为主因"
        lines.append(f"  主因: {conclusion}")
        lines.append("  Alpha基本稳定，但市场整体下跌拖累了绝对收益。")
        lines.append("  建议: 加强市场择时和仓位管理，如自适应调仓/ECT优化。")
    else:
        conclusion = "需进一步排查"
        lines.append(f"  主因: {conclusion}")
        lines.append("  Alpha和市场均无明显恶化，回撤可能来自其他因素。")
        lines.append("  建议: 需增加交易明细数据做更细粒度的归因。")

    lines.append("")

    # 4. 回撤段详情
    if not nav_dd.empty:
        lines.append("-" * 50)
        lines.append("四、显著回撤段（>10%）")
        lines.append("-" * 50)
        for _, row in nav_dd.iterrows():
            recovery = f"{int(row['recovery_days'])}日" if row["recovery_days"] > 0 else "未恢复"
            lines.append(
                f"  {row['start_date']} ~ {row['trough_date']}: "
                f"回撤={row['max_drawdown']*100:.1f}%, "
                f"净值 {row['peak_nav']:.3f} → {row['trough_nav']:.3f}, "
                f"下跌{int(row['drawdown_days'])}日, 恢复{recovery}, "
                f"所在Split={int(row['split_at_trough'])}"
            )

    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"归因报告已保存: {output_path}")

    # 同时打印到控制台
    print("\n" + report)


# ── 主函数 ──────────────────────────────────────────────────────


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="回撤归因分析")
    parser.add_argument(
        "--wf-run-id", required=True,
        help="Walk-forward 运行ID，如 wf_20260325_120126_f641fee9"
    )
    parser.add_argument(
        "--focus-splits", default=None,
        help="聚焦分析的split索引，逗号分隔，如 9,10。不指定则自动检测最差2个"
    )
    parser.add_argument(
        "--output-dir", default="data/reports/drawdown_attribution",
        help="输出目录（默认: data/reports/drawdown_attribution）"
    )
    args = parser.parse_args()

    setup_logger(log_level="INFO")

    # 初始化
    config = Config()
    data_root = Path(config.get("data.root", "data"))
    storage = Storage(root_path=str(data_root))
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    logger.info(f"开始回撤归因分析，WF Run ID: {args.wf_run_id}")
    summary_df, chain_nav_df = load_wf_data(args.wf_run_id, data_root)

    # 2. 确定聚焦splits
    if args.focus_splits:
        focus_splits = [int(s.strip()) for s in args.focus_splits.split(",")]
    else:
        focus_splits = detect_worst_splits(summary_df)
    logger.info(f"聚焦 Splits: {focus_splits}")

    # 3. 构建交易日期
    trade_dates = build_trade_dates(summary_df, chain_nav_df, storage)

    # 4. 运行各分析模块
    split_perf = analyze_split_performance(summary_df, focus_splits)
    signal_quality = analyze_signal_quality(summary_df, focus_splits)
    market_env = analyze_market_environment(summary_df, focus_splits)
    nav_dd = analyze_nav_drawdown(chain_nav_df, trade_dates, summary_df)

    # 5. 保存CSV
    split_perf.to_csv(output_dir / "split_comparison.csv", index=False, encoding="utf-8-sig")
    signal_quality.to_csv(output_dir / "signal_quality.csv", index=False, encoding="utf-8-sig")
    market_env.to_csv(output_dir / "market_environment.csv", index=False, encoding="utf-8-sig")
    nav_dd.to_csv(output_dir / "nav_drawdown_detail.csv", index=False, encoding="utf-8-sig")
    logger.info("CSV文件已保存")

    # 6. 生成图表
    plot_chain_nav(chain_nav_df, trade_dates, summary_df, focus_splits,
                   output_dir / "fig_chain_nav.png")
    plot_split_metrics(summary_df, focus_splits, output_dir / "fig_split_metrics.png")
    plot_alpha_vs_market(summary_df, focus_splits, output_dir / "fig_alpha_vs_market.png")

    # 7. 生成文字报告
    generate_summary_report(
        summary_df, focus_splits, split_perf, signal_quality,
        market_env, nav_dd, output_dir / "attribution_report.txt"
    )

    logger.info(f"回撤归因分析完成，所有输出位于: {output_dir}")


if __name__ == "__main__":
    main()

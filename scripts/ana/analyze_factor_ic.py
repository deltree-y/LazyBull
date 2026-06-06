#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单因子IC分析脚本

对特征截面数据中的每个因子，计算其在不同标签horizon上的RankIC/ICIR，
并支持行业中性化、市值中性化的对比分析。

功能：
- 基础模式：所有因子的IC汇总（IC_mean, IC_std, ICIR, IC_win_rate）
- 衰减模式：指定因子的IC跨horizon衰减曲线
- 对比模式：raw / industry / size / both 四种中性化模式对比
- 自动识别因子列和标签列

输出：
- 终端报告：Top/Bottom因子排名、IC分布统计
- CSV文件：完整因子IC表 + 分类统计

使用示例：
    # 基础分析（默认 raw 模式）
    python scripts/ana/analyze_factor_ic.py --start 20200101 --end 20231231

    # 行业+市值双重中性化
    python scripts/ana/analyze_factor_ic.py --start 20200101 --end 20231231 --neutralize both

    # 对比四种中性化模式
    python scripts/ana/analyze_factor_ic.py --start 20200101 --end 20231231 --compare

    # 指定因子的衰减曲线分析
    python scripts/ana/analyze_factor_ic.py --start 20200101 --end 20231231 --decay neu_ret_20

    # 仅分析特定因子列表
    python scripts/ana/analyze_factor_ic.py --start 20200101 --end 20231231 --factors neu_ret_20,zscore_bp,roe_waa
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_data_root, get_reports_root
from src.lazybull.common.logger import setup_logger
from src.lazybull.factors.factor_evaluation import (
    auto_detect_factor_columns,
    auto_detect_label_columns,
    compare_neutralization_modes,
    compute_factor_ic_series,
    compute_factor_ic_summary,
    compute_ic_decay,
    evaluate_all_factors,
)


# ── 终端报告 ──────────────────────────────────────────────────────


def print_summary_report(
    result_df: pd.DataFrame,
    top_n: int = 20,
    neutralize_label: str = "raw",
) -> None:
    """打印因子IC分析汇总报告

    Args:
        result_df: evaluate_all_factors 的输出
        top_n: 显示的头部/尾部因子数量
        neutralize_label: 中性化模式标签
    """
    if result_df.empty:
        print("\n⚠ 无有效IC数据，请检查数据范围是否包含足够的交易日。")
        return

    print("\n" + "=" * 100)
    print(f"                   单因子 IC 分析报告（中性化: {neutralize_label}）")
    print("=" * 100)

    # 基本信息
    n_factors = result_df["factor"].nunique()
    n_labels = result_df["label"].nunique()
    print(f"\n因子数: {n_factors}  |  标签数: {n_labels}  |  记录数: {len(result_df)}")

    # ICIR 分布
    icir_values = result_df["ICIR"].dropna()
    if len(icir_values) > 0:
        print(
            f"ICIR 分布:  min={icir_values.min():.3f}  "
            f"P25={np.percentile(icir_values, 25):.3f}  "
            f"median={np.median(icir_values):.3f}  "
            f"P75={np.percentile(icir_values, 75):.3f}  "
            f"max={icir_values.max():.3f}"
        )
        print(
            f"IC均值分布: min={result_df['IC_mean'].min():.4f}  "
            f"median={result_df['IC_mean'].median():.4f}  "
            f"max={result_df['IC_mean'].max():.4f}"
        )
        n_positive = (icir_values > 0).sum()
        print(f"ICIR > 0 的因子占比: {n_positive}/{len(icir_values)} ({n_positive/len(icir_values):.0%})")

    # 按标签分别展示
    for label in sorted(result_df["label"].unique()):
        label_df = result_df[result_df["label"] == label].copy()
        if label_df.empty:
            continue

        horizon_days = int(label_df["horizon_days"].iloc[0])
        print(f"\n{'─' * 100}")
        print(f"  ▸ 标签: {label} (未来{horizon_days}日收益)  |  {len(label_df)} 个因子")

        # Top N
        top = label_df.head(top_n)
        print(f"\n  ── Top {top_n} 因子（按ICIR降序）" + "─" * 55)
        _print_factor_table(top, show_rank=True)

        # Bottom N
        bottom = label_df.tail(top_n).iloc[::-1]
        print(f"\n  ── Bottom {top_n} 因子（按ICIR升序）" + "─" * 52)
        _print_factor_table(bottom, show_rank=False)


def _print_factor_table(df: pd.DataFrame, show_rank: bool = True) -> None:
    """打印因子IC表格"""
    cols = ["factor", "IC_mean", "IC_std", "ICIR", "IC_win_rate", "n_days_valid"]
    display = df[cols].copy()
    display["IC_mean"] = display["IC_mean"].map(lambda x: f"{x:.4f}")
    display["IC_std"] = display["IC_std"].map(lambda x: f"{x:.4f}")
    display["ICIR"] = display["ICIR"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "NaN"
    )
    display["IC_win_rate"] = display["IC_win_rate"].map(
        lambda x: f"{x:.0%}" if pd.notna(x) else "NaN"
    )
    display["n_days_valid"] = display["n_days_valid"].astype(int)
    if show_rank:
        display.insert(0, "排名", range(1, len(display) + 1))
    print(display.to_string(index=False))


def print_decay_report(decay_df: pd.DataFrame, factor_col: str) -> None:
    """打印因子IC衰减报告"""
    if decay_df.empty:
        print(f"\n⚠ 因子 [{factor_col}] 无有效衰减数据")
        return

    print(f"\n{'=' * 80}")
    print(f"  因子 IC 衰减曲线: {factor_col}")
    print(f"{'=' * 80}")

    cols = ["horizon", "IC_mean", "IC_std", "ICIR", "IC_win_rate", "n_days_valid"]
    display = decay_df[cols].copy()
    display["IC_mean"] = display["IC_mean"].map(lambda x: f"{x:.4f}")
    display["IC_std"] = display["IC_std"].map(lambda x: f"{x:.4f}")
    display["ICIR"] = display["ICIR"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
    display["IC_win_rate"] = display["IC_win_rate"].map(
        lambda x: f"{x:.0%}" if pd.notna(x) else "NaN"
    )
    display["n_days_valid"] = display["n_days_valid"].astype(int)
    print(display.to_string(index=False))

    # 判断衰减类型
    icir_series = decay_df["ICIR"].dropna()
    if len(icir_series) >= 2:
        if icir_series.iloc[-1] > icir_series.iloc[0] * 0.8:
            print("  → 判断: 因子IC随horizon延长保持稳定，适合中长期持有")
        elif icir_series.iloc[-1] > 0:
            print("  → 判断: 因子IC随horizon延长逐渐衰减，建议匹配较短持有期")
        else:
            print("  → 判断: 因子在长horizon上IC转负，仅适合短期持有")


def print_compare_report(combined_df: pd.DataFrame) -> None:
    """打印中性化模式对比报告"""
    if combined_df.empty:
        return

    print("\n" + "=" * 100)
    print("                    中性化模式对比：Raw vs Industry vs Size vs Both")
    print("=" * 100)

    # 按因子聚合，展示每种模式下的ICIR
    pivot = combined_df.pivot_table(
        index=["factor", "label"],
        columns="neutralize",
        values="ICIR",
        aggfunc="first",
    )

    if pivot.empty:
        print("\n⚠ 无可对比数据")
        return

    # 添加ICIR变化列
    expected_modes = ["raw", "industry", "size", "ind_size"]
    available_modes = [m for m in expected_modes if m in pivot.columns]

    for mode in available_modes:
        if mode == "raw":
            continue
        if "raw" in pivot.columns and mode in pivot.columns:
            pivot[f"{mode}_delta"] = pivot[mode] - pivot["raw"]

    # 找出行业暴露最大的因子（raw ICIR高但industry中性化后下降最多）
    if "industry" in pivot.columns and "raw" in pivot.columns:
        pivot["industry_drop"] = pivot["raw"] - pivot["industry"]
        top_industry_dependent = pivot.nlargest(10, "industry_drop")
        print(f"\n── 行业暴露最大的10个因子（Raw IC高但行业中性化后大幅下降）──")
        _print_compare_subset(top_industry_dependent, available_modes)

    # 找出市值暴露最大的因子
    if "size" in pivot.columns and "raw" in pivot.columns:
        pivot["size_drop"] = pivot["raw"] - pivot["size"]
        top_size_dependent = pivot.nlargest(10, "size_drop")
        print(f"\n── 市值暴露最大的10个因子（Raw IC高但市值中性化后大幅下降）──")
        _print_compare_subset(top_size_dependent, available_modes)

    # 找出真正的alpha因子（所有中性化后ICIR仍 > 0.3）
    if all(m in pivot.columns for m in ["industry", "size"]):
        true_alpha = pivot[
            (pivot["industry"] > 0.3) & (pivot["size"] > 0.3)
        ]
        if len(true_alpha) > 0:
            print(f"\n── 真正的Alpha因子（行业+市值中性化后ICIR仍 > 0.3）: {len(true_alpha)} 个 ──")
            _print_compare_subset(true_alpha.nlargest(10, "industry"), available_modes)
        else:
            print("\n── 真正的Alpha因子: 0 个（无不依赖行业/市值的稳定因子）──")


def _print_compare_subset(df: pd.DataFrame, available_modes: List[str]) -> None:
    """打印对比子集的表格"""
    cols = available_modes + [c for c in df.columns if c.endswith("_delta") or c.endswith("_drop")]
    display = df.reset_index()[["factor", "label"] + cols].copy()
    for col in cols:
        display[col] = display[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
    print(display.to_string(index=False))


# ── 数据加载 ──────────────────────────────────────────────────────


def load_features_data(
    data_root: Path,
    start_date: str,
    end_date: str,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """加载指定日期范围的特征截面数据

    Args:
        data_root: 数据根目录
        start_date: 起始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        columns: 仅加载指定列，None则加载全部

    Returns:
        合并后的多日截面DataFrame
    """
    features_dir = data_root / "features" / "cs_train"

    if not features_dir.exists():
        logger.error(f"特征目录不存在: {features_dir}")
        sys.exit(1)

    # 收集日期范围内的文件
    date_files = sorted(
        [f for f in features_dir.iterdir() if f.suffix == ".parquet"]
    )

    filtered_files = []
    for f in date_files:
        date_str = f.stem  # YYYYMMDD
        if start_date <= date_str <= end_date:
            filtered_files.append(f)

    if not filtered_files:
        logger.error(
            f"在 {start_date} ~ {end_date} 范围内未找到特征文件。"
            f"目录: {features_dir}"
        )
        sys.exit(1)

    logger.info(
        f"加载特征数据: {start_date} ~ {end_date}, "
        f"共 {len(filtered_files)} 个交易日"
    )

    # 批量加载
    dfs = []
    for f in filtered_files:
        try:
            if columns:
                df = pd.read_parquet(f, columns=columns)
            else:
                df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"加载 {f.name} 失败: {e}")

    if not dfs:
        logger.error("未能成功加载任何特征文件")
        sys.exit(1)

    data = pd.concat(dfs, ignore_index=True)
    logger.info(f"加载完成: {len(data)} 行, {len(data.columns)} 列")

    return data


# ── 主入口 ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="单因子IC分析 — 计算每个因子在不同标签horizon上的RankIC/ICIR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录；未指定时使用 configs/base.yaml 中的 data.root",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="起始日期 YYYYMMDD",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="结束日期 YYYYMMDD",
    )
    parser.add_argument(
        "--neutralize",
        type=str,
        default="none",
        choices=["none", "industry", "size", "both"],
        help="中性化模式 (默认: none)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比四种中性化模式（raw/industry/size/both），会覆盖 --neutralize",
    )
    parser.add_argument(
        "--decay",
        type=str,
        default=None,
        help="指定因子名称，输出IC衰减曲线（跨 y_ret_5/10/20）",
    )
    parser.add_argument(
        "--factors",
        type=str,
        default=None,
        help="逗号分隔的因子列表，仅分析指定因子（默认: 全部自动检测）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="报告中显示的头部/尾部因子数量 (默认: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV输出路径 (默认: data/reports/factor_ic.csv)",
    )
    parser.add_argument(
        "--columns-only",
        type=str,
        default=None,
        help="仅加载指定列（逗号分隔），减少内存占用，适用于大批量数据",
    )
    args = parser.parse_args()

    setup_logger("INFO")

    # 解析路径
    data_root = Path(args.data_root or get_data_root())
    output_path = args.output or str(
        Path(get_reports_root(str(data_root / "reports") if args.data_root else None))
        / "factor_ic.csv"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 解析因子列表
    factor_list: Optional[List[str]] = None
    if args.factors:
        factor_list = [f.strip() for f in args.factors.split(",")]

    # 解析列过滤
    column_filter: Optional[List[str]] = None
    if args.columns_only:
        column_filter = [c.strip() for c in args.columns_only.split(",")]

    # ── 加载数据 ──
    # 如果指定了因子列表，确保加载对应的因子列 + 必要的元数据列
    load_columns = column_filter
    if load_columns is None and factor_list is not None:
        # 只加载因子列 + 必要列，加速加载
        essential = ["trade_date", "ts_code", "tradable",
                     "sw_l1_code", "sw_l2_code", "sw_l3_code",
                     "log_total_mv", "total_mv"]
        label_cols = ["y_ret_5", "y_ret_10", "y_ret_20"]
        load_columns = list(set(essential + label_cols + factor_list))

    data = load_features_data(data_root, args.start, args.end, columns=load_columns)

    # ── 对比模式 ──
    if args.compare:
        logger.info("启动中性化模式对比分析...")
        combined = compare_neutralization_modes(
            data,
            factor_cols=factor_list,
            label_cols=None,  # 自动检测
        )
        print_compare_report(combined)

        # 保存完整对比结果
        compare_path = output_path.parent / "factor_ic_compare.csv"
        combined.to_csv(compare_path, encoding="utf-8-sig", index=False)
        logger.info(f"对比报告已保存: {compare_path}")

        # 保存 pivot 表
        pivot = combined.pivot_table(
            index=["factor", "label"],
            columns="neutralize",
            values="ICIR",
            aggfunc="first",
        )
        pivot_path = output_path.parent / "factor_ic_compare_pivot.csv"
        pivot.to_csv(pivot_path, encoding="utf-8-sig")
        logger.info(f"对比透视表已保存: {pivot_path}")

        return

    # ── 衰减模式 ──
    if args.decay:
        factor_name = args.decay
        label_cols = auto_detect_label_columns(data)
        if not label_cols:
            logger.error("未检测到标签列")
            sys.exit(1)

        logger.info(f"计算因子 [{factor_name}] 的IC衰减曲线...")
        neutralize_industry = args.neutralize in ("industry", "both")
        neutralize_size = args.neutralize in ("size", "both")

        decay_df = compute_ic_decay(
            data,
            factor_name,
            label_cols,
            neutralize_industry=neutralize_industry,
            neutralize_size=neutralize_size,
        )
        print_decay_report(decay_df, factor_name)

        # 保存
        decay_path = output_path.parent / f"factor_ic_decay_{factor_name}.csv"
        decay_df.to_csv(decay_path, encoding="utf-8-sig", index=False)
        logger.info(f"衰减报告已保存: {decay_path}")
        return

    # ── 基础模式 ──
    neutralize_label = args.neutralize
    neutralize_industry = neutralize_label in ("industry", "both")
    neutralize_size = neutralize_label in ("size", "both")

    logger.info(
        f"启动全因子IC分析: 中性化={neutralize_label}, "
        f"日期范围={args.start}~{args.end}"
    )

    result = evaluate_all_factors(
        data,
        factor_cols=factor_list,
        label_cols=None,
        neutralize_mode=neutralize_label,
        verbose=True,
    )

    if result.empty:
        logger.error("IC评估结果为空，请检查数据")
        sys.exit(1)

    # 打印报告
    print_summary_report(result, top_n=args.top_n, neutralize_label=neutralize_label)

    # 保存CSV
    export_cols = {
        "factor": "因子名称",
        "label": "标签",
        "horizon_days": "预测天数",
        "neutralize": "中性化模式",
        "IC_mean": "IC均值",
        "IC_std": "IC标准差",
        "ICIR": "ICIR",
        "IC_win_rate": "IC胜率",
        "IC_pos_mean": "正IC均值",
        "IC_neg_mean": "负IC均值",
        "IC_abs_mean": "IC绝对值均值",
        "n_days": "总交易日数",
        "n_days_valid": "有效交易日数",
    }
    export_df = result.rename(columns=export_cols)

    # 按标签分别保存
    for label in export_df["标签"].unique():
        label_df = export_df[export_df["标签"] == label].drop(columns=["标签"])
        label_suffix = label.replace("y_ret_", "h")
        label_path = output_path.parent / f"factor_ic_{label_suffix}.csv"
        label_df.to_csv(label_path, encoding="utf-8-sig", index=False)
        logger.info(f"标签 {label} 的报告已保存: {label_path}")

    # 也保存汇总文件
    export_df.to_csv(output_path, encoding="utf-8-sig", index=False)
    logger.info(f"汇总报告已保存: {output_path}")

    # 快速诊断提示
    top_icir = result.dropna(subset=["ICIR"]).head(5)
    if len(top_icir) > 0:
        print(f"\n💡 提示: 使用 --decay {top_icir.iloc[0]['factor']} 查看该因子的IC衰减曲线")
        print(f"💡 提示: 使用 --compare 对比四种中性化模式下因子的真实Alpha")


if __name__ == "__main__":
    main()

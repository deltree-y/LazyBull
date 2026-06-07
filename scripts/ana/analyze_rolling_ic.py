#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
滚动 IC 稳定性分析脚本

功能：
- 计算每个因子的滚动窗口 ICIR（默认 252 日窗口，约1年）
- 评估因子稳定性：均值、波动、趋势、综合评分
- 识别"过期"因子（ICIR 持续衰减）

输出：
- CSV：逐因子逐日的滚动 ICIR 序列 + 稳定性汇总

使用示例：
    # 全量因子滚动 IC 分析（raw 模式）
    python scripts/ana/analyze_rolling_ic.py --start 20200101 --end 20251231

    # 指定因子 + 双重中性化
    python scripts/ana/analyze_rolling_ic.py --start 20200101 --end 20251231 \
        --factors zscore_bp,neu_ret_20,roe_waa --neutralize both

    # 自定义窗口
    python scripts/ana/analyze_rolling_ic.py --start 20200101 --end 20251231 \
        --window 504 --min-periods 252  # 2年窗口
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_data_root, get_reports_root
from src.lazybull.common.logger import setup_logger
from src.lazybull.factors.factor_evaluation import (
    auto_detect_factor_columns,
    compute_rolling_factor_ic,
    evaluate_ic_stability,
)


def load_features_data(data_root, start_date, end_date, columns=None):
    features_dir = Path(data_root) / "features" / "cs_train"
    if not features_dir.exists():
        logger.error(f"特征目录不存在: {features_dir}")
        sys.exit(1)

    date_files = sorted(f for f in features_dir.iterdir() if f.suffix == ".parquet")
    filtered = [f for f in date_files if start_date <= f.stem <= end_date]

    if not filtered:
        logger.error(f"在 {start_date}~{end_date} 范围内未找到特征文件")
        sys.exit(1)

    logger.info(f"加载特征数据: {start_date}~{end_date}, 共 {len(filtered)} 个交易日")
    dfs = []
    for f in filtered:
        try:
            df = pd.read_parquet(f, columns=columns) if columns else pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"加载 {f.name} 失败: {e}")

    data = pd.concat(dfs, ignore_index=True)
    logger.info(f"加载完成: {len(data)} 行, {len(data.columns)} 列")
    return data


def main():
    parser = argparse.ArgumentParser(description="滚动 IC 稳定性分析")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--factors", type=str, default=None, help="逗号分隔的因子列表")
    parser.add_argument("--label", type=str, default="y_ret_20", help="标签列 (默认 y_ret_20)")
    parser.add_argument("--neutralize", type=str, default="none",
                        choices=["none", "industry", "size", "both"])
    parser.add_argument("--window", type=int, default=252, help="滚动窗口 (交易日, 默认 252≈1年)")
    parser.add_argument("--min-periods", type=int, default=126, help="最小窗口 (默认 126≈半年)")
    parser.add_argument("--columns-only", type=str, default=None)
    args = parser.parse_args()

    setup_logger("INFO")
    data_root = args.data_root or get_data_root()

    factor_list = None
    if args.factors:
        factor_list = [f.strip() for f in args.factors.split(",")]

    column_filter = None
    if args.columns_only:
        column_filter = [c.strip() for c in args.columns_only.split(",")]

    # 加载数据
    data = load_features_data(data_root, args.start, args.end, columns=column_filter)
    if factor_list is None:
        factor_list = auto_detect_factor_columns(data)
    else:
        factor_list = [f for f in factor_list if f in data.columns]

    # 中性化参数
    neut_ind = args.neutralize in ("industry", "both")
    neut_size = args.neutralize in ("size", "both")

    logger.info(
        f"滚动 IC 分析: {len(factor_list)} 因子, "
        f"label={args.label}, neutral={args.neutralize}, "
        f"window={args.window}"
    )

    # 1. 计算滚动 IC
    rolling = compute_rolling_factor_ic(
        data,
        factor_cols=factor_list,
        label_col=args.label,
        window=args.window,
        min_periods=args.min_periods,
        neutralize_industry=neut_ind,
        neutralize_size=neut_size,
        verbose=True,
    )

    out_dir = Path(get_reports_root(str(Path(data_root) / "reports")))
    out_dir.mkdir(parents=True, exist_ok=True)

    rolling.to_csv(out_dir / "factor_rolling_ic.csv", encoding="utf-8-sig", index=False)
    logger.info(f"滚动 IC 序列已保存: factor_rolling_ic.csv")

    # 2. 稳定性评估
    stability = evaluate_ic_stability(rolling)
    stability.to_csv(out_dir / "factor_ic_stability.csv", encoding="utf-8-sig", index=False)
    logger.info(f"稳定性评估已保存: factor_ic_stability.csv")

    # 报告
    print(f"\n{'='*80}")
    print(f"  因子 IC 稳定性报告")
    print(f"{'='*80}")
    print(f"  分析因子数: {len(factor_list)}")
    print(f"  窗口: {args.window} 交易日, 标签: {args.label}")

    # Top 15 最稳定
    print(f"\n── Top 15 最稳定因子 ──")
    top = stability.head(15)
    for _, r in top.iterrows():
        trend_mark = "↑" if r["icir_trend"] > 0.05 else ("↓" if r["icir_trend"] < -0.05 else "→")
        print(
            f"  {r['factor']:35s} "
            f"score={r['stability_score']:.3f}  "
            f"mean={r['icir_mean']:.3f}  "
            f"std={r['icir_std']:.3f}  "
            f"trend={r['icir_trend']:+.3f}{trend_mark}"
        )

    # 衰减因子
    decaying = stability[stability["icir_trend"] < -0.1].head(10)
    if len(decaying) > 0:
        print(f"\n── 衰减因子（ICIR 后半段比前半段低 > 0.1）──")
        for _, r in decaying.iterrows():
            print(
                f"  {r['factor']:35s} "
                f"trend={r['icir_trend']:.3f}  "
                f"mean={r['icir_mean']:.3f}  "
                f"min={r['icir_min']:.3f}"
            )

    # 建议
    print(f"\n💡 提示:")
    print(f"  - 关注 icir_trend < -0.1 的衰减因子，考虑从训练中移除")
    print(f"  - stability_score 综合考虑了均值、波动和趋势")
    print(f"  - 详细数据见: factor_rolling_ic.csv / factor_ic_stability.csv")


if __name__ == "__main__":
    main()
